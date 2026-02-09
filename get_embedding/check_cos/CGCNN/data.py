from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
def collate_pool_matbench(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.
    整理数据列表并返回用于预测晶体特性的批次。
    Parameters
    ----------
l
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    原子类型的原子特征
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    每个原子的M个邻居的键特征
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    每个原子的M个邻居的指数
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    从晶体idx 到 原子idx 的映射
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_elem_ids = [], [], [], []
    crystal_atom_idx, batch_target = [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, elem_ids), target) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_elem_ids.append(elem_ids)

        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),       # x[0]
        torch.cat(batch_nbr_fea, dim=0),        # x[1]
        torch.cat(batch_nbr_fea_idx, dim=0),    # x[2]
        torch.cat(batch_elem_ids, dim=0),       # x[3]
        crystal_atom_idx                         # x[4]
    ), torch.stack(batch_target, dim=0)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


# ----------------------------
# 预计算并行化：worker 全局对象
# ----------------------------
_WORKER_ARI = None
_WORKER_GDF = None
_WORKER_MAX_NUM_NBR = None
_WORKER_RADIUS = None


def _init_precompute_worker(atom_init_file: str, radius: float, dmin: float, step: float, max_num_nbr: int):
    """multiprocessing worker initializer: build ARI/GDF once per process."""
    global _WORKER_ARI, _WORKER_GDF, _WORKER_MAX_NUM_NBR, _WORKER_RADIUS

    # 避免多进程 + MKL/OMP 线程过度订阅导致“越并行越慢”
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _WORKER_ARI = AtomCustomJSONInitializer(atom_init_file)
    _WORKER_GDF = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    _WORKER_MAX_NUM_NBR = int(max_num_nbr)
    _WORKER_RADIUS = float(radius)


def _compute_features(
    crystal: Structure,
    target,
    ari: AtomCustomJSONInitializer,
    gdf: GaussianDistance,
    max_num_nbr: int,
    radius: float,
):
    """
    单个 Structure -> (atom_fea, nbr_fea, nbr_fea_idx, elem_ids, target)
    做了一些小优化：预分配 numpy 数组 + from_numpy 避免多余拷贝。
    """
    # ---------- atom features ----------
    # pymatgen Structure 有 atomic_numbers 属性（list[int]）；若没有则 fallback
    try:
        elem_ids = np.asarray(crystal.atomic_numbers, dtype=np.int64)
    except Exception:
        elem_ids = np.asarray([site.specie.number for site in crystal], dtype=np.int64)

    atom_fea = np.stack([ari.get_atom_fea(int(z)) for z in elem_ids], axis=0).astype(np.float32, copy=False)

    # ---------- neighbor features ----------
    # 取半径内所有邻居，之后每个原子取最近 max_num_nbr 个（不足则 padding）
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)

    n_i = len(all_nbrs)
    nbr_fea_idx = np.zeros((n_i, max_num_nbr), dtype=np.int64)
    # padding distance 用 radius+1，和你原代码一致
    nbr_dists = np.full((n_i, max_num_nbr), radius + 1.0, dtype=np.float32)

    for i, nbrs in enumerate(all_nbrs):
        if not nbrs:
            continue
        nbrs = sorted(nbrs, key=lambda x: x[1])
        if len(nbrs) > max_num_nbr:
            nbrs = nbrs[:max_num_nbr]
        # x[2] neighbor index, x[1] distance
        nn = len(nbrs)
        nbr_fea_idx[i, :nn] = [x[2] for x in nbrs]
        nbr_dists[i, :nn] = [x[1] for x in nbrs]

    nbr_fea = gdf.expand(nbr_dists).astype(np.float32, copy=False)

    return (
        atom_fea,  # np.ndarray
        nbr_fea,
        nbr_fea_idx,
        elem_ids,
        np.array([float(target) if target is not None else 0.0], dtype=np.float32)
    )


def _featurize_task(task):
    """Task wrapper for multiprocessing: (idx, crystal, target) -> (idx, features)."""
    idx, crystal, target = task
    out = _compute_features(
        crystal=crystal,
        target=target,
        ari=_WORKER_ARI,
        gdf=_WORKER_GDF,
        max_num_nbr=_WORKER_MAX_NUM_NBR,
        radius=_WORKER_RADIUS,
    )
    return idx, out
class StruData(Dataset):
    """
    预计算特征的数据集。

    核心优化点：
    1) 支持多进程/多线程并行预计算，减少初始化等待时间
    2) 可选的 cache_path：预计算一次后保存，下次秒级加载
    3) 预分配 numpy 数组 + torch.from_numpy，减少 Python/list 和多余拷贝开销
    """

    # 没打乱
    def __init__(
        self,
        input,
        output=None,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        *,
        precompute=True,
        precompute_workers=0,
        precompute_backend="process",  # "process" | "thread"
        chunksize=32,
        cache_path=None,
        rebuild_cache=False,
    ):
        self.input = input
        self.output = output

        # pandas Series/DataFrame 用 iloc 会有额外开销；转成 list 更快
        self.structures = input.tolist() if hasattr(input, "tolist") else list(input)
        self.targets = output.tolist() if (output is not None and hasattr(output, "tolist")) else (list(output) if output is not None else None)

        self.max_num_nbr = int(max_num_nbr)
        self.radius = float(radius)
        self.dmin = float(dmin)
        self.step = float(step)

        BASE_DIR = Path(__file__).resolve().parent
        self.atom_init_file = os.path.join(BASE_DIR, "atom_init.json")
        assert os.path.exists(self.atom_init_file), "atom_init.json does not exist!"

        self.ari = AtomCustomJSONInitializer(self.atom_init_file)
        self.gdf = GaussianDistance(dmin=self.dmin, dmax=self.radius, step=self.step)

        self.precomputed_data = []

        if not precompute:
            return

        # ---------- cache ----------
        if cache_path is not None and os.path.exists(cache_path) and not rebuild_cache:
            # 直接加载：训练速度最快（内存足够时）
            self.precomputed_data = torch.load(cache_path, map_location="cpu")
        else:
            self.precomputed_data = [None] * len(self.structures)
            self._precompute_all(
                num_workers=precompute_workers,
                backend=precompute_backend,
                chunksize=chunksize,
            )
            if cache_path is not None:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                torch.save(self.precomputed_data, cache_path)

    def _precompute_all(self, num_workers=0, backend="process", chunksize=32):
        """预计算所有样本的特征，避免训练时重复计算。"""
        from tqdm import tqdm

        n = len(self.structures)
        if self.targets is None:
            targets = [None] * n
        else:
            targets = self.targets

        # ----------- serial -----------
        if not num_workers or num_workers <= 1:
            for idx in tqdm(range(n), desc="Precomputing features (serial)"):
                crystal = self.structures[idx]
                target = targets[idx]
                self.precomputed_data[idx] = _compute_features(
                    crystal=crystal,
                    target=target,
                    ari=self.ari,
                    gdf=self.gdf,
                    max_num_nbr=self.max_num_nbr,
                    radius=self.radius,
                )
            return

        # ----------- threaded -----------
        if backend == "thread":
            # 注意：Python 线程是否提速取决于 pymatgen/邻居搜索是否释放 GIL；但不会有 pickle 开销
            with ThreadPoolExecutor(max_workers=int(num_workers)) as ex:
                futures = {}
                for idx in range(n):
                    futures[ex.submit(
                        _compute_features,
                        self.structures[idx],
                        targets[idx],
                        self.ari,
                        self.gdf,
                        self.max_num_nbr,
                        self.radius,
                    )] = idx

                for fut in tqdm(as_completed(futures), total=n, desc="Precomputing features (threads)"):
                    idx = futures[fut]
                    self.precomputed_data[idx] = fut.result()
            return

        # ----------- multiprocess -----------
        # Windows 用 spawn；Linux 推荐 fork（更快、共享内存页），这里按平台自动选
        ctx = multiprocessing.get_context("spawn")  # if os.name == "nt" else "fork")

        tasks = ((idx, self.structures[idx], targets[idx]) for idx in range(n))
        with ctx.Pool(
            processes=int(num_workers),
            initializer=_init_precompute_worker,
            initargs=(self.atom_init_file, self.radius, self.dmin, self.step, self.max_num_nbr),
        ) as pool:
            for idx, out in tqdm(
                pool.imap_unordered(_featurize_task, tasks, chunksize=int(chunksize)),
                total=n,
                desc=f"Precomputing features (processes={num_workers})",
            ):
                self.precomputed_data[idx] = out

    def __len__(self):
        return len(self.precomputed_data)

    def __getitem__(self, idx):
        # 直接从预计算的数据中获取
        atom_fea, nbr_fea, nbr_fea_idx, elem_ids, target = self.precomputed_data[idx]
        return (
            torch.from_numpy(atom_fea),
            torch.from_numpy(nbr_fea),
            torch.from_numpy(nbr_fea_idx),
            torch.from_numpy(elem_ids),
        ), torch.from_numpy(target)


# 小num_worker无
def get_train_loader(dataset, collate_fn=default_collate,
                     batch_size=128, train_ratio=0.75,
                     val_ratio=0.25, **kwargs):
    """
    用于划分数据集以训练、评估数据集
    数据集在使用函数之前需要洗牌
    Parameters
    ----------
    dataset: torch.utils.data.Dataset  要被划分的完整数据集
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      对训练数据进行随机采样的数据加载器

    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.

    """

    total_size = len(dataset)  # 全部数据集多少个
    #
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-valid_size:])

    num_workers = 0  # min(16, multiprocessing.cpu_count() // 2)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              pin_memory=False,
                              num_workers=num_workers,
                              # prefetch_factor=1,
                              # persistent_workers=False  # (num_workers > 0)
                              )
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            collate_fn=collate_fn,
                            pin_memory=False,
                            num_workers=num_workers,
                            # prefetch_factor=1,
                            # persistent_workers=False  # (num_workers > 0)
                            )

    return train_loader, val_loader
