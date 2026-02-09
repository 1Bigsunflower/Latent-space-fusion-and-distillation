# Latent-space fusion and cross-task distillation yield transferable Mat2Vec replacements

This repository contains the code and pretrained models used for learning, extracting, and fusing element embeddings based on **CGCNN** and **MEGNet**, as well as constructing **Mat2Vec-S / Mat2Vec-L / Mat2Vec-H** embeddings for downstream materials property prediction tasks.

---

## 1. Environment Setup

All required Python dependencies are listed in `requirements.txt`.

Install the environment using:

```bash
conda create -n mat_env python=3.10
source activate mat_env
pip install -r requirements.txt
```

---

## 2. Base Model Training (CGCNN & MEGNet)

All training-related code is located in the `train_model` directory.

### 2.1 CGCNN

- `train_model/CGCNN/CGCNN_emb`  
  Training code for **pure learnable element embeddings** with CGCNN.

- `train_model/CGCNN/CGCNN_hm`  
  Training code for **pure handcrafted element features** with CGCNN.

- One-click training script:
  ```bash
  bash train_model/CGCNN/train_cgcnn.sh
  ```

### 2.2 MEGNet

- `train_model/MEGNet/MEGNet_emb`  
  Training code for **pure learnable element embeddings** with MEGNet.

- `train_model/MEGNet/MEGNet_hm`  
  Training code for **pure handcrafted element features** with MEGNet.

### 2.3 Pretrained Models

All pretrained models are saved in:

```
pre-train_model/
```

---

## 3. Element Embedding Extraction

Element embeddings learned by the trained models can be extracted using scripts in the `Embedding Extraction` directory.

Run the following scripts:

```bash
bash run_cgcnn_hm.sh
bash run_cgcnn_emb.sh
bash run_megnet_hm.sh
bash run_megnet_emb.sh
```

The extracted embeddings are saved in:

```
Embedding_weight_orig/
```

---

## 4. Embedding Fusion with Different Ratios

### 4.1 Fusion Model Training

Code for fusing learnable and handcrafted element embeddings with different ratios is located in:

```
train_model/Model_emb_fusion/
```

- CGCNN fusion models:
  ```
  train_model/Model_emb_fusion/CGCNN_fly/
  ```

- MEGNet fusion models:
  ```
  train_model/Model_emb_fusion/MEGNet_fly/
  ```

To train the fusion model, enter the corresponding directory and run:

```bash
bash train_e_form.sh
```

> For other datasets, only the `subset` parameter in the script needs to be modified.

### 4.2 Fusion Model Checkpoints

- CGCNN fusion models:
  ```
  get_embedding/cgcnn_ef_model/
  ```

- MEGNet fusion models:
  ```
  get_embedding/megnet_ef_model/
  ```

### 4.3 Fused Element Embedding Extraction

- CGCNN fused embeddings:
  ```bash
  python get_embedding/CGCNN_ef_get_embedding/get_embedding_ef.py
  ```

- MEGNet fused embeddings:
  ```bash
  python get_embedding/MEGNet_ef_get_embedding/get_embedding_ef.py
  ```

---

## 5. Mat2Vec-S / Mat2Vec-L / Mat2Vec-H Construction and CrabNet Training

### 5.1 Mat2Vec-* Embedding Generation

The following scripts are used to obtain **Mat2Vec-S**, **Mat2Vec-L**, and **Mat2Vec-H** embeddings reported in the paper:

```bash
python get_embedding/get_min_model.py
python get_embedding/get_min_model_L_H.py
```

### 5.2 Post-processing

The generated embeddings are post-processed using z-score normalization:

```bash
python 8task_embedding_result/zscore.py
```

### 5.3 CrabNet Training

- The processed embeddings are stored in:
  ```
  crabnet/data/element_properties/
  ```

- CrabNet can be trained directly using:
  ```bash
  bash train_matbench_emb.sh
  ```

---

## 6. Reproducing Figures in the Paper

The scripts used to reproduce the figures in the paper are provided as follows:

- **Figure 2 / Figure 3 / Figure 5**:
  ```
  get_embedding/fig2.py
  get_embedding/fig3.py
  get_embedding/fig5.py
  ```

- **Figure 4**:
  ```
  get_embedding/check_cos/fig4/
  ```

---

