This repository provides training scripts and configurations for two widely used material property prediction models: CGCNN and MEGNet. It supports training with different types of embeddings, including different dimensions, nn.Embedding, and human knowledge-based embeddings.
<!--
## This section is hidden
## 📁 Project Structure

```
.
├── requirements.txt                        # Python dependencies
├── train_model/
│   ├── CGCNN/
│   │   └── train_cgcnn.sh                  # Train CGCNN with various embedding dimensions
│   └── MEGNet/
│       ├── MEGNet_emb/
│       │   └── megnet_orig.sh             # Train MEGNet using nn.Embedding
│       └── MEGNet_hm/
│           └── megnet_hm.sh               # Train MEGNet using human knowledge-based embeddings
├── pre-train_model/
│   ├── cgcnn/                              # Pretrained CGCNN models
│   ├── megnet/                             # Pretrained MEGNet models
│   └── result/                             # MAE results corresponding to pretrained models
├── Embedding Extraction/
│   ├── CGCNN/
│   │   ├── cgcnn_emb/run_cgcnn_emb.sh    
│   │   └── cgcnn_hm/run_cgcnn_hm.sh     
│   └── MEGNet/
│       ├── megnet_emb/run_megnet_emb.sh  
│       └── megnet_hm/run_megnet_hm.sh            
├── Embedding_weight_orig/
├── Picture/                               # paper figures
│   ├── Fig1/
│   │   └── 1D_order.py                    # Generates Figure 1
│   ├── Fig3-4/
│   │   ├── CGCNN_Bar_Chart.py            # Generates Figure 3
│   │   └── MEGNet_Bar_Chart.py           # Generates Figure 4
│   ├── Fig5/
│   │   ├── RF_composition.py             # Random Forest regression results
│   │   ├── R2_for_best_dummy.py          # R2 scores of leaderboard & dummy models
│   │   └── Fig5_rf.py                    # Generates Figure 5
│   └── Fig6/
│       ├── Embedding_analysis.py         # Computes embedding structure metrics
│       ├── norm_mae.py                   # Normalizes MAEs for each task
│       └── Correlation_Matrix_Heatmap.py # Correlation analysis and heatmap for Fig 6
├── Element with mpnn/                    
└── ...
```
-->
## 🔧 Environment Setup

We recommend using [Conda](https://docs.conda.io/) to manage the environment. The project requires **Python 3.10**.

```bash
conda create -n group_env python=3.10
source activate group_env
pip install -r requirements.txt
```

## 🚀 Quick Start

### Train CGCNN 

```bash
cd train_model/CGCNN
bash train_cgcnn.sh
```

### Train MEGNet

- Using `nn.Embedding`:

  ```bash
  cd train_model/MEGNet/MEGNet_emb
  bash megnet_orig.sh
  ```

- Using human knowledge-based embeddings:

  ```bash
  cd train_model/MEGNet/MEGNet_hm
  bash megnet_hm.sh
  ```

## 🧠 Pretrained Models & Results

Pretrained models from this study are saved in the `pre-train_model/` directory:

- `pre-train_model/cgcnn/` and `pre-train_model/megnet/`: store pretrained weights.
- `pre-train_model/result/`: contains corresponding MAE values for each model.

## 📦 Embedding Extraction

Element embeddings can be extracted using scripts under the `Embedding Extraction/` folder.

For example:

- Extract CGCNN nn.Embedding:

  ```bash
  bash Embedding Extraction/CGCNN/cgcnn_emb/run_cgcnn_emb.sh
  ```

Extracted embeddings are saved in `Embedding_weight_orig/`.

## 📊 Paper Figures

Scripts used to generate figures in the paper are located in the `Picture/` and `Element with mpnn/` directories:

- **Figure 1** (`Picture/Fig1/`):  
  Run `1D_order.py` to generate the ordering visualization.

- **Figures 3 & 4** (`Picture/Fig3-4/`):  
  - Run `CGCNN_Bar_Chart.py` for Figure 3  
  - Run `MEGNet_Bar_Chart.py` for Figure 4

- **Figure 5** (`Picture/Fig5/`):  
  1. Run `RF_composition.py` to compute random forest regression results  
  2. Run `R2_for_best_dummy.py` for leaderboard and dummy R² scores  
  3. Run `Fig5_rf.py` to generate the final plot

- **Figure 6** (`Picture/Fig6/`):  
  1. Run `Embedding_analysis.py` to calculate various embedding structure metrics  
  2. Run `norm_mae.py` to normalize MAE for each task  
  3. Run `Correlation_Matrix_Heatmap.py` to compute correlations and plot the heatmap

`Element with mpnn/` contains scripts for analyzing how element embeddings evolve across message passing layers in both CGCNN and MEGNet architectures.

To extract embeddings at each message passing layer, run the following scripts:

```bash
# CGCNN - nn.Embedding
cd Element with mpnn/CGCNN-emb
bash run_cgcnn_emb.sh

# CGCNN - Human Knowledge
cd ../CGCNN-human
bash run_cgcnn_hm.sh

# MEGNet - nn.Embedding
cd ../MEGNet-emb
bash run_megnet_emb.sh

# MEGNet - Human Knowledge
cd ../MEGNet-human
bash run_megnet_hm.sh
```

These scripts save intermediate embeddings for each model configuration, which are later used for visualization and clustering analysis.

`Element with mpnn/ari_score.sh`: Computes Adjusted Rand Index (ARI) between learned embeddings and traditional element groups.

`Element with mpnn/s_score.sh`: Computes silhouette scores for cluster separation.

`Element with mpnn/kmeans.sh`: Performs k-means clustering on the embeddings.
  
- **Figure 7**(`Element with mpnn/Fig7/`):  
  - Run `run_picture.sh` to generate 2D projections (t-SNE) of element embeddings across message passing layers.

- **Figure 8**(`Element with mpnn/Fig8/ari/`):  
  - Run `Fig8.py` to plot how the ARI scores change across message passing layers for different tasks.

- **Figure 9**:(`Element with mpnn/`)
  - Run `GA_cluster.py` to perform multi-task clustering using a genetic algorithm, producing a **general-purpose periodic table-based clustering**.
  - Results are saved in the `GA_cluster_result/` folder.
  - Visualize the generalized clustering by running:

  ```bash
  python Fig_p_table.py
  ```


