This repository provides training scripts and configurations for two widely used material property prediction models: CGCNN and MEGNet. It supports training with different types of embeddings, including different dimensions, nn.Embedding, and human knowledge-based embeddings.

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
│   └── (One folder per model)             # Saved extracted embeddings
└── ...
```
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
