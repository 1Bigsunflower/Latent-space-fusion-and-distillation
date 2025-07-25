This repository provides training scripts and configurations for two widely used material property prediction models: CGCNN and MEGNet. It supports training with different types of embeddings, including different dimensions, nn.Embedding, and human knowledge-based embeddings.

## 📁 Project Structure

```
.
├── requirements.txt                    # Python dependencies
├── train_model/
│   ├── CGCNN/
│   │   └── train_cgcnn.sh              # Train CGCNN with various embedding dimensions
│   └── MEGNet/
│       ├── MEGNet_emb/
│       │   └── megnet_orig.sh         # Train MEGNet using nn.Embedding
│       └── MEGNet_hm/
│           └── megnet_hm.sh           # Train MEGNet using human knowledge-based embeddings
└── (Other ...)
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

