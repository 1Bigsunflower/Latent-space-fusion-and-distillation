import subprocess

# conda create -n myenv python=3.10

# pip freeze > requirements.txt
# 通过requirements.txt安装环境 跳过无法安装的

failed = []

with open("requirements.txt", encoding="utf-8") as f:
    for line in f:
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        print(f"Installing: {pkg}")
        result = subprocess.run(["pip", "install", pkg])
        if result.returncode != 0:
            failed.append(pkg)

print("\n❌ Failed to install the following packages:")
for pkg in failed:
    print(f"- {pkg}")


# ❌ Failed to install the following packages:
# - dgl==2.1.0+cu118
# - matbench @ file:///home/ubuntu/cht/matbench
# - torch==2.2.1+cu118
# - torchaudio==2.2.1+cu118
# - torchvision==0.17.1+cu118

# pip install torch==2.2.0 torchaudio==2.2.0 torchvision==0.17 --index-url https://download.pytorch.org/whl/cu121
# pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
# pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
# git clone https://github.com/hackingmaterials/matbench
# pip install --user ./matbench