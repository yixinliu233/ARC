# ARC
 This repository is the official implementation of "[ARC: A Generalist Graph Anomaly Detector with In-Context Learning](https://arxiv.org/pdf/2405.16771)", accepted by NeurIPS 2024.

# Setup
```js/java/c#/text
conda create -n ARCGAD python=3.8
conda activate ARCGAD
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.3.1

pip uninstall networkx -y
pip install networkx==2.8.8
pip uninstall dgl -y
pip install dgl==0.9.0
```

# Usage
Just run the script corresponding to the dataset and method you want. For instance:

```js/java/c#/text
python main.py --trial 5 --shot 10
```

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@article{liu2024arc,
  title={ARC: A Generalist Graph Anomaly Detector with In-Context Learning},
  author={Liu, Yixin and Li, Shiyuan and Zheng, Yu and Chen, Qingfeng and Zhang, Chengqi and Pan, Shirui},
  journal={arXiv preprint arXiv:2405.16771},
  year={2024}
}
```

