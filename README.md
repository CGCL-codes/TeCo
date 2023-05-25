# TeCo

The official implementation of our CVPR 2023 paper "Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency".[[Paper](https://arxiv.org/abs/2303.18191)] 

![Backdoor Detection](https://img.shields.io/badge/Backdoor-Detction-yellow.svg?style=plastic)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-orange.svg?style=plastic)

## Abstract
Deep neural networks are proven to be vulnerable to backdoor attacks. Detecting the trigger samples during the inference stage, 
i.e., the test-time trigger sample detection, can prevent the backdoor from being triggered. 
However, existing detection methods often require the defenders to have high accessibility to victim models, extra clean data, 
or knowledge about the appearance of backdoor triggers, limiting their practicality. \
In this paper, we propose the **te**st-time **co**rruption robustness consistency evaluation (**TeCo**), 
a novel test-time trigger sample detection method that only needs the hard-label outputs of the victim models without any extra information. 
Our journey begins with the intriguing observation that the backdoor-infected models have similar performance across different image corruptions for the clean images, 
but perform discrepantly for the trigger samples. 
Based on this phenomenon, we design TeCo to evaluate test-time robustness consistency by calculating the deviation of 
severity that leads to predictions' transition across different corruptions. Extensive experiments demonstrate that compared with state-of-the-art defenses, 
which even require either certain information about the trigger types or accessibility of clean data, 
TeCo outperforms them on different backdoor attacks, datasets, and model architectures, enjoying a higher AUROC by 10% and 5 times of stability.

## Deploy TeCo on BackdoorBench-v1.0 Codebase
### Setup
- **Get TeCo**
```shell 
git clone https://github.com/CGCL-codes/TeCo.git
cd TeCo
```
- **Get BackdoorBench-v1.0**\
*Merge Teco into the [BackdoorBench-v1.0](https://github.com/SCLBD/BackdoorBench/tree/v1) codebase*
```shell 
git clone -b v1 https://github.com/SCLBD/BackdoorBench.git
rsync -av BackdoorBench-v1.0-merge/ BackdoorBench/
cd BackdoorBench
sh ./sh/install.sh
mkdir record
mkdir data
mkdir data/cifar10
mkdir data/cifar100
mkdir data/gtsrb
mkdir data/tiny
```
- **Install Additional Package**\
*Use [imagecorruptions](https://github.com/bethgelab/imagecorruptions) for fast image corruptions deployment.*
```shell 
pip install imagecorruptions
```

### Quick Start
- **Train a Backdoor Model**
```
python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_0_1
```

- **TeCo Detection**
```
python ./defense/teco/teco.py --result_file badnet_0_1 --yaml_path ./config/defense/teco/cifar10.yaml --dataset cifar10
```

For guidance on conducting more evaluations, such as using different attacks, datasets, and model architectures, please refer to [BackdoorBench-v1.0](https://github.com/SCLBD/BackdoorBench/tree/v1).

## BibTeX 
If you find TeCo both interesting and helpful, please consider citing us in your research or publications:
```bibtex
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Xiaogeng and Li, Minghui and Wang, Haoyu and Hu, Shengshan and Ye, Dengpan and Jin, Hai and Wu, Libing and Xiao, Chaowei},
    title     = {Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16363-16372}
}
```
## Acknowledge
```bibtex
@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
```bibtex
@article{michaelis2019dragon,
  title={Benchmarking Robustness in Object Detection: 
    Autonomous Driving when Winter is Coming},
  author={Michaelis, Claudio and Mitzkus, Benjamin and 
    Geirhos, Robert and Rusak, Evgenia and 
    Bringmann, Oliver and Ecker, Alexander S. and 
    Bethge, Matthias and Brendel, Wieland},
  journal={arXiv preprint arXiv:1907.07484},
  year={2019}
}
```
