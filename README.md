# PVeRA: Probabilistic Vector-Based Random Matrix Adaptation
This is the official implementation for PVeRA: Probabilistic Vector-Based Random Matrix Adaptation, which has been accepted for presentation at WACV 2026.

[PVeRA: Probabilistic Vector-Based Random Matrix Adaptation](https://www.arxiv.org/abs/2512.07703)

Leo Fillioux, Enzo Ferrante, Paul-Henry Cournède, Maria Vakalopoulou, Stergios Christodoulidis

## Getting started
1. Clone this repository.
2. Download the VTAB-1k benchmark datasets (or a subset of these datasets). You can use [this repository](https://github.com/BenediktAlkin/vtab1k-pytorch/blob/main/SETUP_DATA.md) explains how to download the benchmark in the correct format.
3. Change the paths in `config/training.yaml` and `config/datasets.yaml` to fit your environment.
4. Setup the environment.
```bash
conda create python=3.10 -n pvera
conda activate pvera
pip install -r requirements.txt
```

## Training a model
Use the following commands to train a model.
```bash
# generic command for training linear probing
python train.py -d <dataset_name> -m DINOv2
# generic command for training adapters
python train.py -d <dataset_name> -m DINOv2 -a <adapter_name> 
# example command for training on cifar with pvera
python train.py -d cifar -m DINOv2 -a pvera
```

## Citation
If you found this repository useful, please consider citing our work using the following BibTeX entry.
```
@InProceedings{fillioux2025pvera,
  title={{PVeRA}: Probabilistic Vector-Based Random Matrix Adaptation},
  author={Fillioux, Leo and Ferrante, Enzo and Cournède, Paul-Henry and Vakalopoulou, Maria and Christodoulidis, Stergios},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```
