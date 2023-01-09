# Unsupervised Learning of Group Invariant and Equivariant Representations 

<img src="https://github.com/bayer-int/giae/blob/main/assets/idea.png" alt="idea" width="500" height="300" />

This repository holds the source code to reproduce the experiments done for the paper "Unsupervised Learning of Group Invariant and 
Equivariant Representations" presented at NeurIPS 2022.  
Check the repositories so2, se3 and sn for more instructions how to run and evaluate the models for the different 
groups and data types.


## Dependencies
- torch
- torch_geometric
- pytorch_lightning
- [e2cnn](https://github.com/QUVA-Lab/e2cnn)

## Installation

```
conda create -n giae
conda activate giae 
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 pyg pytorch-lightning=1.6.2 -c pytorch -c pyg -c conda-forge
pip install .
```

## How to run the code

We have provided instructions for the different group-specific implementations here:
- [SO(2)](so2/README.md)
- [SE(3)](se3/README.md)
- [S(N)](sn/README.md)

as well as accompanying jupyter notebooks to analyze the results.

## References
> @inproceedings{
winter2022unsupervised,
title={Unsupervised Learning of Group Invariant and Equivariant Representations},
author={Robin Winter and Marco Bertolini and Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=47lpv23LDPr}
}
