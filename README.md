# LEAD: Min-Max Optimization from a Physical Perspective
This is the code associated with the paper ["LEAD: Min-Max Optimization from a Physical Perspective"](https://openreview.net/forum?id=vXSsTYs6ZB). If you find this code useful please cite us:

```
@article{
askari hemmat2023lead,
title={{LEAD}: Min-Max Optimization from a Physical Perspective},
author={Reyhane Askari Hemmat and Amartya Mitra and Guillaume Lajoie and Ioannis Mitliagkas},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vXSsTYs6ZB},
note={Featured Certification}
}
```
Video describing the paper: https://www.youtube.com/watch?v=EfwIc0GXb8E 

Blogpost: https://reyhaneaskari.github.io/LEAD.html

For any questions about the code please create an issue.

The code requires pytorch and tensorflow. But TF is only used for computing the inception score.


## Acknowledgement

1. DCGAN code adpoted from https://github.com/Zeleni9/pytorch-wgan
2. ResNet code adopted from https://github.com/GongXinyuu/sngan.pytorch
3. SGA code implemented based on https://github.com/deepmind/symplectic-gradient-adjustment/blob/master/Symplectic_Gradient_Adjustment.ipynb
4. Extra-Adam optim source code from https://github.com/GauthierGidel/Variational-Inequality-GAN
5. CGD optim source code from https://github.com/devzhk/Implicit-Competitive-Regularization
