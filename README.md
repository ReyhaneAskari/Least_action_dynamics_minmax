# Least Action Daynamics
This is the code associated with the paper ["LEAD: Least-Action Dynamics for Min-Max Optimization"](https://arxiv.org/pdf/2010.13846.pdf). If you find this code useful please cite us:

```
@article{hemmat2020LEAD,
  title={LEAD: Least-Action Dynamics for Min-Max Optimization},
  author={Hemmat, Reyhane Askari and Mitra, Amartya and Lajoie, Guillaume and Mitliagkas, Ioannis},
  journal={arXiv preprint arXiv:2010.13846},
  year={2020}
}
```


For any questions about the code please create an issue.

The code requires pytorch and tensorflow. But TF is only used for computing the inception score.


## Acknowledgement

1. DCGAN code adpoted from https://github.com/Zeleni9/pytorch-wgan
2. ResNet code adopted from https://github.com/GongXinyuu/sngan.pytorch
3. SGA code implemented based on https://github.com/deepmind/symplectic-gradient-adjustment/blob/master/Symplectic_Gradient_Adjustment.ipynb
4. Extra-Adam optim source code from https://github.com/GauthierGidel/Variational-Inequality-GAN
5. CGD optim source code from https://github.com/devzhk/Implicit-Competitive-Regularization
