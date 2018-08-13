# Relativistic average GAN with Keras
The implementation [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) with Keras

[[paper]](https://arxiv.org/abs/1807.00734)
[[blog]](https://ajolicoeur.wordpress.com/relativisticgan/)
[[original code(pytorch)]](https://github.com/AlexiaJM/RelativisticGAN)

## How to Run?
``` bash
mkdir result
python RaGAN_CustomLoss.py [dataset] [loss] 
```
[dataset]: mnist fashion_mnist cifar10
[loss]: BXE for binary_crossentropy, LS for Least-Squares