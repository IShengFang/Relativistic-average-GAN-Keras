# Relativistic average GAN with Keras
The implementation [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) with Keras

[[paper]](https://arxiv.org/abs/1807.00734)
[[blog]](https://ajolicoeur.wordpress.com/relativisticgan/)
[[original code(pytorch)]](https://github.com/AlexiaJM/RelativisticGAN)

## How to Run?
### Python3 Script
``` bash
mkdir result
python RaGAN_CustomLoss.py [dataset] [loss] 
python RaGAN_CustomLayers.py [dataset] [loss] 
```
[dataset]: mnist, fashion_mnist, cifar10

[loss]: BXE for binary_crossentropy, LS for Least-Squares

### Jupyter notebook

Custom Loss
[[Colab]](https://drive.google.com/file/d/11NlU_Z829NXrHCdWx4ROIIcmfnaxNdR2/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Loss.ipynb)

Custom Layer
[[Colab]](https://drive.google.com/file/d/1pbUCguHX1h_yeDYMdFcYa0CYuZvtpUik/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Layers.ipynb)


## Result
| 1 epoch | MNIST    | Fashion MNIST | CIFAR10 |
| -------- | -------- | ------------- | -------- |
| Binary Cross Entropy     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_BXE/epoch_000.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_BXE/epoch_000.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_BXE/epoch_000.png)     |
|Least Square|![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_LS/epoch_000.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_LS/epoch_000.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_LS/epoch_000.png)     |

| 10 epoch | MNIST    | Fashion MNIST | CIFAR10 |
| -------- | -------- | ------------- | -------- |
| Binary Cross Entropy     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_BXE/epoch_010.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_BXE/epoch_010.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_BXE/epoch_010.png)     |
|Least Square|![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_LS/epoch_010.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_LS/epoch_010.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_LS/epoch_010.png)     |

| 50epoch | MNIST    | Fashion MNIST | CIFAR10 |
| -------- | -------- | ------------- | -------- |
| Binary Cross Entropy     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_BXE/epoch_049.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_BXE/epoch_049.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_BXE/epoch_049.png)     |
|Least Square|![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_LS/epoch_049.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_LS/epoch_049.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_LS/epoch_049.png)     |

| 100epoch | MNIST    | Fashion MNIST | CIFAR10 |
| -------- | -------- | ------------- | -------- |
| Binary Cross Entropy     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_BXE/epoch_099.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_BXE/epoch_099.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_BXE/epoch_099.png)     |
|Least Square|![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_LS/epoch_099.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_LS/epoch_099.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_LS/epoch_099.png)     |

| Loss | MNIST    | Fashion MNIST | CIFAR10 |
| -------- | -------- | ------------- | -------- |
| Binary Cross Entropy     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_BXE/loss.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_BXE/loss.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_BXE/loss.png)     |
|Least Square|![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/mnist_LS/loss.png )     | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/fashion_mnist_LS/loss.png )         | ![](https://raw.githubusercontent.com/IShengFang/Relativistic-average-GAN-Keras/master/result/cifar10_LS/loss.png)     |