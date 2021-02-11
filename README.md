# Relativistic average GAN with Keras
The implementation [Relativistic average GAN](https://ajolicoeur.wordpress.com/relativisticgan/) with Keras

[[paper]](https://arxiv.org/abs/1807.00734)
[[blog]](https://ajolicoeur.wordpress.com/relativisticgan/)
[[original code(pytorch)]](https://github.com/AlexiaJM/RelativisticGAN)

## How to Run?
### Python3 Script
``` bash
mkdir result
python RaGAN_CustomLoss.py --dataset [dataset] --loss [loss] 
python RaGAN_CustomLayers.py --dataset [dataset] --loss [loss] 
```
[dataset]: mnist, *fashion_mnist*, cifar10

[loss]: *BXE* for Binary Crossentropy, LS for Least Squares

*italic arguments* are default

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


## What is Relativistic average GAN?
### TL;DR
![](https://ajolicoeur.files.wordpress.com/2018/06/screenshot-from-2018-06-30-11-04-05.png?w=656)
### What is different with original GAN
For better math equations rendering, check out [HackMD Version](https://hackmd.io/s/r1VlR5CBm)
#### GAN
The GAN is the two player game which subject as below

![formula](https://render.githubusercontent.com/render/math?math=\max_G\min_DV%28G,D%29)

![formula](https://render.githubusercontent.com/render/math?math=V%28G,D%29) is a value function( aka loss or cost function)
![formula](https://render.githubusercontent.com/render/math?math=G:z%20\longmapsto%20x') is a generator, ![formula](https://render.githubusercontent.com/render/math?math=z) is a sample noise from the distribution we known(usually multidimensional Gaussian distribution). ![formula](https://render.githubusercontent.com/render/math?math=x') is a fake data generated by the generator. We want ![formula](https://render.githubusercontent.com/render/math?math=x') in the real data distribution.
![formula](https://render.githubusercontent.com/render/math?math=D:x\longmapsto[0,1]) is a discriminator, which finds out that ![formula](https://render.githubusercontent.com/render/math?math=x) is a real data (output 1) or a fake data(output 0)
In the training iteration, we will train one neural network first(usual is discriminator), and train the other network. After a lot of iterations, we expect the last generator to map multidimensional Gaussian distribution to the real data distribution.


#### Relativistic average GAN (RaGAN)
RaGAN's Loss function does not optimize discriminator to distinguish data real or fake. Instead, RaGAN's discriminator distinguishes that "*real data* isn’t like *average fake data*" or "*fake data* isn’t like *average real data*".

>the discriminator estimates the probability that the given real data is more realistic than a randomly sampled fake data.
[paper subsection.4.1](https://arxiv.org/pdf/1807.00734.pdf#subsection.4.1)

Given Discriminator output ![formula](https://render.githubusercontent.com/render/math?math=D(x)=\text{sigmoid}(C(x)))
Origin GAN Loss is as below,

![formula](https://render.githubusercontent.com/render/math?math=L_D=-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}[\logD(x_{real})]-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}[\log ( 1-D(x_{fake}))])

![formula](https://render.githubusercontent.com/render/math?math=L_G=-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}[\logD(x_{fake})])


Relativistic average output is ![formula](https://render.githubusercontent.com/render/math?math=\tilde{D}(x_{real})=\text{sigmoid}(C(x_{real})-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}C(x_{fake}))) and ![formula](https://render.githubusercontent.com/render/math?math=\tilde{D}(x_{fake})=\text{sigmoid}(C(x_{fake})-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}C(x_{real})))

RaGAN's Loss is as below,
![formula](https://render.githubusercontent.com/render/math?math=L_D=-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}[\log\tilde{D}(x_{real})]-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}[\log(1-\tilde{D}(x_{fake}))])
![formula](https://render.githubusercontent.com/render/math?math=L_G=-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}[\log\tilde{D}(x_{fake})]-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}[\log(1-\tilde{D}(x_{real}))])

we can also add relativistic average in Least Square GAN or any other GAN
![](https://cdn-images-1.medium.com/max/800/1*QKG1fVOMjGlVUvICYmz8vQ.png)
Modified by [Jonathan Hui](https://medium.com/@jonathan_hui/gan-rsgan-ragan-a-new-generation-of-cost-function-84c5374d3c6e) from [Paper](https://arxiv.org/abs/1807.00734)

### How to implement with Keras?
We got loss, so just code it. :smile:
Just kidding, we have two approaches to implement RaGAN.
The important part of implementation is discriminator output.
![formula](https://render.githubusercontent.com/render/math?math=\tilde{D}(x_{real})=\text{sigmoid}(C(x_{real})-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}C(x_{fake}))) and ![formula](https://render.githubusercontent.com/render/math?math=\tilde{D}(x_{fake})=\text{sigmoid}(C(x_{fake})-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}C(x_{real})))
We need to average ![formula](https://render.githubusercontent.com/render/math?math=C(x_{real})) and ![formula](https://render.githubusercontent.com/render/math?math=C(x_{fake})). We also need "minus" to get ![formula](https://render.githubusercontent.com/render/math?math=C(x_{real})-\mathbb{E}_{x_{fake}\sim\mathbb{P}_{fake}}C(x_{fake})) and ![formula](https://render.githubusercontent.com/render/math?math=C(x_{fake})-\mathbb{E}_{x_{real}\sim\mathbb{P}_{real}}C(x_{real})).

We can use keras.backend to deal with it, but that means we need to custom loss. We can also write custom layers to apply these computations to keras as a layer, and use Keras default loss to train the model.

1. Custom layer
    - Pros:
        - Train our RaGAN easily with keras default loss
    - Cons:
        - Write custom layers to implement it.
    - [[Colab]](https://drive.google.com/file/d/1pbUCguHX1h_yeDYMdFcYa0CYuZvtpUik/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Layers.ipynb)


2. Custom Loss
    - Pros:
        - Do not need to write custom layers. Instead, we need write loss with keras.backend.
        - Custom loss is easy to change loss.
    - Cons:
        - Write custom loss with keras.backend to implement it.
    - [[Colab]](https://drive.google.com/file/d/11NlU_Z829NXrHCdWx4ROIIcmfnaxNdR2/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Loss.ipynb)

#### Code
##### Custom Loss
[[Colab]](https://drive.google.com/file/d/11NlU_Z829NXrHCdWx4ROIIcmfnaxNdR2/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Loss.ipynb)[[python script]](https://github.com/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_CustomLoss.py)
##### Custom Layer
[[Colab]](https://drive.google.com/file/d/1pbUCguHX1h_yeDYMdFcYa0CYuZvtpUik/view?usp=sharing)[[NBViewer]](https://nbviewer.jupyter.org/github/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_with_Custom_Layers.ipynb)[[python script]](https://github.com/IShengFang/Relativistic-average-GAN-Keras/blob/master/RaGAN_CustomLayers.py)
