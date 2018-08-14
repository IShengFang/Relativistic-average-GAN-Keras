import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, Activation, LeakyReLU, Concatenate
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Reshape
import keras.backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import *
from keras.utils.generic_utils import Progbar
from time import time
import sys
import os
import pickle

arg_list = sys.argv
plt.switch_backend('agg')
EPOCHS = 100
BATCHSIZE = 64
TRAINING_RATIO =1
DATASET = arg_list[1] # 'mnist', 'fashion_mnist'  'cifar10'
LOSS = arg_list[2] #'BXE' 'LS'
GENERATE_ROW_NUM = 10
OPT = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

STAMP = '{}_{}'.format(DATASET, LOSS)

if not os.path.isdir('result/'+STAMP):
    print('mkdir result/{}'.format(STAMP))
    os.mkdir('result/'+STAMP)

from keras.datasets import mnist, fashion_mnist, cifar10
exec('(X_train, y_train), (X_test, y_test) = {}.load_data()'.format(DATASET))

X = np.concatenate((X_train, X_test))
if len(X.shape)==3:
    X = np.expand_dims(X, axis=-1)
X = X/255*2-1

def DC_Generator(input_shape=(128,) ,output_shape=(28,28,1), dc_shape=(7,7,128), name='Generator'):
    layer_num=int(np.log2(output_shape[1]/dc_shape[1]))
    
    z = Input(shape=input_shape)
    h = Dense(dc_shape[0]*dc_shape[1]*dc_shape[2], activation='relu',kernel_initializer='glorot_uniform')(z)
    h = Reshape(dc_shape)(h) 
    for i in range(layer_num):  
        h = Conv2DTranspose(int(dc_shape[2]/(2**(i+1))), kernel_size=4, strides=2, padding='same', activation='relu',kernel_initializer='glorot_uniform')(h)
        h = BatchNormalization(momentum=0.9, epsilon=0.00002)(h)
    
    x = Conv2DTranspose(output_shape[-1], kernel_size=3, strides=1, padding='same', activation='tanh',kernel_initializer='glorot_uniform')(h)
    model = Model(z,x, name=name)
    model.summary()
    return model

def DC_Discriminator(input_shape=(28,28,1),layer_num=2, start_dim=64, name='Discriminator'):
    
    x = Input(shape=input_shape)
    h = x
    for i in range(layer_num):
        h = Conv2D(start_dim*(2**i), kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_uniform')(h)
        h = LeakyReLU(0.1)(h)       
    
    h = GlobalAveragePooling2D()(h)
    y = Dense(1,kernel_initializer='glorot_uniform' )(h)
    model = Model(x,y, name=name)
    model.summary()
    return model

if X.shape[2] == 28:
    dc_shape = (7,7,128)
    dis_layer_num = 2
else:
    dc_shape = (4,4,512)
    dis_layer_num = 4
generator = DC_Generator(output_shape=X.shape[1:], dc_shape=dc_shape)
discriminator = DC_Discriminator(input_shape=X.shape[1:], layer_num=dis_layer_num)

def relativistic_average(input_):
    x_0 = input_[0]
    x_1 = input_[1]
    return x_0 - K.mean(x_1, axis=0)


Real_image                         = Input(shape=X.shape[1:])
Noise_input                        = Input(shape=(128,))
Fake_image                         = generator(Noise_input)
Discriminator_real_out             = discriminator(Real_image)
Discriminator_fake_out             = discriminator(Fake_image)
Real_Fake_relativistic_average_out = Lambda(relativistic_average, name='Real_minus_mean_fake')([Discriminator_real_out, Discriminator_fake_out])
Fake_Real_relativistic_average_out = Lambda(relativistic_average, name='Fake_minus_mean_real')([Discriminator_fake_out, Discriminator_real_out])

if LOSS=='BXE':
    Real_Fake_relativistic_average_out = Activation('sigmoid')(Real_Fake_relativistic_average_out)
    Fake_Real_relativistic_average_out = Activation('sigmoid')(Fake_Real_relativistic_average_out)
    
Discriminator_Relativistic_out = Concatenate()([Real_Fake_relativistic_average_out, Fake_Real_relativistic_average_out])

epsilon=0.000001
if LOSS=='BXE':
    LOSS='binary_crossentropy'
elif LOSS=='LS':
    LOSS='mean_squared_error'
    
generator_train = Model([Noise_input, Real_image], Discriminator_Relativistic_out)
discriminator.trainable=False
generator_train.compile(OPT, loss=LOSS)
generator_train.summary()

discriminator_train = Model([Noise_input, Real_image],Discriminator_Relativistic_out)
generator.trainable = False
discriminator.trainable=True
discriminator_train.summary()
discriminator_train.compile(OPT, loss=LOSS)

if LOSS=='binary_crossentropy':
    true_y = np.ones((BATCHSIZE, 1), dtype=np.float32)
    fake_y = np.zeros((BATCHSIZE, 1), dtype=np.float32)
    y_for_dis = np.concatenate((true_y, fake_y), axis=1)
    y_for_gen = np.concatenate((fake_y, true_y), axis=1)
if LOSS=='mean_squared_error':
    true_y = np.ones((BATCHSIZE, 1), dtype=np.float32)
    fake_y = -np.ones((BATCHSIZE, 1), dtype=np.float32)
    y_for_dis = np.concatenate((true_y, fake_y), axis=1)
    y_for_gen = np.concatenate((fake_y, true_y), axis=1)

GENERATE_BATCHSIZE = GENERATE_ROW_NUM*GENERATE_ROW_NUM
test_noise = np.random.randn(GENERATE_BATCHSIZE, 128)

discriminator_loss = list()
generator_loss = list()
for epoch in range(EPOCHS):
    
    np.random.shuffle(X)

    print("epoch {} of {}".format(epoch+1, EPOCHS))
    num_batches = int(X.shape[0] // BATCHSIZE)
    minibatches_size = BATCHSIZE * (TRAINING_RATIO+1)
    print("number of batches: {}".format(int(X.shape[0] // (minibatches_size))))
    
    progress_bar = Progbar(target=int(X.shape[0] // minibatches_size))
    plt.clf()
    start_time = time()
    for index in range(int(X.shape[0] // (minibatches_size))):
        progress_bar.update(index)
        itreation_minibatches = X[index * minibatches_size:(index + 1) * minibatches_size]

        for j in range(TRAINING_RATIO):
            image_batch = itreation_minibatches[j * BATCHSIZE : (j + 1) * BATCHSIZE]
            noise = np.random.randn(BATCHSIZE, 128).astype(np.float32)
            discriminator.trainable = True
            generator.trainable = False
            discriminator_loss.append(discriminator_train.train_on_batch([noise, image_batch],y_for_dis))
            
        image_batch = itreation_minibatches[TRAINING_RATIO*BATCHSIZE : (TRAINING_RATIO + 1) * BATCHSIZE]
        noise = np.random.randn(BATCHSIZE, 128).astype(np.float32)
        discriminator.trainable = False
        generator.trainable = True
        generator_loss.append(generator_train.train_on_batch([noise, image_batch], y_for_gen))

    print('\nepoch time: {}'.format(time()-start_time))
    
    generated_image = generator.predict(test_noise)
    generated_image = (generated_image+1)/2
    for i in range(GENERATE_ROW_NUM):
        if X.shape[3]==1:
            new = generated_image[i*GENERATE_ROW_NUM:i*GENERATE_ROW_NUM+GENERATE_ROW_NUM].reshape(X.shape[2]*GENERATE_ROW_NUM,X.shape[2])
        else:
            new = generated_image[i*GENERATE_ROW_NUM:i*GENERATE_ROW_NUM+GENERATE_ROW_NUM].reshape(X.shape[2]*GENERATE_ROW_NUM,X.shape[2], 3)
        if i!=0:
            old = np.concatenate((old,new),axis=1)
        else:
            old = new
    print('plot generated_image')
    if X.shape[-1]==1:
        plt.imsave('result/{}/epoch_{:03}.png'.format(STAMP, epoch), old, cmap='gray')
    else:
        plt.imsave('result/{}/epoch_{:03}.png'.format(STAMP, epoch), old)
        
    print('plot Loss')
    plt.plot(discriminator_loss)
    plt.plot(generator_loss)
    plt.legend(['discriminator', 'generator'])
    plt.savefig('result/{}/loss.png'.format(STAMP))
    plt.clf()

    pickle.dump({'discriminator_loss': discriminator_loss, 
                 'generator_loss': generator_loss}, 
                open('result/{}/loss-history.pkl'.format(STAMP), 'wb'))