
import numpy as np
import random
import keras
from keras.models import Model
from keras.layers import Input, ReLU, LeakyReLU, BatchNormalization, Add, AveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Conv2D, Concatenate, Lambda, Masking, TimeDistributed, Activation
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf

import constants as cs
from base_model import Base_Model
import os
import itertools
import gc


class Simple_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/simple_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mse', is_compile = True):
        def res_block(x, channel_num, kernel_s = 3, stride = 1):
            # reference : https://arxiv.org/abs/1603.05027
            hidden_1 = x
            hidden_2 = BatchNormalization()(hidden_1)
            hidden_2 = ReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            hidden_2 = BatchNormalization()(hidden_2)
            hidden_2 = ReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            result = Add()([hidden_1, hidden_2])
            return result
    
        def encoder_block(x, n_channel):
            x = Conv2D(n_channel, kernel_size = 3,strides=1, padding = 'same')(x)
            x = ReLU()(x)
            x = MaxPooling2D()(x)
            x = res_block(x, n_channel)
            return x

        def decoder_block(x, n_channel):
            x = Conv2DTranspose(n_channel, kernel_size = 3, strides = 2, padding = 'same')(x)
            x = ReLU()(x)
            x = res_block(x, n_channel)
            return x
        hidden_1 = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        hidden_2 = encoder_block(hidden_1, cs.Unet_channel) # 112 X 112 X 32
        hidden_3 = encoder_block(hidden_2, cs.Unet_channel * 2) # 56 X 56 X 64
        hidden_4 = encoder_block(hidden_3, cs.Unet_channel * 4) # 28 X 28 X 128
        hidden_5 = encoder_block(hidden_4, cs.Unet_channel * 8) # 14 X 14 X 256

        
        self.encoder = Model(inputs = hidden_1, outputs = hidden_5)
        

        hidden_8 = decoder_block(hidden_5, cs.Unet_channel * 4) # 28 X 28 X 128
        hidden_9 = decoder_block(hidden_8, cs.Unet_channel * 2) # 56 X 56 X 64
        hidden_10 = decoder_block(hidden_9, cs.Unet_channel) # 112 X 112 X 32
        result = Conv2DTranspose(cs.img_channel, kernel_size = 3, strides = 2, padding = 'same', activation = 'sigmoid')(hidden_10) # 224 X 224 X 3
        
        self.model = Model(inputs = hidden_1, outputs = result)
        if is_compile:
            self.model.compile(optimizer = op, loss = lo)
            self.model.summary()
    
    def reduce_dim(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = [x[i*batch_s:(i+1)*batch_s] for i in range(num)]
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(1, len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[i])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data.reshape([data.shape[0],-1])
    
    
    def save_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path)
        try:
            if is_denoise:
                self.model.save_weights(self.weight_path + 'model_weight_denoise.h5')
            else:
                self.model.save_weights(self.weight_path + 'model_weight.h5')
            print("save Model weights")
        except:
            pass
        try:
            if is_denoise:
                self.encoder.save_weights(self.weight_path + 'encoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'encoder_weight.h5')
            print("save Encoder weights")
        except:
            pass

    
    def load_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            print(self.weight_path + " does not exist!")
            return
        self.gen_model()
        if is_denoise:
            if os.path.isfile(self.weight_path + 'model_weight_denoise.h5'):
                self.model.load_weights(self.weight_path + 'model_weight_denoise.h5')
                print("Load denoise Model")
            if os.path.isfile(self.weight_path + 'encoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight_denoise.h5')
                print("Load denoise encoder")
        else:
            if os.path.isfile(self.weight_path + 'model_weight.h5'):
                self.model.load_weights(self.weight_path + 'model_weight.h5')
                print("Load Model")
            if os.path.isfile(self.weight_path + 'encoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight.h5')
                print("Load encoder")
    

class Bi_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Bi_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):
        def res_block(x, channel_num, kernel_s = 3, stride = 1):
            # reference : https://arxiv.org/abs/1603.05027
            hidden_1 = x
            hidden_2 = BatchNormalization()(hidden_1)
            hidden_2 = ReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            hidden_2 = BatchNormalization()(hidden_2)
            hidden_2 = ReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            result = Add()([hidden_1, hidden_2])
            return result
    
        def encoder_block(x, n_channel):
            x = Conv2D(n_channel, kernel_size = 3,strides=1, padding = 'same', activation = 'relu')(x)
            x = MaxPooling2D()(x)
            x = res_block(x, n_channel)
            return x

        def decoder_block(x, n_channel):
            x = UpSampling2D(size = 2)(x)
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
            x = res_block(x, n_channel)
            return x   
        def dis_compare(x):
            s = x[0].shape[1]*x[0].shape[2]*x[0].shape[3]
            xx = K.reshape(x[0], [-1, s])
            yy = K.reshape(x[1], [-1, s])
            norm = K.reshape(K.sqrt(K.sum(K.square(xx-yy), axis = -1)), [-1, 1])
            return norm
        hidden_1 = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        hidden_2 = encoder_block(hidden_1, cs.Unet_channel) # 128 X 128 X 32
        hidden_3 = encoder_block(hidden_2, cs.Unet_channel * 2) # 64 X 64 X 64
        hidden_4 = encoder_block(hidden_3, cs.Unet_channel * 4) # 32 X 32 X 128
        hidden_5 = encoder_block(hidden_4, cs.Unet_channel * 8) # 16 X 16 X 256
        #hidden_6 = encoder_block(hidden_5, cs.Unet_channel * 16) # 8 X 8 X 512
        
        self.encoder = Model(inputs = hidden_1, outputs = hidden_5)
        
        #hidden_9 = decoder_block(hidden_6, cs.Unet_channel * 8) # 16 X 16 X 256
        hidden_10 = decoder_block(hidden_5, cs.Unet_channel * 4) # 32 X 32 X 128
        hidden_11 = decoder_block(hidden_10, cs.Unet_channel * 2) # 64 X 64 X 64
        hidden_12 = decoder_block(hidden_11, cs.Unet_channel) # 128 X 128 X 32
        hidden_13 = UpSampling2D(size = 2)(hidden_12)
        result = Conv2D(cs.img_channel, kernel_size = 3, strides = 1, padding = 'same', activation = 'sigmoid')(hidden_13)
        
        ae_model = Model(inputs = hidden_1, outputs = result)

        img_a = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        img_b = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        
        lan_a = (self.encoder)(img_a)
        lan_b = (self.encoder)(img_b)
        
        result_a = ae_model(img_a)
        result_b = ae_model(img_b)
        
        result_n = Lambda(lambda x: dis_compare(x) )([lan_a, lan_b])
        
        self.model = Model(inputs = [img_a, img_b], outputs = [result_a, result_b, result_n])
        self.model.compile(optimizer = op, loss = lo, loss_weights = [50, 50, 1e-3])
        self.model.summary()
    
    def reduce_dim(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = [x[i*batch_s:(i+1)*batch_s] for i in range(num)]
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(1, len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[i])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data.reshape([data.shape[0],-1])
    
    
    def save_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path)
        try:
            if is_denoise:
                self.model.save_weights(self.weight_path + 'model_weight_denoise.h5')
            else:
                self.model.save_weights(self.weight_path + 'model_weight.h5')
            print("save Model weights")
        except:
            pass
        try:
            if is_denoise:
                self.encoder.save_weights(self.weight_path + 'encoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'encoder_weight.h5')
            print("save Encoder weights")
        except:
            pass

    
    def load_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            print(self.weight_path + " does not exist!")
            return
        self.gen_model()
        if is_denoise:
            if os.path.isfile(self.weight_path + 'model_weight_denoise.h5'):
                self.model.load_weights(self.weight_path + 'model_weight_denoise.h5')
                print("Load denoise Model")
            if os.path.isfile(self.weight_path + 'encoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight_denoise.h5')
                print("Load denoise encoder")
        else:
            if os.path.isfile(self.weight_path + 'model_weight.h5'):
                self.model.load_weights(self.weight_path + 'model_weight.h5')
                print("Load Model")
            if os.path.isfile(self.weight_path + 'encoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight.h5')
                print("Load encoder")
   

class Pre_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Pre_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error', is_compile = True):
        def res_block(x, channel_num, kernel_s = 3, stride = 1):
            # reference : https://arxiv.org/abs/1603.05027
            hidden_1 = x
            hidden_2 = BatchNormalization()(hidden_1)
            hidden_2 = LeakyReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            hidden_2 = BatchNormalization()(hidden_2)
            hidden_2 = LeakyReLU()(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            result = Add()([hidden_1, hidden_2])
            return result
    
        def encoder_block(x, n_channel):
            x = Conv2D(n_channel, kernel_size = 3,strides=1, padding = 'same')(x)
            x = LeakyReLU()(x)
            x = MaxPooling2D()(x)
            x = res_block(x, n_channel)
            return x

        def decoder_block(x, n_channel):
            x = Conv2DTranspose(n_channel, kernel_size = 3, strides = 2, padding = 'same')(x)
            x = LeakyReLU()(x)
            x = res_block(x, n_channel)
            return x
        
        def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False): #(reference : https://blog.waya.ai/deep-residual-learning-9610bb62c355)
            shortcut = y

            # down-sampling is performed with a stride of 2
            y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', kernel_initializer='lecun_normal')(y)
            y = BatchNormalization()(y)
            y = Activation('selu')(y)

            y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='lecun_normal')(y)
            y = BatchNormalization()(y)

            # identity shortcuts used directly when the input and output are of the same dimensions
            if _project_shortcut or _strides != (1, 1):
                # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', kernel_initializer='lecun_normal')(shortcut)
                shortcut = BatchNormalization()(shortcut)

            #y = add([shortcut, y])
            y = Add()([shortcut, y])
            y = Activation('selu')(y)

            return y
        
        code_depths = [32, 64, 256, 500]
        strides_x = [2, 2, 5]
        strides_y = [2, 2, 5]
        block_size = [3, 3, 3, 3, 3]
        
        enc_in = Input(shape=(cs.img_height, cs.img_width, cs.img_channel))

        encoded = Conv2D(code_depths[0], kernel_size=(7, 7), strides=(strides_x[0], strides_y[0]), padding='same', kernel_initializer='lecun_normal')(enc_in)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('selu')(encoded)

        encoded = residual_block(encoded, code_depths[0])

        encoded = residual_block(encoded, code_depths[1], _strides=(strides_x[1], strides_y[1]), _project_shortcut=True)

        encoded = residual_block(encoded, code_depths[2], _strides=(strides_x[2], strides_y[2]))

        encoded = Conv2D(code_depths[3], block_size[3], activation='selu', padding='same', kernel_initializer='lecun_normal')(encoded)


        self.encoder = Model(enc_in, encoded)

        #dec_in = decoded = Input(batch_shape=encoded.shape.as_list())

        #decoded = residual_block(decoded, code_depths[2], _project_shortcut=True)
        decoded = residual_block(encoded, code_depths[2], _project_shortcut=True)

        decoded = UpSampling2D((strides_x[2], strides_y[2]))(decoded)

        decoded = residual_block(decoded, code_depths[1], _project_shortcut=True)

        decoded = UpSampling2D((strides_x[1], strides_y[1]))(decoded)

        decoded = residual_block(decoded, code_depths[0], _project_shortcut=True)

        decoded = UpSampling2D((strides_x[0], strides_y[0]))(decoded)

        decoded = Conv2DTranspose(cs.img_channel, block_size[0], activation='sigmoid', padding='same', kernel_initializer='lecun_normal')(decoded)
        #self.decoder = Model(dec_in, decoded)
        
        #encode_output = (self.encoder)(enc_in)
        #output = (self.decoder)(encode_output)
        self.model = Model(enc_in, decoded)
        #self.model = Model(enc_in, output)
        if is_compile:
            self.model.compile(optimizer = op, loss = lo)
            self.model.summary()
    
    def reduce_dim(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = [x[i*batch_s:(i+1)*batch_s] for i in range(num)]
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(1, len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[i])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data.reshape([data.shape[0],-1])
    
    
    def save_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path)
        try:
            if is_denoise:
                self.model.save_weights(self.weight_path + 'model_weight_denoise.h5')
            else:
                self.model.save_weights(self.weight_path + 'model_weight.h5')
            print("save Model weights")
        except:
            pass
        try:
            if is_denoise:
                self.encoder.save_weights(self.weight_path + 'encoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'encoder_weight.h5')
            print("save Encoder weights")
        except:
            pass

    
    def load_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            print(self.weight_path + " does not exist!")
            return
        self.gen_model()
        if is_denoise:
            if os.path.isfile(self.weight_path + 'model_weight_denoise.h5'):
                self.model.load_weights(self.weight_path + 'model_weight_denoise.h5')
                print("Load denoise Model")
            if os.path.isfile(self.weight_path + 'encoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight_denoise.h5')
                print("Load denoise encoder")
        else:
            if os.path.isfile(self.weight_path + 'model_weight.h5'):
                self.model.load_weights(self.weight_path + 'model_weight.h5')
                print("Load Model")
            if os.path.isfile(self.weight_path + 'encoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight.h5')
                print("Load encoder")
    

class Dis_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Dis_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):
        def res_block(x, channel_num, kernel_s = 3, stride = 1):
            # reference : https://arxiv.org/abs/1603.05027
            hidden_1 = x
            hidden_2 = TimeDistributed(BatchNormalization())(hidden_1)
            hidden_2 = TimeDistributed(ReLU())(hidden_2)
            hidden_2 = TimeDistributed(Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same'))(hidden_2)
            hidden_2 = TimeDistributed(BatchNormalization())(hidden_2)
            hidden_2 = TimeDistributed(ReLU())(hidden_2)
            hidden_2 = TimeDistributed(Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same'))(hidden_2)
            result = Add()([hidden_1, hidden_2])
            return result
    
        def encoder_block(x, n_channel):
            x = TimeDistributed(Conv2D(n_channel, kernel_size = 3,strides=1, padding = 'same', activation = "relu"))(x)#, kernel_regularizer = keras.regularizers.l2(0.01)
            x = TimeDistributed(MaxPooling2D())(x)
            x = res_block(x, n_channel)
            return x

        def decoder_block(x, n_channel):
            #x = TimeDistributed(Conv2DTranspose(n_channel, kernel_size = 3, strides = 2, padding = 'same', activation = "relu", kernel_initializer='he_uniform'))(x)
            x = TimeDistributed(Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same', activation = "relu"))(x)
            x = TimeDistributed(UpSampling2D(size=2))(x)
            x = res_block(x, n_channel)
            return x        
        def dis_cal(x):
            '''s = x.shape.as_list()
            length = s[1]
            x = K.reshape(x, [-1,s[1], s[2]*s[3]*s[4]])
            for i in range(length):
                if not K.any(x[:,i]):
                    length = i
                    break
            x = x[:length]
            v1 = 0
            tem = [v1]
            for i in range(length):
                norm = K.sqrt(K.sum(K.square(x-K.reshape(x[:,i],[-1,1,s[2]*s[3]*s[4]])), axis = -1))
                for k in tem:
                    norm[:,k] = np.inf
                v1 = K.argmin(norm, axis=-1)
                tem += [v1]
                
            distance_r = 0
            distance_l = 0
            for xx, yy in itertools.combinations(range(0, length, 2):
                if tem[xx] - tem[yy] > 0:
                    distance_r += 1.0
                if tem[xx] - tem[yy] < 0:
                    distance_l += 1.0
            tem_num = length * (length-1) / 2
            result = K.ones_like(x[:,0])
            err = distance_r/tem_num
            
            return [sequence], err'''
            pass
        def dis_sum(x):
            s = x.shape.as_list()
            x = K.reshape(x,[-1,s[1],s[2]*s[3]*s[4]])
            d = x[:,1:] - x[:,:-1]
            #result = K.reshape(K.sum(tf.norm( d,axis = -1), axis = -1), [-1, 1])
            result = K.reshape(K.sum(K.sqrt(K.sum(K.square(d), axis = -1)), axis = -1), [-1, 1])
            return result
                
        
        hidden_0 = Input(shape=(cs.max_frames_per_data, cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        hidden_2 = encoder_block(hidden_0, cs.Dis_channel) # 128 X 128 X 32
        hidden_3 = encoder_block(hidden_2, cs.Dis_channel * 2) # 64 X 64 X 64
        hidden_4 = encoder_block(hidden_3, cs.Dis_channel * 4) # 32 X 32 X 128
        hidden_5 = encoder_block(hidden_4, cs.Dis_channel * 8) # 16 X 16 X 256
        
        self.encoder = Model(inputs = hidden_0, outputs = hidden_5)
        
        hidden_8 = decoder_block(hidden_5, cs.Dis_channel * 4) # 32 X 32 X 128
        hidden_9 = decoder_block(hidden_8, cs.Dis_channel * 2) # 64 X 64 X 64
        hidden_10 = decoder_block(hidden_9, cs.Dis_channel) # 128 X 128 X 32
        result = TimeDistributed(Conv2DTranspose(cs.img_channel, kernel_size = 3, strides = 2, padding = 'same', activation = 'sigmoid'))(hidden_10)
        
        result_n = Lambda(lambda x: dis_sum(x) )(hidden_5)
        
        self.model = Model(inputs = hidden_0, outputs = [result, result_n])
        #op = Adam(clipvalue=0.5)
        #op='sgd'
        self.model.compile(optimizer = op, loss = lo, loss_weights = [1, 1])
        self.model.summary()
    
    def reduce_dim(self, x, batch_s = 16):
        if x.ndim != 5:
            print("Expect x.ndim is 5, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = [x[i*batch_s:(i+1)*batch_s] for i in range(num)]
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(1, len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[i])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        result = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.any(data[i][j]):
                    result += [data[i][j].flatten()]
        return np.array(result)
    
    
    def save_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path)
        try:
            if is_denoise:
                self.model.save_weights(self.weight_path + 'model_weight_denoise.h5')
            else:
                self.model.save_weights(self.weight_path + 'model_weight.h5')
            print("save Model weights")
        except:
            pass
        try:
            if is_denoise:
                self.encoder.save_weights(self.weight_path + 'encoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'encoder_weight.h5')
            print("save Encoder weights")
        except:
            pass

    
    def load_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            print(self.weight_path + " does not exist!")
            return
        self.gen_model()
        if is_denoise:
            if os.path.isfile(self.weight_path + 'model_weight_denoise.h5'):
                self.model.load_weights(self.weight_path + 'model_weight_denoise.h5')
                print("Load denoise Model")
            if os.path.isfile(self.weight_path + 'encoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight_denoise.h5')
                print("Load denoise encoder")
        else:
            if os.path.isfile(self.weight_path + 'model_weight.h5'):
                self.model.load_weights(self.weight_path + 'model_weight.h5')
                print("Load Model")
            if os.path.isfile(self.weight_path + 'encoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight.h5')
                print("Load encoder")
    

class Fuse_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Fuse_AE/'
        
    
    def gen_model(self, op = 'adam', lo = 'mse', is_compile = True):
        
        def normalize(x):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        def res_block(x, channel_num, kernel_s = 3, stride = 1):
            # reference : https://arxiv.org/abs/1603.05027
            hidden_1 = x
            hidden_2 = normalize(hidden_1)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            hidden_2 = normalize(hidden_2)
            hidden_2 = Conv2D(channel_num, kernel_size = kernel_s, strides = stride, padding = 'same')(hidden_2)
            result = Add()([hidden_1, hidden_2])
            return result
    
        def encoder_block(x, y, n_channel):
            x = normalize(x)
            x = Conv2D(n_channel, kernel_size = 3, strides=1, padding = 'same', activation = 'relu')(x)
            x = MaxPooling2D()(x)
            x = Concatenate()([x,y])
            x = normalize(x)
            x = Conv2D(int(n_channel/2), kernel_size = 1, strides=1, padding = 'same', activation = 'relu')(x)
            x = res_block(x, int(n_channel/2))
            return x

        def decoder_block(x, n_channel):
            x = UpSampling2D(size=2)(x)
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
            #x = res_block(x, n_channel)
            return x       
        
        hidden_inputs = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        
        base_model = keras.applications.vgg19.VGG19(input_tensor = hidden_inputs, weights='imagenet', include_top = False)
        hidden_0_1 = base_model.get_layer('block1_pool').output # 112 X 112 X 64
        hidden_0_2 = base_model.get_layer('block2_pool').output # 56 X 56 X 128
        hidden_0_3 = base_model.get_layer('block3_pool').output # 28 X 28 X 256
        hidden_0_4 = base_model.get_layer('block4_pool').output # 14 X 14 X 512
        
        #hidden_0_5 = base_model.get_layer('block5_pool').output # 7 X 7 X 512
        
        for layer in base_model.layers:
            layer.trainable = False
       
        hidden_1 = encoder_block(hidden_0_1, hidden_0_2, 128) # 56 X 56 X 64
        hidden_2 = encoder_block(hidden_1, hidden_0_3, 256) # 28 X 28 X 128
        encoder_result = encoder_block(hidden_2, hidden_0_4, 512) # 14 X 14 X 256

        
        self.encoder = Model(inputs = hidden_inputs, outputs = encoder_result)

        
        hidden_8 = decoder_block(encoder_result, cs.Unet_channel * 4) # 28 X 28 X 128
        hidden_9 = decoder_block(hidden_8, cs.Unet_channel * 2) # 56 X 56 X 64
        hidden_10 = decoder_block(hidden_9, cs.Unet_channel ) # 112 X 112 X 32
        hidden_11 = UpSampling2D(size=2)(hidden_10)
        result = Conv2D(cs.img_channel, kernel_size = 3, strides = 1, padding = 'same', activation = 'sigmoid')(hidden_11) # 224 X 224 X 3

        self.model = Model(inputs = hidden_inputs, outputs = result)
        if is_compile:
            self.model.compile(optimizer = op, loss = lo)
            self.model.summary()
    
    def reduce_dim(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return


        all_batch_size = x[0].shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = [ x[i*batch_s:(i+1)*batch_s] for i in range(num)]
            remain = [xx[num*batch_s:] for xx in x]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(1, len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[i])])
            if remain[0].shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data.reshape([data.shape[0],-1])
        
        
        return self.encoder.predict
    
    def fit_training(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 10, is_denoise = True, noise_factor = 0.3, max_epoch_num = 1000):
        if len(x_train[0]) != len(y_train[0]):
            print("len of x and y is not equal!")
            return
        
        
        #'''
        if is_denoise:
            x = [(xx + noise_factor * np.random.normal(size = xx.shape)).clip(0.0,1.0) for xx in x_train]
        else:
            x = x_train
        #'''

        #'''
        self.model.fit(x, y_train, batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        #'''
        self.save_mod()
    
    
    
    def save_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            os.makedirs(self.weight_path)
        try:
            if is_denoise:
                self.model.save_weights(self.weight_path + 'model_weight_denoise.h5')
            else:
                self.model.save_weights(self.weight_path + 'model_weight.h5')
            print("save Model weights")
        except:
            pass
        try:
            if is_denoise:
                self.encoder.save_weights(self.weight_path + 'encoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'encoder_weight.h5')
            print("save Encoder weights")
        except:
            pass
        try:
            if is_denoise:
                self.decoder.save_weights(self.weight_path + 'decoder_weight_denoise.h5')
            else:
                self.decoder.save_weights(self.weight_path + 'decoder_weight.h5')
            print("save Decoder weights")
        except:
            pass

    
    def load_mod(self, is_denoise = False):
        if not os.path.isdir(self.weight_path):
            print(self.weight_path + " does not exist!")
            return
        self.gen_model()
        if is_denoise:
            if os.path.isfile(self.weight_path + 'model_weight_denoise.h5'):
                self.model.load_weights(self.weight_path + 'model_weight_denoise.h5')
                print("Load denoise Model")
            if os.path.isfile(self.weight_path + 'encoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight_denoise.h5')
                print("Load denoise encoder")
            if os.path.isfile(self.weight_path + 'decoder_weight_denoise.h5'):
                self.encoder.load_weights(self.weight_path + 'decoder_weight_denoise.h5')
                print("Load denoise decoder")
        else:
            if os.path.isfile(self.weight_path + 'model_weight.h5'):
                self.model.load_weights(self.weight_path + 'model_weight.h5')
                print("Load Model")
            if os.path.isfile(self.weight_path + 'encoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'encoder_weight.h5')
                print("Load encoder")
            if os.path.isfile(self.weight_path + 'decoder_weight.h5'):
                self.encoder.load_weights(self.weight_path + 'decoder_weight.h5')
                print("Load decoder")
    
