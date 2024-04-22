
import numpy as np
import random
import keras
from keras.models import Model
from keras.layers import Input, LeakyReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Conv2D, Concatenate, Lambda, CuDNNLSTM, Reshape, TimeDistributed, ConvLSTM2D, Masking
from keras.optimizers import Adam
import keras.backend as K

import constants as cs
from base_model import Base_Model
import os


# reference : https://arxiv.org/pdf/1711.10684.pdf
class RNN_Encoder(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/RAE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):
    
        def encoder_block(x, n_channel):
            x = ConvLSTM2D(filters = n_channel, kernel_size = 3, strides = 1, padding = 'same', return_sequences = True, activation = 'relu')(x)
            x = TimeDistributed(MaxPooling2D())(x)
            return x
        def decoder_block(x, n_channel):
            x = ConvLSTM2D(filters = n_channel, kernel_size = 3, strides = 1, padding = 'same', return_sequences = True, activation = 'relu')(x)
            x = TimeDistributed(UpSampling2D(size=2))(x)
            return x
        def dis_sum(x):
            s = x.shape.as_list()
            x = K.reshape(x,[-1,s[1],s[2]*s[3]*s[4]])
            d = x[:,1:] - x[:,:-1]
            result = K.reshape(K.sum(K.sqrt(K.sum(K.square(d), axis = -1)), axis = -1), [-1, 1])
            return result
        

        inputs = Input(shape=(cs.max_frames_per_data, cs.img_height, cs.img_width, cs.img_channel)) # 16 X 256 X 256 X 3
        #hidden_1 = Masking(mask_value = 0)(inputs)
        hidden_2 = encoder_block(inputs, cs.RAE_channel * 1) # 16 X 128 X 128 X 32
        hidden_3 = encoder_block(hidden_2, cs.RAE_channel * 2) # 16 X 64 X 64 X 64
        hidden_4 = encoder_block(hidden_3, cs.RAE_channel * 4) # 16 X 32 X 32 X 128
        #hidden_5 = encoder_block(hidden_4, cs.RAE_channel * 8) # 16 X 16 X 16 X 256
        
        #self.encoder = Model(inputs = inputs, outputs = hidden_5)
        #result_n = Lambda(lambda x: dis_sum(x) )(hidden_5)
        
        #hidden_6 = decoder_block(hidden_5, cs.RAE_channel * 8) # 16 X 32 X 32 X 256
        #hidden_7 = decoder_block(hidden_6, cs.RAE_channel * 4) # 16 X 64 X 64 X 128
        self.encoder = Model(inputs = inputs, outputs = hidden_4)
        result_n = Lambda(lambda x: dis_sum(x) )(hidden_4)
        hidden_7 = decoder_block(hidden_4, cs.RAE_channel * 4) # 16 X 64 X 64 X 128
        
        
        hidden_8 = decoder_block(hidden_7, cs.RAE_channel * 2) # 16 X 128 X 128 X 64
        hidden_9 = decoder_block(hidden_8, cs.RAE_channel * 1) # 16 X 256 X 256 X 32
        hidden_10 = ConvLSTM2D(filters = cs.img_channel, kernel_size = 3, strides = 1, padding = 'same', return_sequences = True, activation = 'sigmoid')(hidden_9) # 16 X 256 X 256 X 3
        
        

        self.model = Model(inputs = inputs, outputs = [hidden_10, result_n])
        self.model.compile(optimizer = op, loss = lo, loss_weights = [1000, 1e-3])
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
        
        
if __name__ == '__main__':
    r = RNN_Encoder()
    r.gen_model()