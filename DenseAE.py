
import numpy as np
import random
import keras
from keras.models import Model
from keras.layers import Input, LeakyReLU, BatchNormalization, Add, MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, Flatten, Conv2D, Concatenate, Lambda, ReLU, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K

import constants as cs
from base_model import Base_Model
import os


# reference : https://arxiv.org/pdf/1711.10684.pdf
class Dense_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Dense_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):

        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        sess = tf.Session(config=config)

        KTF.set_session(sess)
        
        def normalize(x):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x    
        def dense_layer(x, n_channel):
            hidden = normalize(x)
            hidden = Conv2D(n_channel*4, kernel_size = 1, strides = 1, padding = 'same')(hidden)
            hidden = Dropout(0.2)(hidden)
            hidden = normalize(hidden)
            hidden = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same')(hidden)
            hidden = Dropout(0.2)(hidden)
            hidden = Concatenate()([x, hidden])
            return hidden
        def transition_layer(x, compression = 0.5):
            x = normalize(x)
            n = int(x.shape.as_list()[-1] * compression)
            x = Conv2D(n, kernel_size = 1, strides = 1, padding = 'same')(x)
            x = AveragePooling2D()(x)
            return x
        def dense_block(x, n_channel, num = 7):
            for i in range(num):
                x = dense_layer(x, n_channel)
            return x
        def encoder_block(x, n_channel, num = 7):
            x = dense_block(x, n_channel, num)
            x = transition_layer(x)
            return x
        def decoder_block(x, n_channel):
            x = normalize(x)
            x = UpSampling2D(size = 2)(x)
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same')(x)
            return x
        
        def dis_sum(x):
            s = x.shape.as_list()
            x = K.reshape(x,[-1,s[1]*s[2]*s[3]])
            d = x[1:] - x[:-1]
            norm = K.sqrt(K.sum(K.square(d), axis = -1))
            result = K.ones_like(x[:,0])
            result = K.reshape(result, [-1, 1])
            result *= K.sum(norm)
            return result
        
        hidden_1 = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        hidden_2 = Conv2D(cs.Dense_channel*2, kernel_size = 7, strides = 2, padding = 'same')(hidden_1) # 112 X 112 X 64
        hidden_3 = MaxPooling2D()(hidden_2) # 56 X 56 X 64
        hidden_4 = encoder_block(hidden_3, cs.Dense_channel, 2) # 28 X 28 X 64
        hidden_5 = encoder_block(hidden_4, cs.Dense_channel, 6) # 14 X 14 X 128
        hidden_6 = encoder_block(hidden_5, cs.Dense_channel, 12) # 7 X 7 X 256
        
        hidden_8 = dense_block(hidden_6, cs.Dense_channel, 8) # 7 X 7 X 512
        
        self.encoder = Model(inputs = hidden_1, outputs = hidden_8)
        
        hidden_9 = decoder_block(hidden_8, cs.Dense_channel * 8) # 14 X 14 X 256
        hidden_10 = decoder_block(hidden_9, cs.Dense_channel * 4) # 28 X 28 X 128
        hidden_11 = decoder_block(hidden_10, cs.Dense_channel * 2) # 56 X 56 X 64
        hidden_12 = decoder_block(hidden_11, cs.Dense_channel) # 112 X 112 X 32
        hidden_13 = UpSampling2D(size = 2)(hidden_12) # 224 X 224 X 32
        result = Conv2D(cs.img_channel, kernel_size = 3, padding = 'same', activation = 'sigmoid')(hidden_13) # 224 X 224 X 3

        self.model = Model(inputs = hidden_1, outputs = result)
        self.model.compile(optimizer = op, loss = lo)
        self.model.summary()
        
        '''
        self.encoder.trainable = False
        decoder_input = Input(shape=(int(cs.img_height / 16), int(cs.img_width / 16), cs.Unet_channel * 8))
        hidden_6 = decoder_block(decoder_input, cs.Unet_channel * 4) # 32 X 18 X 128
        hidden_7 = decoder_block(hidden_6, cs.Unet_channel * 2) # 64 X 36 X 64
        hidden_8 = decoder_block(hidden_7, cs.Unet_channel) # 128 X 72 X 32
        decoder_output = Conv2DTranspose(cs.img_channel, kernel_size = 3, strides = 2, padding = 'same', activation = 'sigmoid')(hidden_8)
        self.decoder = Model(inputs = decoder_input, outputs = decoder_output)
        self.decoder.compile(optimizer = op, loss = lo)
        self.decoder.summary()
        '''
    
    '''
    def train_decoder(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 20, 
                        is_denoise = True, noise_factor = 0.5, max_epoch_num = 1000, op = 'adam', lo = 'mean_squared_error'):
        x = x_train
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
                x[i] = x[i].clip(0.0,1.0)
        self.decoder.fit(self.encode(x), y_train, batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        self.save_mod()
    '''
    '''
    def fit_training(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 20, is_denoise = True, noise_factor = 0.3, max_epoch_num = 1000):
        x = x_train
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
                x[i] = x[i].clip(0.0,1.0)
        self.model.fit(x, [y_train, np.zeros(len(y_train))], batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        self.save_mod()
    '''
    
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
    
    '''
    def encode(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = np.split(x[:num*batch_s], num)
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[1])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data
    
    
    def decode(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.decoder.predict_on_batch(remain)
        else:
            d = np.split(x[:num*batch_s], num)
            remain = x[num*batch_s:]
            data = self.decoder.predict_on_batch(d[0])
            for i in range(len(d)):
                data = np.concatenate([data, self.decoder.predict_on_batch(d[1])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.decoder.predict_on_batch(remain)])
        return data
    '''
    
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
        '''
        try:
            self.model.save_weights(self.weight_path + 'decoder_weight.h5')
            print("save Decoder weights")
        except:
            pass
        '''
    
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
        '''
        if os.path.isfile(self.weight_path + 'decoder_weight.h5'):
            self.decoder.load_weights(self.weight_path + 'decoder_weight.h5')
            print("Load decoder")
        '''



# reference : https://arxiv.org/pdf/1711.10684.pdf
class Bi_Dense_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Bi_Dense_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):
        
        def normalize(x):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x    
        def dense_layer(x, n_channel):
            hidden = normalize(x)
            hidden = Conv2D(n_channel*4, kernel_size = 1, strides = 1, padding = 'same')(hidden)
            hidden = Dropout(0.2)(hidden)
            hidden = normalize(hidden)
            hidden = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same')(hidden)
            hidden = Dropout(0.2)(hidden)
            hidden = Concatenate()([x, hidden])
            return hidden
        def transition_layer(x, compression = 0.5):
            x = normalize(x)
            n = int(x.shape.as_list()[-1] * compression)
            x = Conv2D(n, kernel_size = 1, strides = 1, padding = 'same')(x)
            x = AveragePooling2D()(x)
            return x
        def dense_block(x, n_channel, num = 7):
            for i in range(num):
                x = dense_layer(x, n_channel)
            return x
        def encoder_block(x, n_channel, num = 7):
            x = dense_block(x, n_channel, num)
            x = transition_layer(x)
            return x
        def decoder_block(x, n_channel):
            x = normalize(x)
            x = UpSampling2D(size = 2)(x)
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same')(x)
            return x
        
        def dis_sum(x):
            s = x.shape.as_list()
            x = K.reshape(x,[-1,s[1]*s[2]*s[3]])
            d = x[1:] - x[:-1]
            norm = K.sqrt(K.sum(K.square(d), axis = -1))
            result = K.ones_like(x[:,0])
            result = K.reshape(result, [-1, 1])
            result *= K.sum(norm)
            return result
        
        hidden_1 = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        hidden_2 = Conv2D(cs.Dense_channel*2, kernel_size = 7, strides = 2, padding = 'same')(hidden_1) # 112 X 112 X 64
        hidden_3 = MaxPooling2D()(hidden_2) # 56 X 56 X 64
        hidden_4 = encoder_block(hidden_3, cs.Dense_channel, 2) # 28 X 28 X 64
        hidden_5 = encoder_block(hidden_4, cs.Dense_channel, 6) # 14 X 14 X 128
        hidden_6 = encoder_block(hidden_5, cs.Dense_channel, 12) # 7 X 7 X 256
        
        hidden_8 = dense_block(hidden_6, cs.Dense_channel, 8) # 7 X 7 X 512
        
        self.encoder = Model(inputs = hidden_1, outputs = hidden_8)
        
        hidden_9 = decoder_block(hidden_8, cs.Dense_channel * 8) # 14 X 14 X 256
        hidden_10 = decoder_block(hidden_9, cs.Dense_channel * 4) # 28 X 28 X 128
        hidden_11 = decoder_block(hidden_10, cs.Dense_channel * 2) # 56 X 56 X 64
        hidden_12 = decoder_block(hidden_11, cs.Dense_channel) # 112 X 112 X 32
        hidden_13 = UpSampling2D(size = 2)(hidden_12) # 224 X 224 X 32
        result = Conv2D(cs.img_channel, kernel_size = 3, padding = 'same', activation = 'sigmoid')(hidden_13) # 224 X 224 X 3

        tem_model = Model(inputs = hidden_1, outputs = result)
        
        img_a = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        img_b = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 256 X 256 X 3
        
        lan_a = (self.encoder)(img_a)
        lan_b = (self.encoder)(img_b)
        
        result_a = tem_model(img_a)
        result_b = tem_model(img_b)
        
        result_n = Lambda(lambda x: dis_compare(x) )([lan_a, lan_b])
        
        self.model = Model(inputs = [img_a, img_b], outputs = [result_a, result_b, result_n])
        self.model.compile(optimizer = op, loss = lo, loss_weights = [500, 500, 1e-3])
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
            data = self.encoder.predict_on_batch(x[:batch_s])
            for i in range(1, num):
                data = np.concatenate([data, self.encoder.predict_on_batch(x[i*batch_s: (i+1)*batch_s])])
            remain = x[num*batch_s:]
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



# reference : https://arxiv.org/pdf/1711.10684.pdf
class Fuse_Dense_AE(Base_Model):
    def __init__(self):
        self.weight_path = 'data/model_weight/Fuse_Dense_AE/'
    
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):


        def normalize(x):
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x    
        def fuse_block(x, y, n_channel):
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(x)
            x = MaxPooling2D()(x)
            x = Concatenate()([x, y])
            return x
        def decoder_block(x, n_channel):
            x = normalize(x)
            x = UpSampling2D(size = 2)(x)
            x = Conv2D(n_channel, kernel_size = 3, strides = 1, padding = 'same')(x)
            return x
        
        def dis_sum(x):
            s = x.shape.as_list()
            x = K.reshape(x,[-1,s[1]*s[2]*s[3]])
            d = x[1:] - x[:-1]
            norm = K.sqrt(K.sum(K.square(d), axis = -1))
            result = K.ones_like(x[:,0])
            result = K.reshape(result, [-1, 1])
            result *= K.sum(norm)
            return result
        
        hidden_1 = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        
        base_model = keras.applications.densenet.DenseNet121(input_tensor = hidden_1, weights='imagenet', include_top = False, input_shape = (224,224,3))
        feature_1 = base_model.get_layer('pool2_relu').output # 56 X 56 X 256
        feature_2 = base_model.get_layer('pool3_relu').output # 28 X 28 X 512
        feature_3 = base_model.get_layer('pool4_relu').output # 14 X 14 X 1024
        feature_4 = base_model.get_layer('relu').output # 7 X 7 X 1024
        
        hidden_2 = fuse_block(feature_1, feature_2, 512) # 28 X 28 X 1024
        hidden_3 = Conv2D(512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(hidden_2) # 28 X 28 X 256
        hidden_4 = fuse_block(hidden_3, feature_3, 1024) # 14 X 14 X 2048
        hidden_5 = Conv2D(1024, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(hidden_4) # 14 X 14 X 512
        hidden_6 = fuse_block(hidden_5, feature_4, 1024) # 7 X 7 X 2048
        hidden_7 = Conv2D(512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu')(hidden_6) # 7 X 7 X 512
        
        self.encoder = Model(inputs = hidden_1, outputs = hidden_7)
        
        hidden_9 = decoder_block(hidden_7, cs.Dense_channel * 8) # 14 X 14 X 256
        hidden_10 = decoder_block(hidden_9, cs.Dense_channel * 4) # 28 X 28 X 128
        hidden_11 = decoder_block(hidden_10, cs.Dense_channel * 2) # 56 X 56 X 64
        hidden_12 = decoder_block(hidden_11, cs.Dense_channel) # 112 X 112 X 32
        hidden_13 = UpSampling2D(size = 2)(hidden_12) # 224 X 224 X 32
        result = Conv2D(cs.img_channel, kernel_size = 3, padding = 'same', activation = 'sigmoid')(hidden_13) # 224 X 224 X 3

        self.model = Model(inputs = hidden_1, outputs = result)
        self.model.compile(optimizer = op, loss = lo)
        self.model.summary()
        
        for layer in base_model.layers:
            layer.trainable = False
        
        '''
        self.encoder.trainable = False
        decoder_input = Input(shape=(int(cs.img_height / 16), int(cs.img_width / 16), cs.Unet_channel * 8))
        hidden_6 = decoder_block(decoder_input, cs.Unet_channel * 4) # 32 X 18 X 128
        hidden_7 = decoder_block(hidden_6, cs.Unet_channel * 2) # 64 X 36 X 64
        hidden_8 = decoder_block(hidden_7, cs.Unet_channel) # 128 X 72 X 32
        decoder_output = Conv2DTranspose(cs.img_channel, kernel_size = 3, strides = 2, padding = 'same', activation = 'sigmoid')(hidden_8)
        self.decoder = Model(inputs = decoder_input, outputs = decoder_output)
        self.decoder.compile(optimizer = op, loss = lo)
        self.decoder.summary()
        '''
    
    '''
    def train_decoder(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 20, 
                        is_denoise = True, noise_factor = 0.5, max_epoch_num = 1000, op = 'adam', lo = 'mean_squared_error'):
        x = x_train
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
                x[i] = x[i].clip(0.0,1.0)
        self.decoder.fit(self.encode(x), y_train, batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        self.save_mod()
    '''
    '''
    def fit_training(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 20, is_denoise = True, noise_factor = 0.3, max_epoch_num = 1000):
        x = x_train
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
                x[i] = x[i].clip(0.0,1.0)
        self.model.fit(x, [y_train, np.zeros(len(y_train))], batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        self.save_mod()
    '''
    
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
    
    '''
    def encode(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.encoder.predict_on_batch(remain)
        else:
            d = np.split(x[:num*batch_s], num)
            remain = x[num*batch_s:]
            data = self.encoder.predict_on_batch(d[0])
            for i in range(len(d)):
                data = np.concatenate([data, self.encoder.predict_on_batch(d[1])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.encoder.predict_on_batch(remain)])
        return data
    
    
    def decode(self, x, batch_s = 16):
        if x.ndim != 4:
            print("Expect x.ndim is 4, but get ", x.ndim)
            return
        all_batch_size = x.shape[0]
        num = int(all_batch_size/batch_s)
        if num == 0:
            remain = x
            data = self.decoder.predict_on_batch(remain)
        else:
            d = np.split(x[:num*batch_s], num)
            remain = x[num*batch_s:]
            data = self.decoder.predict_on_batch(d[0])
            for i in range(len(d)):
                data = np.concatenate([data, self.decoder.predict_on_batch(d[1])])
            if remain.shape[0] != 0:
                data = np.concatenate([data, self.decoder.predict_on_batch(remain)])
        return data
    '''
    
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
        '''
        try:
            self.model.save_weights(self.weight_path + 'decoder_weight.h5')
            print("save Decoder weights")
        except:
            pass
        '''
    
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
        '''
        if os.path.isfile(self.weight_path + 'decoder_weight.h5'):
            self.decoder.load_weights(self.weight_path + 'decoder_weight.h5')
            print("Load decoder")
        '''



if __name__ == "__main__":
    import pickle
    m = Res_Unet()
    #m = Simple_AE()
    m.load_mod()
    test_data = pickle.load(open("reduce_data.txt", "rb"))
    latent_vector = []
    for i in range(len(test_data)):
        latent_vector += [m.reduce_dim(test_data[i], batch_s = cs.Unet_batch)]
    pickle.dump(latent_vector, open("reduce_result.txt", "wb"))