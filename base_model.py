
import numpy as np
import random
from random import shuffle
import time
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, LeakyReLU, BatchNormalization, Add, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Conv2D, Concatenate, Lambda, ReLU, Activation
import keras.backend as K
from data_generator import Data_Generator
import constants as cs
from heapq import nlargest as nl
import os
import gc
import itertools
#import psutil

class Base_Model():
    def __init__(self):
        pass
    
    def gen_model(self):
        pass
    
    def training(self, x_train, y_train, validation_ratio = 0.1, patience_num = 5, is_denoise = True, noise_factor = 0.5, max_epoch_num = 1000):
        x = x_train
        x_ground = y_train
        validation_num = int(len(x) * validation_ratio)
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
                x[i] = x[i].clip(0.0,1.0)
        x = list(zip(x, x_ground))
        if validation_num != 0:
            x_v, x_v_ground = zip(*x[-validation_num:])
            x = x[:-validation_num]
        fail_num = 0
        pre_v_loss = pre_t_loss = -1
        for i in range(max_epoch_num):
            start = time.time()
            random.shuffle(x)
            x_t, x_t_ground = zip(*x)
            t_loss = np.array(self.model.train_on_batch(x_t[0], x_t_ground[0]))
            for j in range(1,len(x_t)):
                print("t : ", t_loss)
                t_loss += np.array(self.model.train_on_batch(x_t[j], x_t_ground[j]))
            t_loss /= len(x_t)
            if t_loss.ndim != 0:
                t_loss = t_loss[0]
            if validation_num == 0:
                train_loss = np.array(self.model.test_on_batch(x_t[0], x_t_ground[0]))
                for j in range(1,len(x_t)):
                    train_loss += np.array(self.model.test_on_batch(x_t[j], x_t_ground[j]))
                train_loss /= len(x_t)
                if train_loss.ndim != 0:
                    train_loss = train_loss[0]
                print("Epoch ", i, " : train_loss = ", train_loss, ", time : ", (time.time() - start), " s", flush = True)
                #early stopping and restore weights at best result
                if pre_t_loss <= train_loss and pre_t_loss != -1:
                    fail_num += 1
                    if fail_num == patience_num:
                        self.model.set_weights(w)
                        print("Early stopping at epoch : ", i , " !")
                        break
                else:
                    w = self.model.get_weights()
                    fail_num = 0
                    pre_t_loss = train_loss
            else:
                v_loss = np.array(self.model.test_on_batch(x_v[0], x_v_ground[0]))
                for j in range(1,len(x_v)):
                    v_loss += np.array(self.model.test_on_batch(x_v[j], x_v_ground[j]))
                v_loss /= len(x_v)
                if v_loss.ndim != 0:
                    v_loss = v_loss[0]
                print("Epoch ", i, " : train_loss = ", t_loss, ", valid_loss = ", v_loss, ", time : ", (time.time() - start), " s", flush = True)
                #early stopping and restore weights at best result
                if pre_v_loss <= v_loss and pre_v_loss != -1:
                    fail_num += 1
                    if fail_num == patience_num:
                        self.model.set_weights(w)
                        print("Early stopping at epoch : ", i , " !")
                        break
                else:
                    w = self.model.get_weights()
                    fail_num = 0
                    pre_v_loss = v_loss
        self.save_mod()
    
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
        
        '''
        validation_num = max(int(len(x[0]) * validation_ratio), 1)
        x_v = [xx[-validation_num:] for xx in x]
        x = [xx[:-validation_num] for xx in x]
        y_v = [yy[-validation_num:] for yy in y_train]
        y = [yy[:-validation_num] for yy in y_train]
        
        if len(x) == 1:
            x = x[0]
            x_v = x_v[0]
            num = int(len(x[0])/batch_s)
            x_batch = [x[i*batch_s:(i+1)*batch_s] for i in range(num)] + [x[num*batch_s:]]
            num = int(len(x_v[0])/batch_s)
            x_v_batch = [x_v[i*batch_s:(i+1)*batch_s] for i in range(num)] + [x_v[num*batch_s:]]
        else:
            num = int(len(x[0])/batch_s)
            x_batch = [[xx[i*batch_s:(i+1)*batch_s] for xx in x] for i in range(num)] + [[xx[num*batch_s:] for xx in x]]
            num = int(len(x_v[0])/batch_s)
            x_v_batch = [[xx[i*batch_s:(i+1)*batch_s] for xx in x_v] for i in range(num)] + [[xx[num*batch_s:] for xx in x_v]]
        print("\n", len(x_batch))
        if len(y) == 1:
            y = y[0]
            y_v = y_v[0]
            num = int(len(x[0])/batch_s)
            y_batch = [y[i*batch_s:(i+1)*batch_s] for i in range(num)] + [y[num*batch_s:]]
            num = int(len(x_v[0])/batch_s)
            y_v_batch = [y_v[i*batch_s:(i+1)*batch_s] for i in range(num)] + [y_v[num*batch_s:]]
        else:
            num = int(len(x[0])/batch_s)
            y_batch = [[yy[i*batch_s:(i+1)*batch_s] for yy in y] for i in range(num)] + [[yy[num*batch_s:] for yy in y]]
            num = int(len(x_v[0])/batch_s)
            y_v_batch = [[yy[i*batch_s:(i+1)*batch_s] for yy in y_v] for i in range(num)] + [[yy[num*batch_s:] for yy in y_v]]

        def data_gen(x, y):
            xy = zip(x, y)
            while(True):
                for d in xy:
                    yield d
     
        self.model.fit_generator(generator = data_gen(x_batch, y_batch), steps_per_epoch = len(x_batch),
                epochs = max_epoch_num, callbacks = [EarlyStopping(patience=patience_num, restore_best_weights=True)],
                validation_data = data_gen(x_v_batch, y_v_batch), validation_steps = len(x_v_batch))
        '''
        #'''
        self.model.fit(x, y_train, batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        #'''
        self.save_mod()
    
    def testing(self, x_test, y_test, is_denoise = True, noise_factor = 0.5):
        x = x_test
        if is_denoise:
            for i in range(len(x)):
                x[i] = x[i] + noise_factor * np.random.normal(size = x[i].shape)
        loss = self.model.test_on_batch(x, y_test)
        print("testing loss : ", loss)
    
    def save_mod(self):
        pass
    



class Tri_Model():
    def __init__(self, b):
        self.base = b
        id = self.base.weight_path[:-1].rfind('/')
        self.weight_path = self.base.weight_path[:id] + '/Tri_' + self.base.weight_path[id+1:]
    def gen_model(self, op = 'adam', lo = 'mean_squared_error'):
        def dis_compare(x):
            s = x[0].shape[1]*x[0].shape[2]*x[0].shape[3]
            xx = K.reshape(x[0], [-1, s])
            yy = K.reshape(x[1], [-1, s])
            zz = K.reshape(x[2], [-1, s])
            norm_1 = K.sqrt(K.sum(K.square(xx-yy), axis = -1))
            norm_2 = K.sqrt(K.sum(K.square(xx-zz), axis = -1))
            result = K.reshape(norm_1 - norm_2, [-1,1])
            return result
        def tf_binary_entropy(target, output):
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
        
        tem_model = self.base.model

        img_a = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        img_b = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        img_c = Input(shape=(cs.img_height, cs.img_width, cs.img_channel)) # 224 X 224 X 3
        
        self.encoder = self.base.encoder
        #self.decoder = self.base.decoder
        lan_a = (self.base.encoder)(img_a)
        lan_b = (self.base.encoder)(img_b)
        lan_c = (self.base.encoder)(img_c)
        
        result_a = tem_model(img_a)
        result_b = tem_model(img_b)
        result_c = tem_model(img_c)
        
        result_n = Lambda(lambda x: dis_compare(x) )([lan_a, lan_b, lan_c])
        #op = SGD(lr = 1e-3, momentum = 0.9, nesterov = True)
        self.model = Model(inputs = [img_a, img_b, img_c], outputs = [result_a, result_b, result_c, result_n])
        self.model.compile(optimizer = op, loss = ["mse", "mse", "mse", tf_binary_entropy], loss_weights = [1,1,1,1])
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
        
    def fit_training(self, x_train, y_train, batch_s = 16, validation_ratio = 0.05, patience_num = 10, is_denoise = True, noise_factor = 0.3, max_epoch_num = 1000, is_reproduce = False):
        if len(x_train[0]) != len(y_train[0]):
            print("len of x and y is not equal!")
            return
        print("data preprocessing")
        
        def togray(x):
            return np.dot(x, [0.2989, 0.5870, 0.1140])
        #'''
        if is_denoise:
            x_train = [(xx + noise_factor * np.random.normal(size = xx.shape)).clip(0.0,1.0) for xx in x_train]
        else:
            x = x_train
        #'''
        if (not is_reproduce) and os.path.isfile("data/training/tem_data/x_0.npy") and os.path.isfile("data/training/tem_data/data_a_id.npy") and os.path.isfile("data/training/tem_data/data_b_id.npy") and os.path.isfile("data/training/tem_data/data_c_id.npy"):
            data_a_id = np.load("data/training/tem_data/data_a_id.npy")
            data_b_id = np.load("data/training/tem_data/data_b_id.npy")
            data_c_id = np.load("data/training/tem_data/data_c_id.npy")
        else:
            #print(psutil.Process().memory_info().rss/(2**30))
            n = len(x_train)
            id = np.array([0] + list(itertools.accumulate([len(x_train[i]) for i in range(n)])))
            #print(psutil.Process().memory_info().rss/(2**30))
            d = Data_Generator()
            x = d.convert_frame_data_to_data(x)
            y = d.convert_frame_data_to_data(y_train)
            del d
            #gc.collect()
            #print(psutil.Process().memory_info().rss/(2**30))
            repeat_time = 5 # should at least 2
            data_a_id = np.repeat(np.delete(np.arange(len(x)), id[1:]-1), repeat_time)
            #print(psutil.Process().memory_info().rss/(2**30))
            data_b_id = np.repeat(np.delete(np.arange(len(x)), id[:-1]), repeat_time)
            #print(psutil.Process().memory_info().rss/(2**30))
            data_c_id = []
            for i in range(n):
                for j in range(id[i], id[i+1]-1):
                    ids = nl(repeat_time, list(enumerate(np.mean(np.square(togray(y[id[i]:id[i+1]] - y[j])).reshape([(id[i+1]-id[i]),-1]), axis=-1))), key=lambda x: x[1])
                    data_c_id += [s[0]+id[i] for s in ids]
                    del ids
                #gc.collect()
            data_c_id = np.array(data_c_id)
            np.save("data/training/tem_data/data_a_id.npy", data_a_id)
            np.save("data/training/tem_data/data_b_id.npy", data_b_id)
            np.save("data/training/tem_data/data_c_id.npy", data_c_id)
        from random import shuffle
        all = list(zip(data_a_id, data_b_id, data_c_id))
        shuffle(all)
        data_a_id, data_b_id, data_c_id = zip(*all)
        del all
        #gc.collect()
        data_a_id = np.array(data_a_id)
        data_b_id = np.array(data_b_id)
        data_c_id = np.array(data_c_id)

        #print(psutil.Process().memory_info().rss/(2**30))
        if is_reproduce or not os.path.isfile("data/training/tem_data/x_0.npy"):
            for i in range(len(x)):
                np.save("data/training/tem_data/x_" + str(i) + ".npy", x[i][np.newaxis])
                np.save("data/training/tem_data/y_" + str(i) + ".npy", y[i][np.newaxis])
            del x
            del y
        gc.collect()
        #np.save("z.npy", z)
        #np.save("a_id.npy", data_a_id)
        #np.save("b_id.npy", data_b_id)
        #np.save("c_id.npy", data_c_id)
        
        #'''
        validation_num = max(int(len(data_a_id) * validation_ratio), 1)
        data_a_id_v = data_a_id[-validation_num:]
        data_b_id_v = data_b_id[-validation_num:]
        data_c_id_v = data_c_id[-validation_num:]
        data_a_id = data_a_id[:-validation_num]
        data_b_id = data_b_id[:-validation_num]
        data_c_id = data_c_id[:-validation_num]
        #print(len(data_a_id), " ", len(z), " ", np.concatenate([z[3776:], z[:2]]).shape)
        
        def data_gen(a, b, c, batch_size):
            start = 0
            n = a.shape[0]
            epoch_num = int(len(a)/batch_size)
            while(True):
                all = list(zip(a, b, c))
                shuffle(all)
                tem_a, tem_b, tem_c = zip(*all)
                del all
                tem_a = np.array(tem_a)
                tem_b = np.array(tem_b)
                tem_c = np.array(tem_c)
                
                data_d_id = np.arange(n)
                tem_n = int(n/2)
                shuffle(data_d_id)
                data_d_id = data_d_id[tem_n]
                z = np.zeros([n])
                z[data_d_id] = 1
                tem_id = tem_b[data_d_id]
                tem_b[data_d_id] = tem_c[data_d_id]
                tem_c[data_d_id] = tem_id
                
                for i in range(epoch_num):
                    a_id = a[i*batch_size:(i+1)*batch_size]
                    b_id = b[i*batch_size:(i+1)*batch_size]
                    c_id = c[i*batch_size:(i+1)*batch_size]
                    x_a = np.load("data/training/tem_data/x_" + str(a_id[0]) + ".npy")
                    x_b = np.load("data/training/tem_data/x_" + str(b_id[0]) + ".npy")
                    x_c = np.load("data/training/tem_data/x_" + str(c_id[0]) + ".npy")
                    y_a = np.load("data/training/tem_data/y_" + str(a_id[0]) + ".npy")
                    y_b = np.load("data/training/tem_data/y_" + str(b_id[0]) + ".npy")
                    y_c = np.load("data/training/tem_data/y_" + str(c_id[0]) + ".npy")
                    for j in range(1, batch_size):
                        x_a = np.concatenate([x_a, np.load("data/training/tem_data/x_" + str(a_id[j]) + ".npy")], axis = 0)
                        x_b = np.concatenate([x_b, np.load("data/training/tem_data/x_" + str(b_id[j]) + ".npy")], axis = 0)
                        x_c = np.concatenate([x_c, np.load("data/training/tem_data/x_" + str(c_id[j]) + ".npy")], axis = 0)
                        y_a = np.concatenate([y_a, np.load("data/training/tem_data/y_" + str(a_id[j]) + ".npy")], axis = 0)
                        y_b = np.concatenate([y_b, np.load("data/training/tem_data/y_" + str(b_id[j]) + ".npy")], axis = 0)
                        y_c = np.concatenate([y_c, np.load("data/training/tem_data/y_" + str(c_id[j]) + ".npy")], axis = 0)
                    yield ([x_a, x_b, x_c], [y_a, y_b, y_c, z[i*batch_size:(i+1)*batch_size]])
                
        def v_data_gen(a, b, c, batch_size):
            start = 0
            n = a.shape[0]
            epoch_num = int(len(a)/batch_size)
            while(True):
                all = list(zip(a, b, c))
                shuffle(all)
                tem_a, tem_b, tem_c = zip(*all)
                del all
                tem_a = np.array(tem_a)
                tem_b = np.array(tem_b)
                tem_c = np.array(tem_c)
                
                data_d_id = np.arange(n)
                tem_n = int(n/2)
                shuffle(data_d_id)
                data_d_id = data_d_id[tem_n]
                z = np.zeros([n])
                z[data_d_id] = 1
                tem_id = tem_b[data_d_id]
                tem_b[data_d_id] = tem_c[data_d_id]
                tem_c[data_d_id] = tem_id
                
                for i in range(epoch_num):
                    a_id = a[i*batch_size:(i+1)*batch_size]
                    b_id = b[i*batch_size:(i+1)*batch_size]
                    c_id = c[i*batch_size:(i+1)*batch_size]
                    x_a = np.load("data/training/tem_data/x_" + str(a_id[0]) + ".npy")
                    x_b = np.load("data/training/tem_data/x_" + str(b_id[0]) + ".npy")
                    x_c = np.load("data/training/tem_data/x_" + str(c_id[0]) + ".npy")
                    y_a = np.load("data/training/tem_data/y_" + str(a_id[0]) + ".npy")
                    y_b = np.load("data/training/tem_data/y_" + str(b_id[0]) + ".npy")
                    y_c = np.load("data/training/tem_data/y_" + str(c_id[0]) + ".npy")
                    for j in range(1, batch_size):
                        x_a = np.concatenate([x_a, np.load("data/training/tem_data/x_" + str(a_id[j]) + ".npy")], axis = 0)
                        x_b = np.concatenate([x_b, np.load("data/training/tem_data/x_" + str(b_id[j]) + ".npy")], axis = 0)
                        x_c = np.concatenate([x_c, np.load("data/training/tem_data/x_" + str(c_id[j]) + ".npy")], axis = 0)
                        y_a = np.concatenate([y_a, np.load("data/training/tem_data/y_" + str(a_id[j]) + ".npy")], axis = 0)
                        y_b = np.concatenate([y_b, np.load("data/training/tem_data/y_" + str(b_id[j]) + ".npy")], axis = 0)
                        y_c = np.concatenate([y_c, np.load("data/training/tem_data/y_" + str(c_id[j]) + ".npy")], axis = 0)
                    yield ([x_a, x_b, x_c], [y_a, y_b, y_c, z[i*batch_size:(i+1)*batch_size]])
 
     
        self.model.fit_generator(generator = data_gen(data_a_id, data_b_id, data_c_id, batch_s), steps_per_epoch = int(len(data_a_id)/batch_s),
                epochs = max_epoch_num, callbacks = [EarlyStopping(patience=patience_num, restore_best_weights=True), ReduceLROnPlateau(patience = 3, factor = 0.5, min_lr = 1e-12, verbose = 1, mode = 'min')],
                validation_data = v_data_gen(data_a_id_v, data_b_id_v, data_c_id_v, batch_s), validation_steps = int(len(data_a_id_v)/batch_s), max_queue_size = 1, workers = 1)
        #'''
        
        
        '''
        self.model.fit([x[data_a_id], x[data_b_id], x[data_c_id]], [y[data_a_id], y[data_b_id], y[data_c_id], z], batch_size = batch_s, epochs = max_epoch_num, 
                validation_split = validation_ratio, callbacks=[EarlyStopping(patience=patience_num, restore_best_weights=True)])
        '''
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
                self.encoder.save_weights(self.weight_path + 'decoder_weight_denoise.h5')
            else:
                self.encoder.save_weights(self.weight_path + 'decoder_weight.h5')
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
    
    