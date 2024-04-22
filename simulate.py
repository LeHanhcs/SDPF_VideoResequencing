from keras.backend import shape
from numpy.core.fromnumeric import size
import tensorflow as tf
import keras
from data_generator import Data_Generator
import numpy as np
from simple_AE import Fuse_AE
from DenseAE import Dense_AE, Bi_Dense_AE
from base_model import Tri_Model
import matplotlib.pyplot as plt
from path_explorer import Path_Explorer
import constants as cs
from scipy.stats import kendalltau
import random
import utils
from PIL import Image
import gc

from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as LEM
from sklearn.manifold import Isomap as ISO
from sklearn.preprocessing import StandardScaler as standardize


utils.gpu_limit()


def model_choice(model, is_load, neighbors_n = 2, components_n = 2):    
    is_tranditional = False   
    
    model == "Tri_Fuse_AE"
    m = Fuse_AE()
    m.gen_model(is_compile = False)
    m = Tri_Model(m)       
    return m

def video_resequence(model_name, is_noise = True, video_path = "video", given_direction = None, anime_num = 30, start_frame = 28, key_frame_list = None, inter_num = -1, steps = None, fps = 7.0, mode = 1, model = None, is_reproduce = False):
    import time
    t=time.time()
    
    d = Data_Generator()
    test_data, test_origin, paths = d.load_video_to_data(is_training = False, is_np = False, video_path = video_path)
    
    m = model_choice(model_name, True)   
    latent_vector = []   
    
    for i in range(len(test_data)):        
        latent_vector += [m.reduce_dim(test_data[i], batch_s = cs.Unet_batch)]    

    p = Path_Explorer()            
    
    for i in range(len(test_data)):       
        sequence = p.produce_resequence(test_origin[i], latent_vector[i], paths[i], given_direction = given_direction, anime_num = anime_num, start_frame = start_frame, steps = steps, mode = mode, is_reproduce = is_reproduce, model = m, key_frame_list = key_frame_list, inter_num = inter_num)
        idx_sequence = 0
        for sq in sequence:
            print(idx_sequence, ": ", sq)
            idx_sequence+=1
        d.sequence_to_video(sequence, paths[i], fps, anime_num)        
    print('cost : ', time.time()-t, ' seconds.')

    del m
    return -1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", dest = "mode", default = 0, type = int)
    parser.add_argument("-n", "--model_name", dest = "model_name", default = "Tri_Fuse_AE", type = str)
    parser.add_argument("-r", "--reproduce", dest = "is_reproduce", default = False, action = "store_true")
    parser.add_argument("-s", "--start_frame", dest = "start_frame", default = 0, type = int)
    parser.add_argument("-l", "--key_list", dest = "key_frame_list", default = None, type = int, nargs='+')
    parser.add_argument("-i", "--inter_num", dest = "inter_num", default = -1, type = int)
    args = parser.parse_args()
    modes = args.mode
    model_name = args.model_name
    start_frame = args.start_frame
    is_reproduce = args.is_reproduce
    key_frame_list = args.key_frame_list
    inter_num = args.inter_num    
    if modes == 1: # testing mode
        video_resequence(model_name,False, mode = 1, video_path = "video_testing", given_direction = None, start_frame = start_frame, is_reproduce = is_reproduce)
    