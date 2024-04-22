import numpy as np
import cv2
import os
import errno
import constants as cs
import shutil
import pickle
import imageio
import itertools
import gc

class Data_Generator():
    def __init__(self):
        pass
    
    def load_video_to_data(self, video_path = "video", is_training = False, is_np = True, is_split = False, is_contain_remainder = True, is_gen_inter = False, is_seperate = False, is_reproduce = False):
        data, origin, path = self.load_video_to_frame(is_training, video_path = video_path, is_reproduce = is_reproduce)
        if is_seperate:
            data = self.seperate_data(data)
        if is_np:
            data = self.convert_frame_data_to_data(data)
        else:
            if is_split:
                data, origin, path = self.split_frame_data(data, origin, path, is_contain_remainder)
            if is_gen_inter:
                data = self.gen_inter_data(data)
        if is_training:
            return data
        else:
            return data, origin, path
    
    def seperate_data(self, data, split = 2):
        h, w = img_height//split, img_width//split
        new_data = []
        for d in data:
            new_d = []
            for dd in d:
                for i in range(split):
                    for j in range(split):
                        new_d += [dd[i*h:(i+1)*h, j*w:(j+1)*w]]
            new_data += [new_d]
        return new_data
    
    def gen_inter_data(self, d):
        data_x = []
        data_y = []
        for dd in d:
            data_xx = []
            data_yy = []
            for i in range(dd.shape[0]-2):
                data_xx += np.concatenate([dd[i],dd[i+2]])
                data_yy = [dd[i+1]]
            data_x += [np.array(data_xx)]
            data_y += [np.array(data_yy)]
        return data_x, data_y
    
    def split_frame_data(self, d, o, p, is_contain_remainder = True):
        data = []
        origin = []
        path = []
        for i in range(len(d)):
            s = d[i].shape[0]
            l_d = []
            l_o = []
            l_p = []
            num = s-cs.max_frames_per_data+1
            if num > 0:
                l_d = [d[i][i:i+cs.max_frames_per_data] for i in range(num)]
                l_o = [o[i][i:i+cs.max_frames_per_data] for i in range(num)]
            elif is_contain_remainder:
                tem = np.zeros((cs.max_frames_per_data, )+d[i].shape[1:])
                tem[:len(d[i])] = d[i]
                l_d = [tem]
                tem = np.zeros((cs.max_frames_per_data, )+o[i].shape[1:])
                tem[:len(o[i])] = o[i]
                l_o = [tem]
            
            data += l_d
            origin += l_o
            path += [p[i]]*len(l_d)
        return data, origin, path
    
    def merge_split_frame_data(self, d, o, p):
        data = []
        origin = []
        path = [p[0]]
        tem_data = []
        tem_origin_data = []
        for i in range(len(d)):
            if i != 0 and p[i-1] != p[i]:
                path += [p[i]]
                if len(tem_data) == 1:
                    data += [tem_data[0][np.newaxis]]
                    origin += [tem_origin_data[0][np.newaxis]]
                else:
                    data += [np.array(tem_data)]
                    origin += [np.array(tem_origin_data)]
                del tem_data, tem_origin_data
                gc.collect()
                tem_data = []
                tem_origin_data = []
            tem_data += [d[i]]
            tem_origin_data += [o[i]]
        
        if len(tem_data) != 0:
            path += [p[-1]]
            if len(tem_data) == 1:
                data += [tem_data[0][np.newaxis]]
                origin += [tem_origin_data[0][np.newaxis]]
            else:
                data += [np.array(tem_data)]
                origin += [np.array(tem_origin_data)]
        return data, origin, path

    def merge_split_video_data(self, d, o, p):
        data = []
        origin = []
        tem_data = []
        tem_origin_data = []
        for i in range(len(d)):
            for j in range(len(d[i])):
                if np.any(d[i][j]):
                    tem_data += [d[i][j]]
                    tem_origin_data += [o[i][j]]
            data += [np.array(tem_data)]
            origin += [np.array(tem_origin_data)]

        return data, origin, p

    def convert_frame_data_to_data(self, d):
        data = []
        for dd in d:
            if dd.ndim > 1:
                for ddd in dd:
                    data += [ddd]
            else:
                data += [dd]
        return np.array(data)
    
    def load_video_to_frame(self, is_training = False, video_path = "video", is_reproduce = False, is_split = False):
        path = "data/testing/"
        if is_split:
            video_path = "video_split"
        path_video = path + video_path + "/"
        dirs = os.listdir(path_video)
        if is_split:
            if not os.path.isdir(path + "split_image/"):
                os.makedirs(path + "split_image/")
        else:
            if not os.path.isdir(path + "image/"):
                os.makedirs(path + "image/")
        data = []
        origin = []
        paths = dirs
        for d in dirs:
            path_dir = path_video + d
            cap = cv2.VideoCapture(path_dir)
            if (cap.isOpened() == False): 
                print("Can't open file ", path_dir, " !")
                break
            success, img = cap.read()
            counter = 0
            id = 0
            if is_split:
                path_frame = path + "split_image/" + d + "_split/"
            else:
                path_frame = path + "image/" + d + "/"
            is_already_produce = False
            if os.path.isdir(path_frame):
                if is_reproduce:
                    shutil.rmtree(path_frame)
                    os.makedirs(path_frame)
                else:
                    is_already_produce = True
            else:
                os.makedirs(path_frame)
            images = []
            origin_img = []
            
            if is_already_produce:
                imgs = os.listdir(path_frame)
                imgs = [im for im in imgs if im[-4:] == ".jpg"]
                imgs.sort(key = lambda x : int(x[:-4]))
                for im in imgs:
                    img = cv2.imread(path_frame + im)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #origin_img += [img[:,:]]
                    img = cv2.resize(img, (cs.img_width, cs.img_height), interpolation = cv2.INTER_LINEAR)
                    origin_img += [img[:,:]]
                    images += [img[:,:]/255]
            else:
                while success:
                    counter += 1
                    if counter == cs.frame_num_per_key_frame:
                        counter = 0
                        cv2.imwrite(path_frame + str(id) + ".jpg", img)
                        img = cv2.imread(path_frame + str(id) + ".jpg")
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        #origin_img += [img[:,:]]#[cv2.resize(img, (1024, 768), interpolation = cv2.INTER_LINEAR)]
                        img = cv2.resize(img, (cs.img_width, cs.img_height), interpolation = cv2.INTER_LINEAR)
                        origin_img += [img[:,:]]
                        images += [img[:,:]/255]
                        id += 1
                    success, img = cap.read()
            # release object
            cap.release()
            images = np.array(images)
            origin_img = np.array(origin_img)
            data += [images]
            origin += [origin_img]
        return data, origin, paths
    
    def load_frame_to_data(self):
        pass
    
    def sequence_to_video(self, sequence, video_name, fps = 5.0, anime_num = 5):
        path = "data/testing/"
        path_frame = path + "image/"
        dirs = os.listdir(path_frame)
        path_result = path + "result/"
        if not os.path.isdir(path_result):
            os.makedirs(path_result)
        
        d = video_name
        images = os.listdir(path_frame + d + "/")
        if not images:
            print("There is no images in ", path_frame + d + "/", " !")
            return
        images = [im for im in images if im[-4:] == ".jpg"]
        images.sort(key = lambda x: int(x[:-4]))
        img_shape = cv2.imread(path_frame + d + "/" + images[0]).shape
         
        data = []
        for i in range(len(images)):
            frame = cv2.imread(path_frame + d + "/" + images[i])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.data:
                data += [frame]
            else:
                print("Can't open file ", path_frame + d + "/" + images[i], " !")
                break
        
        if os.path.isdir(path_result+d+'/'):
            for dd in os.listdir(path_result+d+'/'):
                os.remove(path_result+d+'/'+dd)
        else:
            os.makedirs(path_result+d+'/')
        
        for i in range(anime_num):
            writer = imageio.get_writer(path_result + d + '/output_'+str(i)+'.mp4',fps=fps)
            for j in sequence[i]:
                writer.append_data(data[j])
            writer.close()
    
    
    
    
if __name__ == '__main__':
    d = Data_Generator()
    _ = d.load_video_to_frame(is_training = False, video_path = "video_split", is_reproduce = True, is_split = True)
    
    
    
    
    
    