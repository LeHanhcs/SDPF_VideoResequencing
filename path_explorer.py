
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import shortest_path
import random
import itertools
import cv2
import constants as cs
from sklearn.metrics.pairwise import cosine_similarity
from skimage.measure import compare_ssim
from numpy.linalg import norm
from heapq import nsmallest as ns
from skimage.measure import compare_ssim, compare_psnr
from scipy.stats import kendalltau

import pickle
import utils
import os

#reference by previous autoencoder work
class Path_Explorer():
    def __init__(self):
        pass
    
    def produce_resequence(self, test_x, x, video_name, given_direction = None, anime_num = 5, start_frame = 0, steps = None, mode = 0, key_frame_list = None, inter_num = -1, is_reproduce = False, model = None):
        path = "data/testing/image/" + video_name + "/"
        self.origin = test_x
        self.build_distance_graph(x)
        self.model = model
        if mode == 0:
            self.graph = self.build_graph(path = path, start_frame = start_frame)
            return self.mo_explore(anime_num, start_frame, steps)
        elif mode == 1:
            self.dis = utils.clean_image(path, self.origin, self.dis, is_reproduce = False)
            utils.image_optflow(path, self.origin, self.dis, is_reproduce = False)
            self.graph = self.build_graph(path = path, start_frame = start_frame)
            return self.my_explore(path, given_direction, anime_num, start_frame, steps, is_reproduce = is_reproduce)       
        
    
    def build_distance_graph(self, x):
        self.vertex_num = len(x)
        if x.ndim != 2:
            self.pos = np.reshape(x, [self.vertex_num,-1])
        else:
            self.pos = np.copy(x)
        self.dis = np.zeros((self.vertex_num, self.vertex_num))
        for i in range(self.vertex_num):
            self.dis[i, :] = norm(self.pos - self.pos[i], axis=1)
    
    def build_direction(self, path):
        #self.orginal_sequence = [0,20,24,29,33,37,46]
        #if not os.path.isfile(path+"test_opt_dir.npy"): #################
        if os.path.isfile(path+"test_opt_dir.npy"): #################
            given_direction = np.full([self.vertex_num], -10, dtype = "float")
            for i in range(len(self.orginal_sequence)-1):
                with open(path+"opt_"+str(self.orginal_sequence[i])+".txt", "rb") as f:
                    tem_optical = pickle.load(f)
                    for j in range(i+1, len(self.orginal_sequence)):
                        if not tem_optical[self.orginal_sequence[j]] is None:
                            given_direction[self.orginal_sequence[i]] = utils.optical_direction(tem_optical[self.orginal_sequence[j]],-10)
                            break
            
        else:
            given_direction = np.load(path+"test_opt_dir.npy")
    
        dir_choice = np.zeros_like(self.orginal_sequence)
        print(given_direction)
        for i in range(len(self.orginal_sequence)):
            if given_direction[self.orginal_sequence[i]] == -10:
                continue
            k = 0
            j = i
            while (j+1 < len(self.orginal_sequence)):
                j+=1
                if given_direction[self.orginal_sequence[j]] == -10:
                    continue
                else:
                    if utils.direction_diff(given_direction[self.orginal_sequence[j]],given_direction[self.orginal_sequence[i]]) <= 45*np.pi/180:
                        k += 1
                    else:
                        break
            if k < 2:
                if dir_choice[i] == 0:
                    given_direction[self.orginal_sequence[i]] = -10
            else:
                for k in range(i,j):
                    dir_choice[k] = 1
        
        return given_direction
    
    def my_interpolation(self, path, given_direction = None, key_frame_list = None, inter_num = -1, anime_num = 5):
        if inter_num < 1 or key_frame_list is None:
            return None
        if given_direction is None:
            given_direction = self.build_direction(path)
        result = []
        
        for i in range(len(key_frame_list)):
            if np.sum(self.dis[key_frame_list[i]]) == 0:
                while(np.sum(self.dis[key_frame_list[i]]) == 0):
                    key_frame_list[i] -= 1
        print("key_frame_list = ", key_frame_list)
		
        for i in range(anime_num):
            sequence = [key_frame_list[0]]
            for j in range(len(key_frame_list)-1):
                if j == 0:
                    sequence += self.start_end_interpolation(key_frame_list[j], key_frame_list[j+1], path, given_direction, inter_num, pre_v = -1)
                else:
                    sequence += self.start_end_interpolation(key_frame_list[j], key_frame_list[j+1], path, given_direction, inter_num, pre_v = key_frame_list[j-1])
                
                sequence += [key_frame_list[j+1]]
            result += [sequence]
            print('Number ', i, ' : ')
            print(sequence)
        
        tt = [len(ss) for ss in result]
        print(np.argmax(tt), " ", np.max(tt))
        return result

    def start_end_interpolation(self, start, end, path, given_direction, inter_num = -1, pre_v = -1):
        result = []
        num_limit = 10
        tem_s = utils.dfs(self.mst, start, end)
        print(tem_s)
     
        if inter_num == -1:
            tem_num = len(tem_s)-2
        else:
            tem_num = min(len(tem_s)-2, inter_num)
        pre_id = pre_v
        pre_arg_id = 0
        v1 = tem_s[0]
        if tem_num > 0:
            tem_d = int((len(tem_s)-1)/(tem_num+1))
        for k in range(tem_num):
            
            tem_id = tem_s[pre_arg_id+1:k-tem_num]
            
            tem_mean = np.mean(self.dis[v1][tem_id])
            
            tem = [ [tem_id[l], self.dis[v1][tem_id[l]], 0, pre_arg_id+l+1] for l in range(len(tem_id)) if self.dis[v1][tem_id[l]] <= tem_mean]
            #tem = ns(num_limit, tem, key = lambda x:x[1])
            print([t[0] for t in tem])
            with open(path+"opt_"+str(v1)+".txt", "rb") as f:
                tem_optical = pickle.load(f)
                tem = [t for t in tem if not (given_direction[v1] != -10 and given_direction[t[0]] != -10 and utils.direction_diff(given_direction[t[0]], given_direction[v1]) > (np.pi/3))]
                print([t[0] for t in tem])                
                tem = [t for t in tem if not (given_direction[v1] != -10 and utils.optical_direction(tem_optical[t[0]],given_direction[v1], model = self.model) > (np.pi/3))]
                print([t[0] for t in tem])                
                tem = [t for t in tem if not (given_direction[t[0]] != -10 and utils.optical_direction(tem_optical[t[0]],given_direction[t[0]], model = self.model) > (np.pi/3))]
                print([t[0] for t in tem])

            if len(tem) == 0:
                continue

            
            if pre_id != -1:
                with open(path+"opt_"+str(v1)+".txt", "rb") as f:
                    tem_optical = pickle.load(f)
                    opt_eps_list = []
                    opt_eps, mag_eps = utils.optical_eps(tem_optical[pre_id], model = self.model)
                    for l in range(len(tem)):
                        tem_result = utils.optical_dis(tem_optical[pre_id], tem_optical[tem[l][0]], model = self.model, mag_eps = mag_eps)
                        tem[l][2] = tem_result[0]
                        opt_eps_list += [tem_result[1]]
                    #opt_eps, mag_eps = utils.optical_eps(tem_optical[pre_v], eps = np.max(opt_eps_list), model = self.model)
                
                print(k, " num = ", [t[2] for t in tem])
                temp_list = [t[2] for t in tem if t[2] != 0]
                #if len(temp_list) != 0:
                if len(temp_list) <= 2:
                    tem_mean = opt_eps
                elif len(temp_list) != 0:
                    tem_mean = np.mean(temp_list)
                    
                tem = [t for t in tem if t[2] <= tem_mean]
                print(k, " num = ", [t[2] for t in tem])
                if len(tem) == 0:
                    continue
                else:
                    num = np.array([t[2] for t in tem])
            else:
                if len(tem) == 0:
                    continue
                else:
                    num = np.array([t[1] for t in tem])
            
            id = [t[0] for t in tem]
            arg_id = [t[3] for t in tem]
            if num.min() == num.max():
                num = np.ones_like(num)
            else:
                num = (num-num.min())/(num.max()-num.min())
            s_num = np.exp(-num)
            s_num = s_num/np.sum(s_num)
            v2 = np.random.choice(len(id), p = s_num)
            pre_arg_id = arg_id[v2]
            pre_id = v1
            v1 = id[v2]
            result += [v1]
            #'''
        return result

    
    def my_explore(self, path, given_direction = None, anime_num = 5, start_frame = 0, steps = None, is_reproduce = False):
        print("is_reproduce is ", is_reproduce)
        cycles = [False]*anime_num
        if given_direction is None:
            given_direction = self.build_direction(path)
        if np.sum(self.dis[start_frame]) == 0:
            while(np.sum(self.dis[start_frame]) == 0):
                start_frame -= 1
        num_limit = 10#max(10,int(len(orgi)/10))#10
        optflow_eps = 120*np.pi/180        
        is_already_index = os.path.isfile(path+"index.txt")
               
        is_already_index = False      
        
        np.set_printoptions(suppress = True)
        
        if is_already_index and (not is_reproduce):
            with open(path+"index.txt", "rb") as f:
                self.index_graph = pickle.load(f)
        else:
            temp_optic = np.zeros((self.vertex_num, self.vertex_num))
            self.index_graph = []
            tem_dis_mean = []
            for i in range(self.vertex_num):
                tem_dis_mean += [np.mean(self.dis[i][np.nonzero(self.dis[i])[0]])]  
            
            
            for i in range(self.vertex_num):
                temp = []
                for j in range(self.vertex_num):
                    if 0 < self.dis[i,j]:# < tem_dis_mean[i]:
                        temp += [[j, self.dis[i, j], 0, 0]]
                self.index_graph += [temp]
                tem_id = [t[0] for t in temp]
                temp_optic[i,tem_id] = 1
                temp_optic[tem_id,i] = 1
            with open(path+"index.txt", "wb") as f:
                pickle.dump(self.index_graph, f)
        
        for i in range(self.vertex_num):
            if num_limit > 2:
                pass#self.index_graph[i] = ns(num_limit, self.index_graph[i], key = lambda x:x[1])
        print("index done")      
       
        
        tem_id_min = []
        for i in range(self.vertex_num):
            ttt = [t[1] for t in self.index_graph[i]]
            if len(ttt) > 0:
                tem_id_min += [np.min(ttt)]
            else:
                tem_id_min += [1]
        
        #min_possible_num = 5
        #remain_possible_ratio = 0.3
        second_frame = -1#start_frame+2#-1
        anime_sequence = []
        if steps is None:
            steps = self.vertex_num
        #i = 0
        for i in range(anime_num):
            anime_path = []
            E_total = 0
            pre_v = -1
            v1 = start_frame
            anime_path += [v1]
            tem_weight = np.ones(self.vertex_num, dtype='float')
            tem_weight[v1] += 1
            print(given_direction)
            for j in range(steps):                
                reverse_state = False
                tem = [t for t in self.index_graph[v1] if t[0] != pre_v]
                tem_mean = np.mean([t[1] for t in tem])
                tem = [t for t in tem if t[1] <= tem_mean and tem_weight[t[0]] <= 3]

                if given_direction is None:
                    if num_limit > 0:
                        tem = ns(num_limit, tem, key = lambda x:x[1])
                        tem_clique = ns(num_limit, tem, key = lambda x:x[1])                
                else:                    
                    if num_limit > 0:
                        tem = ns(num_limit, tem, key = lambda x:x[1])
                        tem_clique = ns(num_limit, tem, key = lambda x:x[1])     
                    print([t[0] for t in tem])
                    
                    with open(path+"opt_"+str(v1)+".txt", "rb") as f:
                        tem_optical = pickle.load(f)
                        tem = [t for t in tem if not (given_direction[v1] != -10 and given_direction[t[0]] != -10 and utils.direction_diff(given_direction[t[0]], given_direction[v1]) > (np.pi/3))]
                        print([t[0] for t in tem])                        
                        tem = [t for t in tem if not (given_direction[v1] != -10 and utils.optical_direction(tem_optical[t[0]],given_direction[v1], model = self.model) > (np.pi/3))]
                        print([t[0] for t in tem])                        
                        tem = [t for t in tem if not (given_direction[t[0]] != -10 and utils.optical_direction(tem_optical[t[0]],given_direction[t[0]], model = self.model) > (np.pi/3))]
                        print([t[0] for t in tem])
                        
                print(v1)
                
                if len(tem) == 0:
                    print("mean early stop at ", j, " steps!\n")
                    break               
                
                if len(tem) == 0:
                    print("same direction early stop at ", j, " steps!\n")
                    break
                if pre_v != -1:
                    opt_eps_list = []
                    with open(path+"opt_"+str(v1)+".txt", "rb") as f:
                        tem_optical = pickle.load(f)
                        opt_eps, mag_eps = utils.optical_eps(tem_optical[pre_v], model = self.model)                       
                        for k in range(len(tem)):                           
                            tem_result = utils.optical_dis(tem_optical[pre_v], tem_optical[tem[k][0]], model = self.model, mag_eps = mag_eps)
                            tem[k][2] = tem_result[0]
                            opt_eps_list += [tem_result[1]]                            
                        del tem_optical                       
                    tem_id = [t[0] for t in tem]
                    print("tem_id : ", tem_id)
                    
                    k_num = [t[2] for t in tem]
                    print("k_num : ", k_num)
                    
                    temp_list = [t[2] for t in tem if t[2] != 0]# and t[2] <= opt_eps]
                    #if len(temp_list) != 0:
                    if len(temp_list) <= 1:
                        if len(temp_list) == 2:
                            tem_mean = min(np.mean(temp_list), opt_eps)
                        else:
                            tem_mean = opt_eps
                        tem_mean = opt_eps
                    elif len(temp_list) != 0:
                        tem_mean = np.mean(temp_list)
                    
                    tem = [t for t in tem if t[2] <= tem_mean]                    
                    tem_id = [t[0] for t in tem]
                    p_num = np.array([t[1] for t in tem])
                    num = np.array([t[2] for t in tem])
                    print("num : ", num)
                    print("lantent: ", p_num)
                    if len(tem) == 0:
                        print("optical flow early stop at ", j, " steps!\n")
                        break
                    #num = [t[3] for t in tem]
                else:
                    p_num = np.array([t[1] for t in tem])
                    num = np.array([t[1] for t in tem])
                    #num = np.ones_like(p_num)
                
                #'''
                if num.max()-num.min() != 0:
                    num = (num-num.min())/(num.max()-num.min()) # min max normalize
                else:
                    num = np.full(num.shape, 1/(len(num)))
                
                id = [t[0] for t in tem]
                print("id : ", id)
                print()
                #id, num = zip(*tem)
                if reverse_state:
                    random_dis = np.exp(num)
                else:
                    random_dis = np.exp(-num)
                random_dis = random_dis/sum(random_dis) # softmax
             
                print("random_dis : ", random_dis)
                choose_twice = False
                if choose_twice:
                    v2 = np.random.choice(len(tem), 2, p = random_dis)
                    print("v2 = ", v2)
                    if p_num[v2[0]] > p_num[v2[1]]:
                        v2 = v2[1]
                    else:
                        v2 = v2[0]
                else:
                    v2 = np.random.choice(len(tem), p = random_dis)
                    print("v2 = ", v2)
                
                is_inter = False
                if p_num[v2]/tem_id_min[v1] >= 2:
                    is_inter = True
                
                is_inter = False
                if is_inter:
                    anime_path += self.start_end_interpolation(v1, id[v2], path, given_direction, inter_num = 2, pre_v = pre_v)
                
                user_input_bool = False
                
                if pre_v == -1 and second_frame != -1:
                    v2 = second_frame
                    pre_v = v1
                    v1 = v2
                else:
                    pre_v = v1
                    v1 = id[v2]
                    
                    if user_input_bool:
                        user_input = int(input("input:"))
                        if user_input != -1:
                            v1 = user_input
                    
                    tem_weight[v1] += 1
                    for t in tem_clique:
                        if t[0] != v1:
                            tem_weight[t[0]] += 0.3#(?
                anime_path += [v1]    
                E_total += num[v2]
            
            #if len(anime_path) <= 7:
            #    i -= 1
            #    continue
            #print(anime_path)
            #anime_path = self.my_interpolation(path, key_frame_list = anime_path, inter_num = 3, anime_num = 1)[0]
            is_cycle = False
            if is_cycle:
                cycle_path = self.start_end_interpolation(anime_path[-1], anime_path[0], path, given_direction, inter_num = -1, pre_v = anime_path[-2])
                anime_path += cycle_path
                cycles[i] = len(cycle_path) != 0
            anime_sequence += [anime_path]
            print('Number ', i, ' : ')
            #print('Total Energy : ' + str(E_total/steps))
            print(anime_path)
        print(cycles)
        for i in range(len(cycles)):
            if cycles[i]:
                anime_sequence[i] = anime_sequence[i]*3
        return anime_sequence
    
    def build_graph(self, path, start_frame):
        orgi, _ = self.origin_explore(path, start_frame = start_frame)
        self.orginal_sequence = orgi[0]
        self.mst = minimum_spanning_tree(self.dis).toarray()
        self.short_path = shortest_path(self.dis, directed = False, return_predecessors = True)[1]
        self.max_edge = np.max(self.mst)
        self.mst = self.mst + np.transpose(self.mst)
        return (self.dis * (self.dis<=self.max_edge))
    
    
    def mo_explore(self, anime_num, start_frame = 0, steps = None):
        weight_Edisp = 0 #weight of distance
        weight_Es = 1 #weight of angle
        step = steps
        if step is None:
            step = self.vertex_num
            
        anime_sequence = []
        anime_total_cost = np.inf
        for i in range(anime_num):
            anime_path = []
            E_total = 0

            v_1 = start_frame
            anime_path += [v_1]
            
            neighbors = (self.graph[v_1]>0).nonzero()[0]
            v_2 = random.choice(neighbors)
            
            anime_path += [v_2]
            v_i = v_1
            v_j = v_2

            for j in range(step - 2):  # anime_step - 2 => because the first two frames are determined
                dir_pre = self.pos[v_j] - self.pos[v_i]

                v_k_sets = (self.graph[v_j]>0).nonzero()[0]
                Idx_Pexp_pair_list = []
                E_list = []

                for v_k in v_k_sets:
                    dir_next = self.pos[v_k] - self.pos[v_j]
                    Es = norm(dir_pre - dir_next)  ##########################################
                    Edisp = norm(self.pos[v_k] - self.pos[v_j])

                    E = weight_Edisp * Edisp + weight_Es * Es
                    E_list += [E]

                    P_exp = np.exp(-E / self.max_edge)
                    Idx_Pexp_pair_list += [[v_k, P_exp]]

                prob = random.uniform(0.0, 1.0)  # sample a random number between 0.0 and 1.0

                Idx_Pexp_pair_list = sorted(Idx_Pexp_pair_list, key=lambda x: x[1], reverse=True)
                sum_of_P_exp = 0
                for pair in Idx_Pexp_pair_list:
                    sum_of_P_exp = sum_of_P_exp + pair[1]
                Aij = 1.0 / sum_of_P_exp

                sample_prob_list = []
                total_prob = 1.0
                for pair in Idx_Pexp_pair_list:
                    total_prob = total_prob - Aij * pair[1]
                    sample_prob_list += [total_prob]

                for k in range(len(sample_prob_list)):
                    if (prob > sample_prob_list[k]):
                        next_frame = Idx_Pexp_pair_list[k][0]
                        break

                anime_path += [next_frame]
                E_total = E_total + E_list[k]

                v_i = v_j
                v_j = next_frame
            
            anime_sequence += [anime_path]
            print('Number ', i, ' : ')
            print('Total Energy : ' + str(E_total))
            print(anime_path)
        
        print('')
        return anime_sequence
        
    def origin_explore(self, path, start_frame):
        sequence = []
        tem = np.copy(self.dis)       
        tem[tem==0] = np.inf
        v1 = start_frame
        for i in range(self.vertex_num):
            v2 = np.argmin(tem[v1])
            if tem[v1, v2] == np.inf : 
                sequence += [v1]
                break
            tem[v1] = np.inf
            tem[:,v1] = np.inf
            sequence += [v1]
            v1 = v2
        print(sequence)
        print('')
        
        #err = np.mean([sequence[i]!=i for i in range(self.vertex_num)])
        #err = (kendalltau(np.array(sequence), np.arange(self.vertex_num))[0]+1)/2
        #'''
        pairs = itertools.combinations(range(0, len(sequence)-start_frame), 2)
        distance_r = 0
        distance_l = 0
        for x, y in pairs:
            if sequence[x] - sequence[y] > 0:
                distance_r += 1.0
            if sequence[x] - sequence[y] < 0:
                distance_l += 1.0
        tem_num = (len(sequence)-start_frame) * (len(sequence)-start_frame-1) / 2
        #err += [min(distance_l, distance_r)/tem_num]
        err = distance_r/tem_num
        #'''
        return [sequence], err
        
        