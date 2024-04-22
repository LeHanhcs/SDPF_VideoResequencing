
from __future__ import absolute_import, division, print_function
from copy import deepcopy
from skimage.io import imread

from tfoptflow.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from tfoptflow.visualize import display_img_pairs_w_flows
import cv2
import numpy as np
import constants as cs
import subprocess
import os
import pickle
from skimage.measure import compare_ssim, compare_psnr
from scipy.interpolate import Rbf
from heapq import nsmallest as ns
from heapq import nlargest as nl
import gc

class Identity_model():
    def __init__(self):
        pass
    def transform(self, x):
        return np.reshape(x, [len(x), -1])


class pwc_net():
    def __init__(self, s, paths = 'tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'):
        self.path = paths
        self.__build_net(s)
        #self.nn.print_config()
    
    def __build_net(self, s):
        gpu_devices = ['/device:GPU:0']  
        controller = '/device:GPU:0'

        self.nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        self.nn_opts['verbose'] = True
        self.nn_opts['ckpt_path'] = self.path
        self.nn_opts['batch_size'] = 1
        self.nn_opts['gpu_devices'] = gpu_devices
        self.nn_opts['controller'] = controller

        self.nn_opts['use_dense_cx'] = True
        self.nn_opts['use_res_cx'] = True
        self.nn_opts['pyr_lvls'] = 6
        self.nn_opts['flow_pred_lvl'] = 2
        
        self.nn_opts['adapt_info'] = (1, s[0], s[1], 2)
        
        self.nn = ModelPWCNet(mode='test', options=self.nn_opts)
    
    def pwc_predict(self, x, y, is_display = False):
        if x.shape != y.shape:
            print('x.shape = ', x.shape, ' ,y.shape = ', y.shape, ' are not equal!')
            return
        img_pairs = [(x, y)]
        pred_labels = self.nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
        pred_labels[0][np.all(np.abs(x.astype("int") - y.astype("int"))<=4, axis = -1)] = 0
        '''import matplotlib.pyplot as plt
        xx = x.copy()
        xx[np.all(np.abs((x/255) - (y/255))<=(5/255), axis = -1),:] = 0
        yy = y.copy()
        yy[np.all(np.abs(x.astype("int") - y.astype("int"))<=5, axis = -1), :] = 0
        plt.figure()
        plt.imshow(xx)
        plt.figure()
        plt.imshow(yy)
        plt.show()'''
        if is_display:
            display_img_pairs_w_flows(img_pairs, pred_labels)
        return pred_labels[0]
        
    def pwc_predict_test(self, x, y, is_display = False):
        if x.shape != y.shape:
            print('x.shape = ', x.shape, ' ,y.shape = ', y.shape, ' are not equal!')
            return
        img_pairs = [(x, y)]
        pred_labels = self.nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
        if is_display:
            display_img_pairs_w_flows(img_pairs, pred_labels)
        return pred_labels[0]

def direction_diff(x, y):
    if np.abs(x-y) > np.pi:
        return 2*np.pi-np.abs(x-y)
    else:
        return np.abs(x-y)

def flow_img(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img/255


def optical_direction(x, dir, eps = 128, model = None):
    mag_1, ang_1 = cv2.cartToPolar(x[...,0], x[...,1])
    
    #mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_1 = (mag_1 - mag_1.min())/(mag_1.max() - mag_1.min())
    eps = 0.5
    #while (np.sum(mag_1 >= eps) < 224):
    #    if eps < (1.0/16): 
    #        eps = 0
    #        break
    #    eps /= 2
    
    #'''
    if dir != -10:
        '''
        tem = np.zeros_like(x)
        
        tem[..., 0] = np.cos(dir)
        tem[..., 1] = np.sin(dir)
        tem *= mag_1[..., None]
        first = optical_dis(tem, x, model = model)
        
        tor = np.pi/3
        if dir + tor > 2*np.pi:
            tem[..., 0] = np.cos(2*np.pi - dir - tor)
            tem[..., 1] = np.sin(2*np.pi - dir - tor)
        else:
            tem[..., 0] = np.cos(dir + tor)
            tem[..., 1] = np.sin(dir + tor)
        tem *= mag_1[..., None]
        second = optical_dis(tem, x, model = model)
        
        if dir - tor < 0:
            tem[..., 0] = np.cos(2*np.pi + dir - tor)
            tem[..., 1] = np.sin(2*np.pi + dir - tor)
        else:
            tem[..., 0] = np.cos(dir - tor)
            tem[..., 1] = np.sin(dir - tor)
        tem *= mag_1[..., None]
        third = optical_dis(tem, x, model = model)
        
        print(first, " , ", second, " , ", third)
        if third[0] >= first[0] and second[0] >= first[0]:
            return True
        else:
            return False
        '''
        tem = np.abs(ang_1[mag_1 >= eps ]-dir)
        tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
        return np.mean(tem)
        #return direction_diff(optical_direction(x, -10, model = model), dir)
    else:
        #mm = np.copy(mag_1)
        xx = np.mean(np.cos(ang_1[mag_1 >= eps]))
        yy = np.mean(np.sin(ang_1[mag_1 >= eps]))
        #xx = np.mean(x[mag_1 >= eps, 0])
        #yy = np.mean(x[mag_1 >= eps, 1])
        m = np.linalg.norm([xx,yy])
        tem_a = np.angle(xx+(yy*1j))
        if tem_a < 0:
            tem_a=tem_a+2*np.pi
        #print(i, xx, " ", yy, " ", m, " ", np.mean(mm[mag_1 >= eps]), " ", np.median(mm[mag_1 >= eps]))
        #print(ang_1[mag_1 >= eps])

        #print(optical_direction(x, tem_a, i))
        #0/0
        return tem_a
    #'''
    
    '''
    xx = np.mean(np.cos(ang_1[mag_1 >= eps]))
    yy = np.mean(np.sin(ang_1[mag_1 >= eps]))
    m = np.linalg.norm([xx,yy])
    if m < 1e-3:
        angle = -10
    else:
        angle = np.angle(xx+(yy*1j))
    if dir == -10:
        return angle
    else:
        return direction_diff(angle, dir)
    '''
    
def optical_dis(x, y, eps = 0.5, model = None, mag_eps = -1, cal_opt_eps = True):
    if x is None or y is None:
        return np.inf
    xx = np.copy(x)
    yy = np.copy(y)

    #'''
    mag_1, ang_1 = cv2.cartToPolar(xx[...,0], xx[...,1])
    mag_2, ang_2 = cv2.cartToPolar(yy[...,0], yy[...,1])
    #mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    #mag_2 = cv2.normalize(mag_2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_1 = (mag_1 - mag_1.min())/(mag_1.max() - mag_1.min())
    mag_2 = (mag_2 - mag_2.min())/(mag_2.max() - mag_2.min())
    
    
    
    tem_mag_mul = (mag_1 < eps ) * (mag_2 < eps )
    tem_mag_add = (mag_1 < eps ) + (mag_2 < eps )
    
    if mag_eps == -1:
        while (np.sum(~tem_mag_add) < 224):
            if eps < (1.0/16):
            #if eps < 16:
                #eps = 0
                print("tem_mag_add = ", np.sum(~tem_mag_add))    
                return 0, 0
            eps /= 2
            tem_mag_mul = (mag_1 < eps ) * (mag_2 < eps )
            tem_mag_add = (mag_1 < eps ) + (mag_2 < eps )
            print(eps, " ", np.sum(~tem_mag_add))
        mag_eps = np.sum(~tem_mag_add)
        mag_output = True
    else:
        mag_output = False
        #if mag_eps > 224:
        #    mag_eps = 224
    #mag_eps = 224
    
    print(eps, " ", np.sum(~tem_mag_add))
    '''while (np.sum(~tem_mag_add) < mag_eps):
        if eps < (1.0/16):
        #if eps < 16:
            #eps = 0
            print("tem_mag_add = ", np.sum(~tem_mag_add))    
            return 0, 0
        eps /= 2
        tem_mag_mul = (mag_1 < eps ) * (mag_2 < eps )
        tem_mag_add = (mag_1 < eps ) + (mag_2 < eps )
        print(eps, " ", np.sum(~tem_mag_add))
    '''
    tem_mag_all = np.where(mag_1 > mag_2, mag_1, mag_2)
    tem_mag_eps = nl(mag_eps, tem_mag_all.flatten())[-1]
    tem_mag_add = tem_mag_all < tem_mag_eps
    #if cal_opt_eps:
    #    tem_mag_all = np.where(mag_1 < mag_2, mag_1, mag_2)
    #    tem_mag_eps = nl(mag_eps, tem_mag_all.flatten())[-1]
    #    tem_mag_add = tem_mag_all < tem_mag_eps
    #else:
    #    tem_mag_all = mag_1#np.where(mag_1 < mag_2, mag_1, mag_2)
    #    tem_mag_add = tem_mag_all < eps
    if cal_opt_eps:
        print("tem_mag_add = ", np.sum(~tem_mag_add))    
        print("eps = ", eps)
    else:
        print("opt__tem_mag_add = ", np.sum(~tem_mag_add))    
        print("opt__eps = ", eps)
    
    xx[tem_mag_add] = 0
    yy[tem_mag_add] = 0
    
    x_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    y_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    
    x_img[...,0] = (np.cos(ang_1)/2)+0.5#*(mag_1/2+0.5) + 0.5
    x_img[...,1] = (np.sin(ang_1)/2)+0.5#*(mag_1/2+0.5) + 0.5
    x_img[...,:2][tem_mag_add] = 0.5
    #x_img[...,:2] = x_img[...,:2]*(mag_1/2+0.5)[...,None]
    x_img[...,2][tem_mag_add] = 0
    y_img[...,0] = (np.cos(ang_2)/2)+0.5#*(mag_2/2+0.5) + 0.5
    y_img[...,1] = (np.sin(ang_2)/2)+0.5#*(mag_2/2+0.5) + 0.5
    y_img[...,:2][tem_mag_add] = 0.5
    #y_img[...,:2] = y_img[...,:2]*(mag_2/2+0.5)[...,None]
    y_img[...,2][tem_mag_add] = 0
        
    result = model.reduce_dim(np.concatenate([x_img[None], y_img[None]]))
    result_dis = -np.linalg.norm(result[0]-result[1])
    #if cal_opt_eps:
    #    opt_eps = optical_eps(xx, eps = eps, model = model)[0]
    #    print("opt_eps in dis : ", opt_eps, " compare with ", result_dis)
    #    if result_dis > opt_eps:
    #        result_dis = 0
    
    #result_1 = model.reduce_dim(np.concatenate([flow_img(yy)[None], flow_img(x)[None]]))
    #return min(np.linalg.norm(result[0]-result[1]), np.linalg.norm(result_1[0]-result_1[1]))
    if mag_output:
        return result_dis, np.sum(~tem_mag_add)
    else:
        return result_dis, eps
    #'''
    
    #############################################
    '''
    mag_eps = 224
    
    mag_1, ang_1 = cv2.cartToPolar(xx[...,0], xx[...,1])
    mag_2, ang_2 = cv2.cartToPolar(y[...,0], y[...,1])
    mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_2 = cv2.normalize(mag_2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    tem_mag_mul = (mag_1 >= eps) * (mag_2 >= eps)
    tem_mag_add = (mag_1 >= eps) + (mag_2 >= eps)
    while (np.sum(tem_mag_mul) <= mag_eps):
        if eps < 32:
            print("tem_mag_mul = ", np.sum(tem_mag_mul))    
            return 0
        eps /= 2
        tem_mag_mul = (mag_1 >= eps) * (mag_2 >= eps)
        tem_mag_add = (mag_1 >= eps) + (mag_2 >= eps)
        
    print("tem_mag_mul = ", np.sum(tem_mag_mul))
    #print(np.sum(tem_mag_mul))
    #print(np.sum(tem_mag_add))
    tem_only_1 = tem_mag_add^(mag_2 >= np.mean(mag_2[mag_2>0]))
    tem_only_2 = tem_mag_add^(mag_1 >= np.mean(mag_1[mag_1>0]))
    

    #ang_1[tem_mag_other] = ang_2[tem_mag_other] + 120*np.pi/180
    add_or_mul = True
    add_or_mul = np.sum(tem_mag_mul) <= mag_eps
    if add_or_mul:
        print("zero")
        return 0
        #return direction_diff(optical_direction(xx, -10), optical_direction(y, -10))
        ####################################
        tem_axis = np.nonzero(mag_1 >= eps)
        tem_axis_1 = np.argwhere(tem_only_2)
        n = min(len(tem_axis), 224)
        if n != 0:
            for ax in tem_axis_1:
                tem_pixel = [q[0] for q in ns(n, enumerate(np.abs(tem_axis[0] - ax[0]) + np.abs(tem_axis[1] - ax[1])), key = lambda x:x[1])]
                tem_pixel = tuple([tem_axis[0][tem_pixel], tem_axis[1][tem_pixel]])
                rx = Rbf(tem_pixel[0], tem_pixel[1], xx[...,0][tem_pixel] , function = 'gaussian')
                xx[...,0][tem_only_2] = rx(tem_axis_1[:,0], tem_axis_1[:,1])
                ry = Rbf(tem_pixel[0], tem_pixel[1], xx[...,1][tem_pixel] , function = 'gaussian')
                xx[...,1][tem_only_2] = ry(tem_axis_1[:,0], tem_axis_1[:,1])
                del rx, ry
        
        tem_axis = np.nonzero(mag_2 >= eps)
        tem_axis_2 = np.argwhere(tem_only_1)
        n = min(len(tem_axis), 224)
        if n != 0:
            for ax in tem_axis_2:
                tem_pixel = [q[0] for q in ns(n, enumerate(np.abs(tem_axis[0] - ax[0]) + np.abs(tem_axis[1] - ax[1])), key = lambda x:x[1])]
                tem_pixel = tuple([tem_axis[0][tem_pixel], tem_axis[1][tem_pixel]])
                rx = Rbf(tem_pixel[0], tem_pixel[1], y[...,0][tem_pixel] , function = 'gaussian')
                y[...,0][tem_only_1] = rx(tem_axis_2[:,0], tem_axis_2[:,1])
                ry = Rbf(tem_pixel[0], tem_pixel[1], y[...,1][tem_pixel] , function = 'gaussian')
                y[...,1][tem_only_1] = ry(tem_axis_2[:,0], tem_axis_2[:,1])
                del rx, ry
        #########################3
        #ang_1[tem_only_2] = optical_direction(x,-10)#np.mean(ang_1[mag_1 >= eps])
        #ang_2[tem_only_1] = optical_direction(y,-10)#np.mean(ang_2[mag_2 >= eps])
        #tem_mag_other = tem_mag_add^tem_mag_mul
        tem = ang_1[tem_mag_add] - ang_2[tem_mag_add]
    else:
        tem = ang_1[tem_mag_mul] - ang_2[tem_mag_mul]
    if tem.size == 0:
        return 0
    tem = np.abs(tem)
    tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
    #tem = tem * tem_ang_weight[tem_mag_add]
    #result = np.mean(np.square(tem))
    result = np.mean(tem)
    return -result
    '''

def optical_image_vector(x, y, eps = 0.5, model = None, eps_output = True):
    
    xx = np.copy(x)
    yy = np.copy(y)
    
    mag_eps = 224#int(224*224/100)

    mag_1, ang_1 = cv2.cartToPolar(xx[...,0], xx[...,1])
    mag_2, ang_2 = cv2.cartToPolar(yy[...,0], yy[...,1])
    
    mag_1 = (mag_1 - mag_1.min())/(mag_1.max() - mag_1.min())
    mag_2 = (mag_2 - mag_2.min())/(mag_2.max() - mag_2.min())
    
    tem_mag = (mag_1 >= eps ) * (mag_2 >= eps )
    
    while (np.sum(tem_mag) <= mag_eps):
        if eps < (1.0/16):
            print("tem_mag_add = ", np.sum(tem_mag))
            if eps_output:
                return 0, -1
            else:
                return 0
        eps /= 2
        tem_mag = (mag_1 >= eps ) * (mag_2 >= eps )
        print(eps, " ", np.sum(tem_mag))
    
    if eps_output:
        print("tem_mag_add = ", np.sum(tem_mag))    
        print("eps = ", eps)
    else:
        print("opt__tem_mag_add = ", np.sum(tem_mag))    
        print("opt__eps = ", eps)
    tem_other_mag = ~tem_mag
    xx[tem_other_mag] = 0
    yy[tem_other_mag] = 0
    
    x_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])

    x_img[...,0] = (np.cos(ang_1)/2)+0.5
    x_img[...,1] = (np.sin(ang_1)/2)+0.5
    x_img[...,:2][tem_other_mag] = 0.5
    x_img[...,2][tem_other_mag] = 0
    
    y_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    
    y_img[...,0] = (np.cos(ang_2)/2)+0.5
    y_img[...,1] = (np.sin(ang_2)/2)+0.5
    y_img[...,:2][tem_other_mag] = 0.5
    y_img[...,2][tem_other_mag] = 0
        
    result = model.reduce_dim(np.concatenate([x_img[None], y_img[None]]))

    if eps_output:
        return result[0], result[1], eps
    else:
        return result[0], result[1]
    #'''


def optical_check(x, y, eps = 0.5, model = None):
    eps = 0.5
    angle_eps = np.pi/3 #################
    xx = np.copy(x)
    xx[...,0] = np.cos(angle_eps)*x[...,0] - np.sin(angle_eps)*x[...,1] 
    xx[...,1] = np.sin(angle_eps)*x[...,0] + np.cos(angle_eps)*x[...,1]
    
    xxx = np.copy(x)
    xxx[...,0] = np.cos(-angle_eps)*x[...,0] - np.sin(-angle_eps)*x[...,1] 
    xxx[...,1] = np.sin(-angle_eps)*x[...,0] + np.cos(-angle_eps)*x[...,1]
    
    a1, mag_eps = optical_dis(x, xx, eps = eps, model = model, cal_opt_eps = False)
    a2, _ = optical_dis(x, xxx, eps = eps, model = model, cal_opt_eps = False)
    a3, _ = optical_dis(x, y, eps = eps, model = model, mag_eps = mag_eps) #cal_opt_eps = False)
    print("a1 : ", a1, ", a2 : ", a2, ", a3 : ", a3)
    
    if min(a1, a2) > a3:
        return True
    elif max(a1, a2) < a3:
        return False
    else:
        a4, _ = optical_dis(xx, y, eps = eps, model = model, mag_eps = mag_eps) #cal_opt_eps = False)
        a5, _ = optical_dis(xxx, y, eps = eps, model = model, mag_eps = mag_eps) #cal_opt_eps = False)
        print("a4 : ", a4, ", a5 : ", a5)
        if a1 > a3:
            return (a4 > a5)
        elif a2 > a3:
            return (a5 > a4)
        else:
            return False
    



def optical_eps(x, eps = 0.5, model = None):
    angle_eps = np.pi/2 #################
    xx = np.copy(x)
    xx[...,0] = np.cos(angle_eps)*x[...,0] - np.sin(angle_eps)*x[...,1] 
    xx[...,1] = np.sin(angle_eps)*x[...,0] + np.cos(angle_eps)*x[...,1]
    
    xxx = np.copy(x)
    xxx[...,0] = np.cos(-angle_eps)*x[...,0] - np.sin(-angle_eps)*x[...,1] 
    xxx[...,1] = np.sin(-angle_eps)*x[...,0] + np.cos(-angle_eps)*x[...,1]
    
    '''
    mag, ang = cv2.cartToPolar(x[...,0], x[...,1])
    mag = (mag-mag.min())/(mag.max()-mag.min())
    
    eps = 0.5
    
    while (np.sum(mag >= eps) <= 224):
        if eps < (1.0/16):
        #if eps < 16:
            print("tem_mag_add = ", np.sum(mag >= eps) )    
            return 0, 224
        eps /= 2
    print("tem_mag_add = ", np.sum(mag >= eps) )
    tem_mag_add = mag < eps
    
    
    _ , ang_1 = cv2.cartToPolar(xx[...,0], xx[...,1])
    _ , ang_2 = cv2.cartToPolar(xxx[...,0], xxx[...,1])
    #mag_max = np.max([np.max(mag_1), np.max(mag_2)])
    
    x_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    y_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    z_img = np.ones([cs.img_height, cs.img_width, cs.img_channel])
    
    x_img[...,0] = np.cos(ang)/2 + 0.5
    x_img[...,1] = np.sin(ang)/2 + 0.5
    x_img[...,:2][tem_mag_add] = 0.5
    x_img[...,2][tem_mag_add] = 0
    y_img[...,0] = np.cos(ang_1)/2 + 0.5
    y_img[...,1] = np.sin(ang_1)/2 + 0.5
    y_img[...,:2][tem_mag_add] = 0.5
    y_img[...,2][tem_mag_add] = 0
    z_img[...,0] = np.cos(ang_2)/2 + 0.5
    z_img[...,1] = np.sin(ang_2)/2 + 0.5
    z_img[...,:2][tem_mag_add] = 0.5
    z_img[...,2][tem_mag_add] = 0
    
    result = model.reduce_dim(np.concatenate([x_img[None], y_img[None], z_img[None]]))
    
    a1 = -np.linalg.norm(result[0]-result[1])
    a2 = -np.linalg.norm(result[0]-result[2])
    '''
    
    
    a1, m1 = optical_dis(x, xx, eps = eps, model = model, cal_opt_eps = False)
    a2, _ = optical_dis(x, xxx, eps = eps, model = model, cal_opt_eps = False)
    
    '''
    test_w = np.full([224,224,3], 1.0)
    test1 = np.full([224,224,3], 1.0)
    test2 = np.full([224,224,3], 0.0)
    test_b = np.full([224,224,3], 0.0)
    test1[:10,:,2] = 0 #496
    test2[:10,:,2] = 1 #265
    result = model.reduce_dim(np.concatenate([test_w[None], test1[None], test2[None], test_b[None]]))
    print(np.linalg.norm(result[0]-result[1]), " ", np.linalg.norm(result[3]-result[1]))
    print(np.linalg.norm(result[0]-result[2]), " ", np.linalg.norm(result[3]-result[2]))
    '''
    
    result = min(a1, a2)
    print("opt_eps = ", result)

    return result, m1

def optical_predict(x, y, batch_size = -1):
    
    if batch_size == -1:
        result = np.zeros([x.shape[0], x.shape[1], x.shape[2], 2])
        np.save("1.npy", x)
        np.save("2.npy", y)
        subprocess.call(["python", "predict.py"])
        
        result[:] = np.load("result.npy")
    else:
        n = int(len(x)/batch_size)
        batch_x = [x[i*batch_size:(i+1)*batch_size] for i in range(n)] + [x[n*batch_size:]]
        batch_y = [y[i*batch_size:(i+1)*batch_size] for i in range(n)] + [y[n*batch_size:]]
        result = np.zeros([x.shape[0], x.shape[1], x.shape[2], 2])
        for i in range(len(batch_x)):
            np.save("1.npy", batch_x[i])
            np.save("2.npy", batch_y[i])
            subprocess.call(["python", "predict.py"])
            result[i*batch_size:(i+1)*batch_size] = np.load("result.npy")
    
    os.remove("1.npy")
    os.remove("2.npy")
    os.remove("result.npy")
    return result    
    

def reduce_latent(x, model = "Unet"):
    import pickle
    pickle.dump(x, open("reduce_data.txt", "wb"))
    if model == "Unet":
        import Unet
        subprocess.call(["python", "Unet.py"])
    result = pickle.load(open("reduce_result.txt", "rb"))
    os.remove("reduce_result.txt")
    return result

def clean_image(path, imgs, dis, is_reproduce = False):
    result = np.zeros_like(dis).astype("bool")
    num = len(dis)
    is_already_clean = os.path.isfile(path+"clean_id.npy") and (not is_reproduce)
    if is_already_clean:
        result = np.load(path+"clean_id.npy")
    else:
        for i in range(num):
            for j in range(i+1, num):
                if result[i,j]:
                    continue
                if compare_psnr(imgs[i], imgs[j]) >= 30:
                    result[:,j] = True
                    result[j,:] = True
        np.save(path+"clean_id.npy", result) 
    dis[result] = 0
    print("clean done")
    return dis

def image_optflow(path, imgs, dis, is_reproduce = False):
    is_already_optflow = not is_reproduce
    num = len(dis)
    for i in range(num):
        if is_already_optflow:
            is_already_optflow = os.path.isfile(path+"opt_"+str(i)+".txt")
        else:
            break

    if not is_already_optflow:
        for i in range(num):
            tem_result = [None] * num
            id = np.nonzero(dis[i])[0]
            if len(id) != 0:
                opt_result = optical_predict(np.repeat((imgs[i])[np.newaxis], len(id), axis = 0), imgs[id])
                for t in range(len(id)):
                    tem_result[id[t]] = opt_result[t]
            with open(path+"opt_"+str(i)+".txt", "wb") as f:
                pickle.dump(tem_result, f)
    print("optical flow done")

def find_path(p, i, j):
    path = [j]
    k = j
    while p[i, k] != -9999:
        path.append(p[i, k])
        k = p[i, k]
    return path[::-1]

def dfs(t, v1, v2):
    sequence = []
    if t.ndim != 2:
        print("expect tree's ndim is 2, but get ", t.ndim, "\n")
        return sequence
    non = t[0][0]
    n = len(t)
    vis = [False] * n
    stack = []
    stack += [v1]
    sequence += [v1]
    vis[v1] = True
    while stack:
        s = stack.pop(-1)
        is_end = True
        for i in range(n):
            tem = t[s][i]
            if tem != non and not vis[i]:
                sequence += [i]
                if i == v2:
                    return sequence
                vis[i] = True
                stack += [s]
                stack += [i]
                is_end = False
                break
        if is_end:
            sequence.pop(-1)
    print(v2, " is not in the tree!!\n")
    return []
    
def gpu_limit():
    import tensorflow as tf
    import keras.backend as K

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    sess = tf.compat.v1.Session(config=config)

    K.set_session(sess)

def reproduce_all_opt(path):
    from data_generator import Data_Generator
    from path_explorer import Path_Explorer
    d = Data_Generator()
    test_data, test_origin, paths = d.load_video_to_data(is_training = False, is_np = False, video_path = path)
    p = Path_Explorer()
    for i in range(len(test_data)):
        tem_p = "data/testing/image/" + paths[i] + "/"
        p.build_distance_graph(test_data[i])
        p.dis = clean_image(tem_p, test_origin[i], p.dis, is_reproduce = True)
        image_optflow(tem_p, test_origin[i], p.dis, is_reproduce = True)
    
import torch
import torch.nn as nn
from torch.autograd import Variable

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask



if __name__ == '__main__':
    #a = np.array([[0,1,0,0,0,0,0,0], [1,0,1,1,0,0,0,0], [0,1,0,0,1,1,0,0], [0,1,0,0,0,0,1,1], [0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0]])
    #print(dfs(a, 4,7))

    #reproduce_all_opt("video_testing")
    #'''
    def bins(x, s, n, an):
        switch = s
        numb = n#max(int(np.sum(x!=-10)/30), 3)
        xx = np.copy(x)
        xxx = np.zeros_like(x)
        i = 0
        while(i < len(x)):
            k = 0
            j = i
            while(j+1 < len(x)):
                j += 1
                if x[j] == -10:
                    continue
                else:
                    if x[i] != -10 and direction_diff(x[j],x[i]) <= an*np.pi/180:
                        k += 1
                    else:
                        break
            if k < numb:
                if switch:
                    if xxx[i] == 0:
                        xx[i] = -10
                else:
                    xx[i] = -10
                i += 1
            else:
                if switch:
                    for k in range(i,j):
                        xxx[k] = 1
                    i += 1
                else:
                    i = j
        np.set_printoptions(suppress = True)
        #print(xx)
        #print(xx[0:57])
        #print(xx[86:171])
        #print(switch, " ", numb, " ", an)
        #print(np.sum(xx==-10), " ", np.sum(xx==-10)/len(x))
        return [switch, numb, an, np.sum(xx==-10), np.sum(xx==-10)/len(x)]
    
    a = np.load("data/testing/image/test_chinese_ink.mp4/test_opt_dir.npy")
    a[0] = np.pi
    a[87] = 0
    b = np.load("data/testing/image/test_buffalo_1.mp4/test_opt_dir.npy")
    cc = np.load("testttt.npy")
    #c = np.load("data/testing/image/test_hippo_funk.mp4/test_opt_dir.npy")
    print(len(a), " ", len(b), " ", len(cc))
    #d = np.load("data/testing/image/test_fight_1.mp4/test_opt_dir.npy")
    #e = np.load("data/testing/image/test_fight_3.mp4/test_opt_dir.npy")
    #f = np.load("data/testing/image/test_minions.mp4/test_opt_dir.npy")
    #g = np.load("data/testing/image/test_river.mp4/test_opt_dir.npy")
    #h = np.load("data/testing/image/test_falling.mp4/test_opt_dir.npy")
    '''
    s = [True]#, False]
    n = [3, 4, 5, 6]
    an = [30, 45, 60, 90]
    tttt = []
    for ss in s:
        for nn in range(3, 11):
            for aann in an:#range(30, 91):
                aa = bins(a, ss, nn, aann)
                #print()
                bb = bins(b, ss, nn, aann)
                #print()
                #if aa[4] < 0.3:
                tttt += [[aa, bb]]
    from heapq import nlargest as nl
    from heapq import nsmallest as ns
    print(tttt)
    print()
    print(ns(3, tttt, key=lambda x:np.abs(x[0][4]+(1-x[1][4]))))
    print()
    print(nl(3, tttt, key=lambda x:x[1][4]))
    print()
    print(ns(3, tttt, key=lambda x:x[0][4]))
    '''
    #print(bins(cc, True, 5, 45))
    #print(bins(b, True, 5, 45))
    #print(bins(c, True, 5, 45))
    #print("\n")
    #bins(c)
    #print("\n")
    #bins(d)
    #print("\n")
    #bins(e)
    #print("\n")
    #bins(f)
    #print("\n")
    #bins(g)
    #print("\n")
    #bins(h)
    #'''
    
    #0/0
    i_1 = 8
    i_2 = 9
    i_3 = 10
    
    #x = cv2.imread('tfoptflow/samples/test_artist-directed_walk_1.mp4/' + str(i_1) + '.jpg')
    #y = cv2.imread('tfoptflow/samples/test_artist-directed_walk_1.mp4/0_' + str(i_2) + '.jpg')
    #z = cv2.imread('tfoptflow/samples/test_artist-directed_walk_1.mp4/0_' + str(i_3) + '.jpg')
    #x = cv2.imread('data/testing/image/test_chinese_ink.mp4/' + str(i_1) + '.jpg')
    #y = cv2.imread('data/testing/image/test_chinese_ink.mp4/' + str(i_2) + '.jpg')
    #z = cv2.imread('data/testing/image/test_chinese_ink.mp4/' + str(i_3) + '.jpg')
    x = cv2.imread('tfoptflow/samples/test_chinese_ink.mp4/' + str(i_1) + '.jpg')
    y = cv2.imread('tfoptflow/samples/test_chinese_ink.mp4/' + str(i_2) + '.jpg')
    z = cv2.imread('tfoptflow/samples/test_chinese_ink.mp4/' + str(i_3) + '.jpg')
    #x = cv2.imread('tfoptflow/samples/test_falling.mp4/' + str(i_1) + '.jpg')
    #y = cv2.imread('tfoptflow/samples/test_falling.mp4/' + str(i_2) + '.jpg')
    #z = cv2.imread('tfoptflow/samples/test_falling.mp4/' + str(i_3) + '.jpg')
    #x = cv2.imread('tfoptflow/samples/' + str(i_1) + '.jpg')
    #y = cv2.imread('tfoptflow/samples/' + str(i_2) + '.jpg')
    #z = cv2.imread('tfoptflow/samples/' + str(i_3) + '.jpg')

    x = cv2.resize(x, (224, 224), interpolation = cv2.INTER_LINEAR)
    y = cv2.resize(y, (224, 224), interpolation = cv2.INTER_LINEAR)
    z = cv2.resize(z, (224, 224), interpolation = cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
    
    
    '''
    import matplotlib.pyplot as plt
    from Unet import Res_Unet
    m = Res_Unet()
    m.load_mod()
    plt.figure()
    plt.imshow(x/255)
    plt.figure()
    plt.imshow(m.model.predict((x/255)[np.newaxis])[0])
    plt.figure()
    plt.imshow(y/255)
    plt.figure()
    plt.imshow(m.model.predict((y/255)[np.newaxis])[0])
    plt.show()
    0/0
    '''
    
    
    from skimage.measure import compare_mse, compare_ssim, compare_psnr
    print(compare_mse(x,y), " ", compare_mse(z,y))
    print(compare_ssim(x,y, multichannel=True), " ", compare_ssim(z,y, multichannel=True))
    print(compare_psnr(x,y), " ", compare_psnr(z,y))
    
    is_dis = True
    #x = x[:,:]/255
    #y = y[:,:]/255
    #z = z[:,:]/255
    
    if is_dis:
        #is_dis = False
        p = pwc_net(x.shape)
        a = p.pwc_predict(x, y, is_display = is_dis)
        b = p.pwc_predict(y, x, is_display = is_dis)
        c = p.pwc_predict(y, z, is_display = is_dis)
        d = p.pwc_predict(z, y, is_display = is_dis)
    else:
        xx = np.concatenate([x[None], y[None], y[None], z[None]])
        yy = np.concatenate([y[None], x[None], z[None], y[None]])
        result = optical_predict(xx,yy)
        a = result[0]
        b = result[1]
        c = result[2]
        d = result[3]
    
    
    import matplotlib.pyplot as plt
    mm1, _ = cv2.cartToPolar(a[...,0], a[...,1])
    mm2, aa2 = cv2.cartToPolar(b[...,0], b[...,1])
    mm3, _ = cv2.cartToPolar(c[...,0], c[...,1])
    mm4, _ = cv2.cartToPolar(d[...,0], d[...,1])
    mm1 = (mm1-mm1.min())/(mm1.max()-mm1.min())
    mm2 = (mm2-mm2.min())/(mm2.max()-mm2.min())
    mm3 = (mm3-mm3.min())/(mm3.max()-mm3.min())
    mm4 = (mm4-mm4.min())/(mm4.max()-mm4.min())
    plt.figure()
    plt.imshow(np.concatenate([(np.cos(aa2)/2+0.5)[...,None], (np.sin(aa2)/2+0.5)[...,None], (np.ones_like(aa2))[...,None]], axis = -1))
    #plt.imshow(np.concatenate([mm1[...,None], mm1[...,None], mm1[...,None]], axis = -1))
    plt.figure()
    plt.imshow(np.concatenate([mm2[...,None], mm2[...,None], mm2[...,None]], axis = -1))
    plt.figure()
    plt.imshow(np.concatenate([mm3[...,None], mm3[...,None], mm3[...,None]], axis = -1))
    plt.figure()
    plt.imshow(np.concatenate([mm4[...,None], mm4[...,None], mm4[...,None]], axis = -1))
    plt.show()
    0/0
    
    def test_dis(x, y, bb, eps = 128):
        xx = x
        
        from base_model import Tri_Model
        from simple_AE import Fuse_AE
        m = Fuse_AE()
        m.gen_model(is_compile = False)
        m = Tri_Model(m)
        m.load_mod()
        eps = 0.5
        print("He ", np.sum(bb-x))
        print(optical_eps(xx, model = m))
        print("He ", np.sum(bb-x))
        print(optical_dis(xx, y, model = m))
        print("He ", np.sum(bb-x))
        
        mag_1, ang_1 = cv2.cartToPolar(xx[...,0], xx[...,1])
        mag_2, ang_2 = cv2.cartToPolar(y[...,0], y[...,1])
        print("He ", np.sum(bb-x))
        
        mag_1 = (mag_1-mag_1.min())/(mag_1.max()-mag_1.min())
        mag_2 = (mag_2-mag_2.min())/(mag_2.max()-mag_2.min())
        
        tem_mag_mul = (mag_1 < eps ) * (mag_2 < eps )
        tem_mag_add = (mag_1 < eps ) + (mag_2 < eps )
        while (np.sum(~tem_mag_add) <= 224):
            eps /= 2
            tem_mag_mul = (mag_1 < eps ) * (mag_2 < eps )
            tem_mag_add = (mag_1 < eps ) + (mag_2 < eps )
        
        print("eps = ", eps)
        xx[tem_mag_add] = 0
        y[tem_mag_add] = 0
        print("He ", np.sum(bb-x))        
        #if np.sum(~tem_mag_add) == 0:
        #    print("\n\n", np.sum(~tem_mag_add))
        #    eps /= 2
        #    xx[(mag_1 < eps ) + (mag_2 < eps )] = 0
        #    y[(mag_1 < eps ) + (mag_2 < eps )] = 0
        #else:
        #    xx[tem_mag_add] = 0
        #    y[tem_mag_add] = 0
        
        #xx[mag_1 < eps] = 0
        #y[mag_2 < eps] = 0
        
        result = m.reduce_dim(np.concatenate([flow_img(xx)[None], flow_img(y)[None]]))
        #result = m.reduce_dim(np.concatenate([xx[None], y[None]]))
        return -np.linalg.norm(result[0]-result[1])
    #print()
    #print(optical_direction(c, 2.94580017, 64))
    #print()
    bb = np.copy(b)
    print("Hi ", np.sum(bb-b))
    print("test_dis = ", test_dis(b, c, bb, eps = 128))
    print("Hi ", np.sum(bb-b))
    f = np.copy(b)
    f[...,0] = np.cos(np.pi/2)*b[...,0]-np.sin(np.pi/2)*b[...,1]
    f[...,1] = np.sin(np.pi/2)*b[...,0]+np.cos(np.pi/2)*b[...,1]
    print("Hi ", np.sum(bb-b))
    print("test_dis = ", test_dis(b, f, eps = 128))
    q = np.copy(b)
    q[...,0] = np.cos(-np.pi/2)*b[...,0]-np.sin(-np.pi/2)*b[...,1]
    q[...,1] = np.sin(-np.pi/2)*b[...,0]+np.cos(-np.pi/2)*b[...,1]
    print("Hi ", np.sum(bb-b))
    print("test_dis = ", test_dis(b, q, eps = 128))
    print("Hi ", np.sum(bb-b))
    print("opt_dis : ", optical_dis(b,c, eps = 128))
    0/0
    b = -b
    
    mag_1, ang_1 = cv2.cartToPolar(b[...,0], b[...,1])
    mag_2, ang_2 = cv2.cartToPolar(c[...,0], c[...,1])
    
    #mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    #mag_2 = cv2.normalize(mag_2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    
    mag = (mag_1 < np.mean(mag_1)) + (mag_2 < np.mean(mag_2))
    b[mag] = 0
    c[mag] = 0
    print(len(np.nonzero(b)[0]))
    from matplotlib import pyplot as plt
    #plt.figure()
    #plt.subplot(121)
    #plt.imshow(flow_img(b))
    #plt.subplot(122)
    #plt.imshow(flow_img(c))
    #plt.show()
    
    def ttttt(a, b):
        n = len(a)
        q = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                ii = int(i+b[i,j,0])
                jj = int(j+b[i,j,1])
                if 0 <= ii < n and 0 <= jj < n and a[ii,jj,0] != 0 and a[ii,jj,1] != 0 and b[i,j,0] != 0 and b[i,j,1] != 0:
                    q[i,j] = np.sum(np.square(a[ii,jj]+b[i,j]))-0.5-0.01*( np.sum(np.square(a[ii,jj]))+np.sum(np.square(b[i,j])))
                else:
                    q[i,j] = np.inf
                
        
        #print(q[(q != -0.5) * (q != np.inf)])
        #print(np.sum((q != -0.5) * (q != np.inf) * (q < 0)))
        #print(np.sum((q != -0.5) * (q != np.inf) * (q > 0)))
        return (q != np.inf) * (q <= 0)
        
    
    
    
    #0/0
    '''
    q = np.zeros_like(a)
    mag_1, ang_1 = cv2.cartToPolar(a[...,0], a[...,1])
    mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    q[mag_1 >= 128, 0] = np.cos(np.pi)
    q[mag_1 >= 128, 1] = np.sin(np.pi)
    print("test_dis first = ", test_dis(-q, a))
    q[mag_1 >= 128] = q[mag_1 >= 128] * mag_1[mag_1 >= 128][:, None]
    print("test_dis first = ", test_dis(-q, a))
    q[mag_1 >= 128, 0] = np.cos(np.pi/2)
    q[mag_1 >= 128, 1] = np.sin(np.pi/2)
    print("test_dis second = ", test_dis(-q, a))
    q[mag_1 >= 128] = q[mag_1 >= 128] * mag_1[mag_1 >= 128][:, None]
    print("test_dis second = ", test_dis(-q, a))
    q[mag_1 >= 128, 0] = np.cos(3*np.pi/2)
    q[mag_1 >= 128, 1] = np.sin(3*np.pi/2)
    print("test_dis third = ", test_dis(-q, a))
    q[mag_1 >= 128] = q[mag_1 >= 128] * mag_1[mag_1 >= 128][:, None]
    print("test_dis third = ", test_dis(-q, a))
    '''
    
    print(b.min(), " ", b.max())
    print(c.min(), " ", c.max())
    mag_1, ang_1 = cv2.cartToPolar(a[...,0], a[...,1])
    mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    print(np.sum(mag_1>=128))
    print(np.sum((ang_1[mag_1>=128])<=(np.pi/2)))
    print(optical_direction(a,-10))
    print(optical_direction(-b,-10))
    print(optical_direction(c,-10))
    print("opt_dis : ", optical_dis(b,c))
    np.set_printoptions(threshold = np.inf)
    
    
    b=-b
    qq = ttttt(a, b)
    print("qq = ", np.sum(qq))
    qqq = ttttt(c, d)
    print("qqq = ", np.sum(qqq))
    qqqq = qq*qqq
    print("qqqq = ", np.sum(qqqq))
    b = -b
    bb = b+c
    #mag_1, ang_1 = cv2.cartToPolar(b[...,0], b[...,1])
    #mag_2, ang_2 = cv2.cartToPolar(c[...,0], c[...,1])
    #print(np.mean(np.abs(ang_1[qqqq]-ang_2[qqqq])))
    mag, ang = cv2.cartToPolar(bb[...,0], bb[...,1])
    print("all : ", np.mean(ang[qqqq]))
    print(optical_direction(bb,-10))
    b = -b
    xx = np.mean(np.cos(ang[qqqq]))
    yy = np.mean(np.sin(ang[qqqq]))
    #xx = np.sum(x[mag_1 >= eps, 0])
    #yy = np.sum(x[mag_1 >= eps, 1])
    m = np.linalg.norm([xx,yy])
    tem_a = np.angle(xx+(yy*1j))
    if tem_a < 0:
        tem_a=tem_a+2*np.pi
    print(tem_a)
    eps = 128
    mag_1, ang_1 = cv2.cartToPolar(b[...,0], b[...,1])
    mag_2, ang_2 = cv2.cartToPolar(c[...,0], c[...,1])
    
    tem = np.abs(ang_1[qqqq]-ang_2[qqqq])
    tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
    print("dis : ", np.mean(tem))
    
    mag_1 = cv2.normalize(mag_1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_2 = cv2.normalize(mag_2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    
    tem_mag_mul = ((mag_1 >= np.mean(mag_1[mag_1>0]))*qqqq) * ((mag_2 >= np.mean(mag_2[mag_2>0]))*qqqq)
    tem_mag_add = ((mag_1 >= np.mean(mag_1[mag_1>0]))*qqqq) + ((mag_2 >= np.mean(mag_2[mag_2>0]))*qqqq)
    #print(np.sum(tem_mag_mul))
    #print(np.sum(tem_mag_add))
    tem_only_1 = tem_mag_add^(mag_2 >= eps)
    tem_only_2 = tem_mag_add^(mag_1 >= eps)
    #ang_1[tem_mag_other] = ang_2[tem_mag_other] + 120*np.pi/180
    add_or_mul = False
    if add_or_mul:
        ang_1[tem_only_2] = optical_direction(b,-10)#np.mean(ang_1[mag_1 >= eps])
        ang_2[tem_only_1] = optical_direction(c,-10)#np.mean(ang_2[mag_2 >= eps])
        tem_mag_other = tem_mag_add^tem_mag_mul
        tem = ang_1[tem_mag_add] - ang_2[tem_mag_add]
    else:
        tem = ang_1[tem_mag_mul] - ang_2[tem_mag_mul]
    if tem.size == 0:
        print("None")
    tem = np.abs(tem)
    tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
    #tem = tem * tem_ang_weight[tem_mag_add]
    #result = np.mean(np.square(tem))
    print(np.mean(tem))
    
    
    #0/0
    ta = torch.FloatTensor(np.rollaxis(a[np.newaxis],3,1))
    tx = torch.FloatTensor(np.rollaxis(x[np.newaxis]/255,3,1))
    tb = torch.FloatTensor(np.rollaxis(b[np.newaxis],3,1))
    ty = torch.FloatTensor(np.rollaxis(y[np.newaxis]/255,3,1))
    tc = torch.FloatTensor(np.rollaxis(c[np.newaxis],3,1))
    tz = torch.FloatTensor(np.rollaxis(z[np.newaxis]/255,3,1))
    td = torch.FloatTensor(np.rollaxis(d[np.newaxis],3,1))
    import matplotlib.pyplot as plt
    def shows(x,t,y, id):
        #cv2.imwrite(str(id)+".png", cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        #cv2.imwrite(str(id+1)+".png", cv2.cvtColor((t*255).astype("uint8"), cv2.COLOR_RGB2BGR))
        #cv2.imwrite(str(id+2)+".png", cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
        '''
        plt.figure()
        plt.axis('off')
        plt.imshow(x)
        plt.figure()
        plt.axis('off')
        plt.imshow(t)
        plt.figure()
        plt.axis('off')
        plt.imshow(y)
        '''
        return compare_mse((t*255).astype("int"),y.astype("int"))
    
    a1 = shows(x,np.rollaxis(warp(tx, tb).numpy(),1,4)[0],y,0)
    a1 += shows(np.rollaxis(warp(tx, tb).numpy(),1,4)[0],np.rollaxis(warp(warp(tx, tb), td).numpy(),1,4)[0],z,3)
    
    a2 = shows(z,np.rollaxis(warp(tz, tc).numpy(),1,4)[0],y,6)
    a2 += shows(np.rollaxis(warp(tz, tc).numpy(),1,4)[0],np.rollaxis(warp(warp(tz, tc), ta).numpy(),1,4)[0],x,9)
    
    a3 = shows(y,np.rollaxis(warp(ty, ta).numpy(),1,4)[0],x,12)
    a3 += shows(y,np.rollaxis(warp(ty, td).numpy(),1,4)[0],z,15)
    
    print(a1)
    print(a2)
    print(a3)

    plt.show()
    
    
    0/0

    print(optical_direction(b,3*np.pi/2))
    print(optical_direction(c,3*np.pi/2))
    #b = p.pwc_predict_test(y, x, is_display = True)
    #c = p.pwc_predict_test(y, z, is_display = True)
    #d = p.pwc_predict(z, y, is_display = True)
    print(np.sum(x.astype('int')-y.astype("int")))

    '''import pickle
    with open("opt_" + str(i_2) + ".txt", "rb") as f:
        s = pickle.load(f)
        #print(s[0])
        #print(s[0].shape)
        #print(s[22])
        #print(s[22].shape)
        #display_img_pairs_w_flows([(y,x)], s[0][np.newaxis])
        #display_img_pairs_w_flows([(y,z)], s[22][np.newaxis])
        #print(np.sum(b-s[3]))
    '''
    np.set_printoptions(threshold=np.inf)
    #nn1 = np.load("0.npy")
    #nn2 = np.load("1.npy")
    #nn3 = np.load("5.npy")


    #ss = (int(x.shape[0]/2), int(x.shape[1]/2))
    #print(a[ss], " ", x[ss], " ", x.shape)
    #print(y[ss[0]+int(a[ss[0], ss[1],0]):ss[0]-int(a[ss[0], ss[1], 0]), ss[1]+int(a[ss[0],ss[1],1]):ss[1]-int(a[ss[0], ss[1],1])])
    #print(x[0,470:480])
    #print(y[0,420:430])
    #print(a[0, 470:480])
    #print(b[0, 420:430])
    #print(c[0, 420:430])
    print(optical_dis(b, c, 64))
    #print(optical_dis(s[i_1], s[i_3], 128))
    b = p.pwc_predict(y, x, is_display = True)
    print(np.max(b), " ", np.min(b))
    x=x/256
    y=y/256
    qq = np.zeros_like(y)
    ww = np.zeros(y.shape[:-1])
    for i in range(256):
        for j in range(256):
            ii = np.abs(b[i,j,0]-int(b[i,j,0]))
            jj = np.abs(b[i,j,1]-int(b[i,j,1]))
            if b[i,j,0] >= 0:
                iii = int(b[i,j,0])+1
            else:
                iii = int(b[i,j,0])-1
            if b[i,j,1] >= 0:
                jjj = int(b[i,j,1])+1
            else:
                jjj = int(b[i,j,1])-1
            if 0 < i+iii < 256 and 0 < j+jjj < 256:
                qq[i+iii, j+jjj] += ii*jj*y[i,j]
                ww[i+iii, j+jjj] += ii*jj
            if 0 < i+iii < 256 and 0 < j+int(b[i,j,1]) < 256:
                qq[i+iii, j+int(b[i,j,1])] += ii*(1-jj)*y[i,j]
                ww[i+iii, j+int(b[i,j,1])] += ii*(1-jj)
            if 0 < i+int(b[i,j,0]) < 256 and 0 < j+jjj < 256:
                qq[i+int(b[i,j,0]), j+jjj] += (1-ii)*jj*y[i,j]
                ww[i+int(b[i,j,0]), j+jjj] += (1-ii)*jj
            if 0 < i+int(b[i,j,0]) < 256 and 0 < j+int(b[i,j,1]) < 256:
                qq[i+int(b[i,j,0]), j+int(b[i,j,1])] += (1-ii)*(1-jj)*y[i,j]
                ww[i+int(b[i,j,0]), j+int(b[i,j,1])] += (1-ii)*(1-jj)
    for i in range(256):
        for j in range(256):
            if ww[i,j] != 0:
                qq[i,j] /= ww[i,j]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(y)
    plt.figure()
    plt.imshow(x)
    plt.figure()
    plt.imshow(qq)
    plt.show()
    
    0/0
    c = p.pwc_predict(y, z, is_display = True)
    b = -b
    
    #mag_a, ang_a = cv2.cartToPolar(a[...,0], a[...,1])
    mag_b, ang_b = cv2.cartToPolar(b[...,0], b[...,1])
    mag_c, ang_c = cv2.cartToPolar(c[...,0], c[...,1])
    mask_b = np.all(np.abs(x.astype("int")-y.astype("int"))<10, axis = -1)
    mask_c = np.all(np.abs(z.astype("int")-y.astype("int"))<10, axis = -1)
    
    mag_b = cv2.normalize(mag_b, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_c = cv2.normalize(mag_c, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_b[mask_b] = 0
    mag_c[mask_c] = 0
    
    eps = 128
    tem_mag_mul = (mag_b >= eps) * (mag_c >= eps)
    tem_mag_add = (mag_b >= eps) + (mag_c >= eps)
    tem_mag_other = tem_mag_add^tem_mag_mul
    ang_b[tem_mag_other] = ang_c[tem_mag_other] + 120*np.pi/180
    tem = np.abs(ang_b[tem_mag_add] - ang_c[tem_mag_add])
    #tem = np.abs(ang_b[tem_mag_mul] - ang_c[tem_mag_mul])
    print(np.max(mag_b))
    print(np.max(mag_c))
    print(np.min(mag_b))
    print(np.min(mag_c))
    tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
    #print(np.mean(tem))
    #print(np.mean(mag_b[temp]))
    #print(np.mean(mag_c[temp]))
    #print(np.mean(np.abs(mag_b[temp]-mag_c[temp])))
    print(np.mean(tem))
    
    mag_b, ang_b = cv2.cartToPolar(b[...,0], b[...,1])
    mag_c, ang_c = cv2.cartToPolar(c[...,0], c[...,1])
    mask_b = np.all(np.abs(x.astype("int")-y.astype("int"))<5, axis = -1)
    mask_c = np.all(np.abs(z.astype("int")-y.astype("int"))<5, axis = -1)
    
    mag_b = cv2.normalize(mag_b, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_c = cv2.normalize(mag_c, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    mag_b[mask_b] = 0
    mag_c[mask_c] = 0
    
    eps = 128
    tem_mag_mul = (mag_b > eps) * (mag_c > eps)
    tem_mag_add = (mag_b > eps) + (mag_c > eps)
    tem_mag_other = tem_mag_add^tem_mag_mul
    ang_b[tem_mag_other] = ang_c[tem_mag_other] + 120*np.pi/180
    tem = np.abs(ang_b[tem_mag_add] - ang_c[tem_mag_add])
    #tem = np.abs(ang_b[tem_mag_mul] - ang_c[tem_mag_mul])
    tem_ang_weight = np.ones_like(ang_b)
    tem_ang_weight[tem_mag_other] = 2
    print(np.max(mag_b))
    print(np.max(mag_c))
    print(np.min(mag_b))
    print(np.min(mag_c))
    tem[tem>np.pi] = 2*np.pi - tem[tem>np.pi]
    tem = tem * tem_ang_weight[tem_mag_add]
    #print(np.mean(tem))
    #print(np.mean(mag_b[temp]))
    #print(np.mean(mag_c[temp]))
    #print(np.mean(np.abs(mag_b[temp]-mag_c[temp])))

    print(np.mean(tem))
    #print(np.average(tem, weights=(mag_b[tem_mag_mul] + mag_c[tem_mag_mul])))
    
    
    print(optical_dis(-b, c))
    






