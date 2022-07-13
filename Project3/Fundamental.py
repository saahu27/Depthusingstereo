from re import U
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math



class FundamentalMatrix():
    
    def normalize(self,uv):

        uv_ = np.mean(uv, axis=0)
        u_,v_ = uv_[0], uv_[1]
        u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

        s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
        T_scale = np.diag([s,s,1])
        T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
        T = T_scale.dot(T_trans)

        x_ = np.column_stack((uv, np.ones(len(uv))))
        x_norm = (T.dot(x_.T)).T

        return  x_norm, T
    

    def fit(self,data):
        normalised = True
        
        xL,xR = data[0], data[1]
        if normalised == True:
            xL_norm, T1 = self.normalize(xL)
            xR_norm, T2 = self.normalize(xR)
        else:
            xL_norm,xR_norm = xL,xR
            
        A = np.zeros((len(xL_norm),9))

        for i in range(0, len(xL_norm)):
            x_1,y_1 = xL_norm[i][0], xL_norm[i][1]
            x_2,y_2 = xR_norm[i][0], xR_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, D, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)
        
        U, D, VT = np.linalg.svd(F)
        D = np.diag(D)
        D[2,2] = 0
        F = np.dot(U,np.dot(D,VT))
        
        if normalised:
            F = np.dot((T2.T),np.dot( F, T1))
        return F

    def error(self,xL,xR,F): 

        xLtmp=np.array([xL[0], xL[1], 1]).T
        xRtmp=np.array([xR[0], xR[1], 1])

        error = np.abs(np.dot(xLtmp, np.dot(F, xRtmp)))
        
        return error
    
    def check(self,data, F):

        xLpts, xRpts =  data[0], data[1]
        errors = []
        for i,(pt1, pt2) in enumerate(zip(xLpts,xRpts)):
            error = self.error(pt1,pt2,F)
            errors.append(error)
        return errors

class Ransac:
    def __init__(self, weights):
        self.weights = weights
    
    def fit(self,data,threshold):

        num_iter = 100
        num_sample = 8
        threshold = 0.02
        max_inlier_count = 0
        inlier_count = 0
        best_model = None
        iter_done = 0
        n_rows = len(data[0])
        mask = np.zeros(n_rows)


        while num_iter > iter_done:
            
            random_indices = np.random.choice(n_rows, size=num_sample)
            data = np.array(data)
            sample_data = np.zeros((data.shape[0], num_sample, data.shape[2]), dtype=np.float32)
            for i in range(data.shape[0]):
                sample_data[i] =  data[i][random_indices]
            
            sample_data = data[:num_sample, :]
            estimated_model = self.weights.fit(sample_data)

            errors = self.weights.check(data, estimated_model)

            for idx in range(len(errors)):
                if errors[idx] < threshold:
                    inlier_count+=1
                    mask[idx] = 1
 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model
                best_mask = mask

            inlier_count = 0
            iter_done = iter_done + 1

        return best_model,best_mask