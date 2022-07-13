import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy import linalg
import time
from tqdm import tqdm


print("Enter the Dataset Number ")
Dataset_Num = int(input())

if Dataset_Num <= 0 or Dataset_Num >=4:
    print("Invalid Dataset Number")
    exit(0)

if Dataset_Num == 1:

    K_left = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    K_right = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax=88.39,1758.23,1920,1080,220,55,195
    folder_name = r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Project3\data\curule\*.png"

elif Dataset_Num == 2:
    K_left = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    K_right = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax =221.76,1742.11,1920,1080,100,29,61
    folder_name = r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Project3\data\octagon\*.png"

elif Dataset_Num == 3:
    K_left = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    K_right = np.array([[1729.05, 0 ,-364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    Baseline,Focallength,width,height,ndisp,vmin,vmax =174.019,1729.05,1920,1080,180,25,150
    folder_name = r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Project3\data\pendulum\*.png"

images = []

path = glob.glob(folder_name)
for file in path:
    image = cv2.imread(file)
    images.append(image)


def rescale(image, scale):

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


sift = cv2.SIFT_create()

image_Left = images[0].copy()
image_Right = images[1].copy()

image_Left = rescale(image_Left,0.6)
image_Right = rescale(image_Right,0.6)

h1, w1 = image_Left.shape[:2]
h2, w2 = image_Right.shape[:2]

image_Left_gray = cv2.cvtColor(image_Left, cv2.COLOR_BGR2GRAY) 
image_Right_gray = cv2.cvtColor(image_Right, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(image_Left_gray, None)
kp2, des2 = sift.detectAndCompute(image_Right_gray, None)

bf = cv2.BFMatcher()
matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

matched_image = cv2.drawMatches(image_Left_gray,kp1,image_Right_gray,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matched_image)
plt.show()

chosen_matches = matches[:100]

matched_pts_left = np.array([kp1[m.queryIdx].pt for m in chosen_matches]).reshape(-1, 2)
matched_pts_right = np.array([kp2[m.trainIdx].pt for m in chosen_matches]).reshape(-1, 2)

matched_pts = (matched_pts_left,matched_pts_right)

class FundamentalMatrix():

    '''To normalize the function, the code is referred from the below link'''
    '''https://github.com/h-gokul/StereoDepthEstimation/blob/master/Code/misc/EstimateFudamentalMatrix.py'''

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
        
        pL,pR = data[0], data[1]
        if normalised == True:
            pL_norm, T1 = self.normalize(pL)
            pR_norm, T2 = self.normalize(pR)
        else:
            pL_norm,pR_norm = pL,pR
            
        A = np.zeros((len(pL_norm),9))

        for i in range(0, len(pL_norm)):
            x_1,y_1 = pL_norm[i][0], pL_norm[i][1]
            x_2,y_2 = pR_norm[i][0], pR_norm[i][1]
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

    def error(self,pL,pR,F): 

        pLtmp=np.array([pL[0], pL[1], 1]).T
        pRtmp=np.array([pR[0], pR[1], 1])

        error = np.abs(np.dot(pLtmp, np.dot(F, pRtmp)))
        
        return error
    
    def Calculate_erros(self,data, F):

        pLpts, pRpts =  data[0], data[1]
        errors = []
        for i,(pt1, pt2) in enumerate(zip(pLpts,pRpts)):
            error = self.error(pt1,pt2,F)
            errors.append(error)
        return errors

class Ransac:
    def __init__(self, weights):
        self.weights = weights
    
    def fit(self,data):

        if Dataset_Num == 1:
            num_iter = 1000
            threshold = 0.01
        if Dataset_Num == 2:
            num_iter = 1000
            threshold = 0.1
        if Dataset_Num == 3:
            num_iter = 1000
            threshold = 0.01

        num_sample = 25
        
        max_inlier_count = 0
        inlier_count = 0
        best_model = None
        iter_done = 0
        n_rows = len(data[0])
        mask = np.zeros(n_rows)
        best_mask = None


        while num_iter > iter_done:
            
            random_indices = np.random.choice(n_rows, size=num_sample)
            data = np.array(data)
            sample_data = np.zeros((data.shape[0], num_sample, data.shape[2]), dtype=np.float32)
            for i in range(0,data.shape[0]):
                sample_data[i] =  data[i][random_indices]
            
            sample_data = data[:num_sample, :]
            estimated_model = self.weights.fit(sample_data)

            errors = self.weights.Calculate_erros(data, estimated_model)

            for idx in range(len(errors)):
                if errors[idx] < threshold:
                    inlier_count+=1
                    mask[idx] = 1
 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model
                best_mask = mask

            # num_iter = (np.log(1 - 0.99))/np.log(1 - ((0.5)**8))

            inlier_count = 0
            iter_done = iter_done + 1

        return best_model,best_mask

F_M = FundamentalMatrix()
ransac_model = Ransac(F_M)
FM, inlier_mask = ransac_model.fit(matched_pts)

print("Fundamental matrix:", FM)

matched_pts_left_chosen = matched_pts_left[np.where(inlier_mask==1)]
matched_pts_right_chosen = matched_pts_right[np.where(inlier_mask==1)]

# print(len(matched_pts_left_chosen))

def EssentialMatrixFromFundamentalMatrix(K_left,K_right, FM):
    E = (K_right.T).dot(FM).dot(K_left)
    U,D,VT = np.linalg.svd(E)
    D = [1,1,0]
    EM = np.dot(U,np.dot(np.diag(D),VT))
    return EM

EM = EssentialMatrixFromFundamentalMatrix(K_left,K_right, FM)

print("Essential matrix:", EM)

def ExtractCameraPose(E):
    U, D, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W, VT)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W, VT)))
    C.append(U[:, 2])
    R.append(np.dot(U, np.dot(W.T, VT)))
    C.append(-U[:, 2])
    R.append(np.dot(U, np.dot(W.T, VT)))

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

R, C = ExtractCameraPose(EM)

print('Dataset1')
print("Rotation:",R,'\n')
print("Translation:",C,'\n')

Pts_3D = []
R1  = np.identity(3)
C1  = np.zeros((3, 1))
I = np.identity(3)


def Triangulation(PL, PR, ptsL, ptsR):
 
    A = [ptsL[1]*PL[2,:] - PL[1,:], PL[0,:] - ptsL[0]*PR[2,:], ptsR[1]*PR[2,:] - PR[1,:], PR[0,:] - ptsR[0]*PR[2,:]]

    A = np.array(A).reshape((4,4))
 
    A = np.dot(A.T, A)
    
    U, D, VT = linalg.svd(A, full_matrices = False)
 
    return VT[3,0:3]/VT[3,3]


for i in range(len(R)):
    R2 =  R[i]
    C2 =   C[i].reshape(3,1)
    ProjectionM_left = np.dot(K_left, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
    ProjectionM_right = np.dot(K_right, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))

    for pL,pR in zip(matched_pts_left_chosen, matched_pts_right_chosen):
        pts_3d = Triangulation(ProjectionM_left, ProjectionM_right, np.float32(pL), np.float32(pR) )
        pts_3d = np.array(pts_3d)
        Pts_3D.append(pts_3d)

def Cheirality_Condition(P_3D,C_,R_3):
    num_positive = 0
    for P in P_3D:
        P = P.reshape(-1,1)
        if R_3.dot(P - C_) > 0 and P[2]>0:
            num_positive+=1
    return num_positive

best_i = 0
max_Positive = 0

for i in range(len(R)):
    R_, C_ = R[i],  C[i].reshape(-1,1)
    R_3 = R_[2].reshape(1,-1)
    num_Positive = Cheirality_Condition(Pts_3D,C_,R_3)

    if num_Positive > max_Positive:
        best_i = i
        max_Positive = num_Positive

R_Config, C_Config, P3D = R[best_i], C[best_i], Pts_3D[best_i]

print(" R of camera Pose",R_Config)
print("C of camera Pose", C_Config)


'''Funtion to draw epipolar lines after using cv2.computeCorrespondEpilines'''

# def drawlines(imgL, imgR, lines, ptsL, ptsR):
    
#     r, c = img1.shape
#     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
      
#     for r, pt1, pt2 in zip(lines, ptsL, ptsR):
          
#         color = tuple(np.random.randint(0, 255,
#                                         3).tolist())
          
#         x0, y0 = map(int, [0, -r[2] / r[1] ])
#         x1, y1 = map(int, 
#                      [c, -(r[2] + r[0] * c) / r[1] ])
          
#         img1 = cv2.line(img1, 
#                         (x0, y0), (x1, y1), color, 1)
#         img1 = cv2.circle(img1,
#                           tuple(pt1), 5, color, -1)
#         img2 = cv2.circle(img2, 
#                           tuple(pt2), 5, color, -1)
#     return img1, img2

# linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1,
#                                                            1,
#                                                            2),
#                                           2, F)
# linesLeft = linesLeft.reshape(-1, 3)
# img5, img6 = drawlines(imgLeft, imgRight, 
#                        linesLeft, ptsLeft,
#                        ptsRight)
   
# linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2), 
#                                            1, F)
# linesRight = linesRight.reshape(-1, 3)
  
# img3, img4 = drawlines(imgRight, imgLeft, 
#                        linesRight, ptsRight,
#                        ptsLeft)




def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x


def getEpipolarLines(PtsL, PtsR, F, imgL, imgR, rectified = False):

    '''One of the elegant solutions to plot the lines, is referred from the below link'''
    '''https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/Code/Utils/GeometryUtils.py'''

    img_epi1 = imgL.copy()
    img_epi2 = imgR.copy()
    linesL, linesR = [], []

    for i in range(PtsL.shape[0]):
        x1 = np.array([PtsL[i,0], PtsL[i,1], 1]).reshape(3,1)
        x2 = np.array([PtsR[i,0], PtsR[i,1], 1]).reshape(3,1)

        lineR = np.dot(F, x1)
        linesR.append(lineR)

        lineL = np.dot(F.T, x2)
        linesL.append(lineL)
    
        if not rectified:
            y2_min = 0
            y2_max = imgR.shape[0]
            x2_min = getX(lineR, y2_min)
            x2_max = getX(lineR, y2_max)

            y1_min = 0
            y1_max = imgL.shape[0]
            x1_min = getX(lineL, y1_min)
            x1_max = getX(lineL, y1_max)
        else:
            x2_min = 0
            x2_max = imgR.shape[1] - 1
            y2_min = -lineR[2]/lineR[1]
            y2_max = -lineR[2]/lineR[1]

            x1_min = 0
            x1_max = imgL.shape[1] -1
            y1_min = -lineL[2]/lineL[1]
            y1_max = -lineL[2]/lineL[1]



        cv2.circle(img_epi2, (int(PtsR[i,0]),int(PtsR[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(PtsL[i,0]),int(PtsL[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    concat = np.concatenate((img_epi1, img_epi2), axis = 1)
    plt.imshow(concat)
    plt.show()
    return linesL, linesR

linesL, linesR = getEpipolarLines(matched_pts_left_chosen, matched_pts_right_chosen, FM, image_Left, image_Right, False)

_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(matched_pts_left_chosen), np.float32(matched_pts_right_chosen), FM, imgSize=(w1, h1))

print(" H1 ", H1)
print(" H2 ", H2)

image_Left_rectified = cv2.warpPerspective(image_Left, H1, (w1, h1))
image_Right_rectified = cv2.warpPerspective(image_Right, H2, (w2, h2))

matched_pts_left_chosen_rectified = cv2.perspectiveTransform(matched_pts_left_chosen.reshape(-1, 1, 2), H1).reshape(-1,2)
matched_pts_right_chosen_rectified = cv2.perspectiveTransform(matched_pts_right_chosen.reshape(-1, 1, 2), H2).reshape(-1,2)

H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
FM_rectified = np.dot(H2_T_inv, np.dot(FM, H1_inv))

linesL_rectified, linesR_recrified = getEpipolarLines(matched_pts_left_chosen_rectified, matched_pts_right_chosen_rectified, FM_rectified, image_Left_rectified, image_Right_rectified, True)


image_Left_rectified_resized = rescale(image_Left_rectified,0.4)
image_Right_rectified_resized = rescale(image_Right_rectified,0.4)

imgL_Rectified_gray = cv2.cvtColor(image_Left_rectified_resized,cv2.COLOR_BGR2GRAY)
imgR_Rectified_gray = cv2.cvtColor(image_Right_rectified_resized,cv2.COLOR_BGR2GRAY)

height,width,_ = image_Right_rectified_resized.shape

# image_Left = cv2.cvtColor(image_Left,cv2.COLOR_BGR2GRAY)
# image_Right = cv2.cvtColor(image_Right,cv2.COLOR_BGR2GRAY)
# stereo = cv2.StereoBM_create(numDisparities=96, blockSize=5)
# disparity = stereo.compute(image_Left,image_Right)
# plt.imshow(disparity,'gray')
# plt.show()



Disparity_img = np.zeros((height,width))

height_block =  3   #5-3     #4-2
max_depth = vmax
max_ssd =   100     #50-3 50           #50-2
cost = 100

def norm(arr):
    num = arr-arr.mean()
    den  = np.sqrt((num**2).sum())
    return num/den

for h in tqdm(range(height_block,height-height_block)):
    Lshift = 0
    DSI = np.zeros((width-2*height_block,max_depth))
    for w in range(height_block,width-height_block):
        if (w-height_block)>= max_depth:
            Lshift+=1
        for d in range(0,max_depth):
            Rshift = w-d-height_block
            if(Rshift>=0):
                left_patch = imgL_Rectified_gray[h-height_block:h+height_block+1,w-height_block:w+height_block+1].copy()
                right_patch = imgR_Rectified_gray[h-height_block:h+height_block+1,w-height_block-Rshift+Lshift:w+height_block+1-Rshift+Lshift].copy()
                ssd = ((norm(left_patch)-norm(right_patch))**2).sum()
                index  = Rshift-Lshift
            else:
                index = d
                ssd = max_ssd
                pass
            DSI[w-height_block,index] = ssd

    DSI[DSI==0] = np.inf

    for i in range(1,DSI.shape[0]):
        for j in range(i,min(DSI.shape[1],i+max_depth)):
            DSI[i,j-i] = DSI[i,j-i] + min(DSI[i-1,:] + cost*np.abs(np.arange(0,DSI.shape[1])-(i-j)))
    
    Disparity_img[h,height_block:-height_block] = np.argmin(DSI, axis = 1)

rescaled_Disparity = ((Disparity_img/Disparity_img.max())*255).astype(np.uint8)

plt.imshow(rescaled_Disparity, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(rescaled_Disparity, cmap='gray', interpolation='nearest')
plt.show()

depth = (Baseline * Focallength) / (Disparity_img + 1e-10)
depth[depth > 1000000] = 1000000

rescaled_Depth = np.uint8(depth * 255 / np.max(depth))
plt.imshow(rescaled_Depth, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(rescaled_Depth, cmap='gray', interpolation='nearest')
plt.show()





