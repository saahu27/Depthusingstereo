import numpy as np
import cv2
# import matplotlib
# matplotlib.use('Agg')
import argparse
import math
from tqdm import tqdm
import matplotlib.pyplot as plt


def siftFeatures2Array(sift_matches, kp1, kp2):
    matched_pairs = []
    for i, m1 in enumerate(sift_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    return matched_pairs

def normalize(uv):

    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def makeImageSizeSame(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized

def getEpipolarLines(set1, set2, F, image0, image1, filename, rectified = False):
    # set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = getX(line2, y2_min)
            x2_max = getX(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = getX(line1, y1_min)
            x1_max = getX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]



        cv2.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*255)), 2)
    

        cv2.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*255)), 2)

    image_1, image_2 = makeImageSizeSame([img_epi1, img_epi2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv2.resize(concat, (1920, 660))
    #displaySaveImage(concat, file_name)
    plt.imshow(concat)
    plt.savefig(filename)
    # cv2.imwrite("epilines.png", concat)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return lines1, lines2

def EstimateFundamentalMatrix(feature_matches):
    normalised = True

    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def errorF(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1tmp=np.array([x1[0], x1[1], 1]).T
    x2tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1tmp, np.dot(F, x2tmp))
    
    return np.abs(error)

def getInliers(features):
    n_iterations = 1000
    error_thresh = 0.02
    inliers_thresh = 0
    chosen_indices = []
    chosen_f = 0

    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = EstimateFundamentalMatrix(features_8)
        for j in range(n_rows):
            feature = features[j]
            error = errorF(feature, f_8)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_f = f_8

    filtered_features = features[chosen_indices, :]
    return chosen_f, filtered_features

def getEssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected

# def drawLines(im1, lines, pts):
#     img1 = im1.copy()
#     for l, pt in zip(lines,pts):
#         l_color = (0,255,0)
#         (x0,y0), (x1,y1) =l[0], l[1]
#         img1 = cv2.line(img1, (int(x0),int(y0)), (int(x1),int(y1)), l_color,1)
        
#         p_color = (0,0,255)
#         x, y = pt[0], pt[1]
#         cv2.circle(img1, (int(x), int(y)), 1, p_color, -1)
        
    # return img1
    
def ExtractCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

def get3DPoints(K1, K2, matched_pairs, R2, C2):
    pts3D_4 = []
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))

    for i in range(len(C2)):
        pts3D = []
        x1 = matched_pairs[:,0:2].T
        x2 = matched_pairs[:,2:4].T

        P2 = np.dot(K2, np.dot(R2[i], np.hstack((I, -C2[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)  
        pts3D_4.append(X)
    return pts3D_4

def Cheirality_Condition(P_3D,C_,R_3):
    num_positive = 0
    for P in P_3D:
        P = P.reshape(-1,1)
        if R_3.dot(P - C_) > 0 and P[2]>0:
            num_positive+=1
    return num_positive

def Triangulation(PL, PR, ptsL, ptsR):
  
     A = [ptsL[1]*PL[2,:] - PL[1,:], PL[0,:] - ptsL[0]*PR[2,:], ptsR[1]*PR[2,:] - PR[1,:], PR[0,:] - ptsR[0]*PR[2,:]]

     A = np.array(A).reshape((4,4))
  
     A = np.dot(A.T, A)
     
     U, D, VT = np.linalg.svd(A, full_matrices = False)
  
     return VT[3,0:3]/VT[3,3]

def main():
    
    # img1 = cv2.imread('im0_cu.png')
    # img2 = cv2.imread('im1_cu.png')
    
    dataset_number = 2
    
    
    if dataset_number == 1:
        K1 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15],[ 0, 0, 1]])
        K2 = K1
        baseline=88.39
        f = K1[0,0]
        depth_thresh = 100000
        img1 = cv2.imread('im0_cu.png')
        img2 = cv2.imread('im1_cu.png')
        window = 5

    elif dataset_number == 2:
        K1 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22],[ 0, 0, 1]])
        K2 = K1
        baseline=221.76
        f = K1[0,0]
        depth_thresh = 1000000
        img1 = cv2.imread('im0_oct.png')
        img2 = cv2.imread('im1_oct.png')
        window = 7

    elif dataset_number == 3:
        K1 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22],[ 0, 0, 1]])
        K2 = K1
        baseline=537.75
        f = K1[0,0]
        depth_thresh = 100000
        img1 = cv2.imread('im0_pend.png')
        img2 = cv2.imread('im1_pend.png')
        window = 3

    else:
        print("invalid datset number")
    print('baseline val= ',baseline)
    
    sift = cv2.SIFT_create()
    image0 = img1
    image1 = img2

    image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

#------------- CALIBRATION ----------------------    
    image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) 
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    print("Finding matches")
    kp1, des1 = sift.detectAndCompute(image0_gray, None)
    kp2, des2 = sift.detectAndCompute(image1_gray, None)
    
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]
    
    matched_image = cv2.drawMatches(image0_rgb,kp1,image1_rgb,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(matched_image)
    # plt.savefig('curule_matched_image.png')
    # cv2.imshow('s', matched_image)
    print('draw matches plot done')
    
    matched_pairs = siftFeatures2Array(chosen_matches, kp1, kp2)
    print("Estimating F and E matrix \n")
    F_best, matched_pairs_inliers = getInliers(matched_pairs)
    
    # matched_img_1 = drawLines(image0, lines, pts)
    
    E = getEssentialMatrix(K1, K2, F_best)
    
    print("For DATASET", dataset_number  ,"\n")
    # print('The Funndamental Matrix: \n',F_best,'\n')
    # print('The Essential Matrix: \n', E,'\n')
    
    R2_, C2_ = ExtractCameraPose(E)
    # pts3D_4 = get3DPoints(K1, K2, matched_pairs_inliers, R2, C2)
    
    # print('The Rotation Matrix: ', R2,'\n')
    # print('The Translation Matrix: ',C2, '\n')
    Pts_3D = []
    R1  = np.identity(3)
    C1  = np.zeros((3, 1))
    I = np.identity(3)


   


    for i in range(len(R2_)):
        R2 =  R2_[i]
        C2 =   C2_[i].reshape(3,1)
        ProjectionM_left = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
        ProjectionM_right = np.dot(K2, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))

        for xL,xR in zip(matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]):
            # pts_3d = Triangulation(ProjectionM_left, ProjectionM_right, np.float32(xL), np.float32(xR) )
            pts_3d = cv2.triangulatePoints(ProjectionM_left, ProjectionM_right, np.float32(xL), np.float32(xR))
            pts_3d = np.array(pts_3d)
            pts_3d = pts_3d[0:3,0]
            Pts_3D.append(pts_3d)

    

    best_i = 0
    max_Positive = 0

    for i in range(len(R2_)):
        R_, C_ = R2_[i],  C2_[i].reshape(-1,1)
        R_3 = R_[2].reshape(1,-1)
        num_Positive = Cheirality_Condition(Pts_3D,C_,R_3)

        if num_Positive > max_Positive:
            best_i = i
            max_Positive = num_Positive

    R_Config, C_Config, P3D = R2_[best_i], C2_[best_i], Pts_3D[best_i]

    # print(" Rotation Matrix of Camera Pose: \n",R_Config,'\n')
    # print("Translation Matrix of Camera Pose: \n", C_Config)
    
#---------------- RECTIFICATION --------------------------
    
    set1,set2= matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
    
    #lines1, lines2 = getEpipolarLines(set1, set2, F_best, image0, image1, "COPYepi_polar_lines_" + str(dataset_number)+ ".png", False)
    
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), F_best, imgSize=(w1, h1))
    print("Estimated H1 and H2 as \n Homography Matrix 1: \n", H1,'\nHomography Matrix 2:\n ', H2)
    
    img1_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(image1, H2, (w2, h2))
       
    plt.plot(img1_rectified)
    plt.show()
    set1_rectified = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
    set2_rectified = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)
    
    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))

    # lines1_rectified, lines2_recrified = getEpipolarLines(set1_rectified, set2_rectified, F_rectified, img1_rectified, img2_rectified, "joRECT_epi_polar_lines_" + str(dataset_number)+ ".png",  True)
  
    img1_rectified_reshaped = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
    img2_rectified_reshaped = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))

    img1_rectified_reshaped = cv2.cvtColor(img1_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.cvtColor(img2_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    
#------------------ CORRESPONDENCE ------------------------------
  
    left_array, right_array = img1_rectified_reshaped, img2_rectified_reshaped
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        raise "Left-Right image shape mismatch!"
    h, w = left_array.shape
    disparity_map = np.zeros((h, w))
  
    x_new = w - (2 * window)
    for y in tqdm(range(window, h-window)):
        block_left_array = []
        block_right_array = []
        for x in range(window, w-window):
            block_left = left_array[y:y + window,
                                    x:x + window]
            block_left_array.append(block_left.flatten())
  
            block_right = right_array[y:y + window,
                                    x:x + window]
            block_right_array.append(block_right.flatten())
  
        block_left_array = np.array(block_left_array)
        block_left_array = np.repeat(block_left_array[:, :, np.newaxis], x_new, axis=2)
  
        block_right_array = np.array(block_right_array)
        block_right_array = np.repeat(block_right_array[:, :, np.newaxis], x_new, axis=2)
        block_right_array = block_right_array.T
  
        abs_diff = np.abs(block_left_array - block_right_array)
        sum_abs_diff = np.sum(abs_diff, axis = 1)
        idx = np.argmin(sum_abs_diff, axis = 0)
        disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity 
  
  
  
  
    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('disparity_image_heat' +str(dataset_number)+ ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('disparity_image_gray' +str(dataset_number)+ ".png")

#--------------------DEPTH--------------------------------------------------
    
    if(dataset_number == 1):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 2):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 3):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    else:
        print("Invalid dataset number")
    
  
    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('depth_image_heat' +str(dataset_number)+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('depth_image_gray' +str(dataset_number)+ ".png")


    
if __name__ == '__main__':
    main()