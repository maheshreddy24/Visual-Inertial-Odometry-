import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
import pandas as pd
import datetime

class visual_odometry():
    def __init__(self):
        pass
    
    def compute_left_disparity_map(self, img_left, img_right, matcher ='bm', rgb = False, verbose = False):
        sad_window = 6
        num_disparities = sad_window*16
        block_size = 11
        matcher_name = matcher
        
        if matcher_name == 'bm':
            matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                        blockSize=block_size
                                        )
            
        elif matcher_name == 'sgbm':
            matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                            minDisparity=0,
                                            blockSize=block_size,
                                            P1 = 8 * 3 * sad_window ** 2,
                                            P2 = 32 * 3 * sad_window ** 2,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                        )
        if rgb:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        start = datetime.datetime.now()
        disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
        end = datetime.datetime.now()
        if verbose:
            print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)
        
        return disp_left

    def decompose_projection_matrix(self, p):
        # this decomposes the projection matrix
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
        t = (t / t[3])[:3]
        
        return k, r, t
    
    def calc_depth_map(self, disp_left, k_left, t_left, t_right, rectified = True):
        
        '''
        Calculate depth map using a disparity map, intrinsic camera matrix, and translation vectors
        from camera extrinsic matrices (to calculate baseline). Note that default behavior is for
        rectified projection matrix for right camera. If using a regular projection matrix, pass
        rectified=False to avoid issues.
        
        Arguments:
        disp_left -- disparity map of left camera
        k_left -- intrinsic matrix for left camera
        t_left -- translation vector for left camera
        t_right -- translation vector for right camera
        
        Optional Arguments:
        rectified -- (bool) set to False if t_right is not from rectified projection matrix
        
        Returns:
        depth_map -- calculated depth map for left camera
        
        '''
        # Get focal length of x axis for left camera
        f = k_left[0][0]
        
        # Calculate baseline of stereo pair
        if rectified:
            b = t_right[0] - t_left[0] 
        else:
            b = t_left[0] - t_right[0]
            
        # Avoid instability and division by zero
        disp_left[disp_left == 0.0] = 0.1
        disp_left[disp_left == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth_map = np.ones(disp_left.shape)
        depth_map = f * b / disp_left
        
        return depth_map
            
    # this function takes the left and right images, the projection matrices for the left and right cameras,
    def stereo_2_depth(self, img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, 
                    rectified=True):
        '''
        Takes stereo pair of images and returns a depth map for the left camera. If your projection
        matrices are not rectified, set rectified=False.
        
        Arguments:
        img_left -- image of left camera
        img_right -- image of right camera
        P0 -- Projection matrix for the left camera
        P1 -- Projection matrix for the right camera
        
        Optional Arguments:
        matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM
        rgb -- (bool) set to True if images passed are RGB. Default is False
        verbose -- (bool) set to True to report computation time and method
        rectified -- (bool) set to False if P1 not rectified to P0. Default is True
        
        Returns:
        depth -- depth map for left camera
        
        '''
        # Compute disparity map
        disp = self.compute_left_disparity_map(img_left, 
                                        img_right, 
                                        matcher=matcher, 
                                        rgb=rgb, 
                                        verbose=verbose)
        # Decompose projection matrices
        k_left, r_left, t_left = self.decompose_projection_matrix(P0)
        k_right, r_right, t_right = self.decompose_projection_matrix(P1)
        # Calculate depth map for left camera
        depth = self.calc_depth_map(disp, k_left, t_left, t_right)
        
        return depth
    
    #this function extracts features
    # the variable mask is added if in case we want to mask the image
    def extract_features(self, img, detector = 'sift', mask = None):
        if detector == 'sift':
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, mask)
        elif detector == 'orb':
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(img, mask)
        elif detector == 'surf':    
            surf = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = surf.detectAndCompute(img, mask)
        
        return keypoints, descriptors
    
    def filter_matches(self, matches):
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)
        return good_matches
    
    def match_features(self, des1, des2, matching = 'BF', detector = 'sift',  sort = True, k = 2):
        
        if matching == 'BF':
            if detector == "sift":
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            elif detector == "orb":
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif matching == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=k)
        
        if sort:
            matches = sorted(matches, key=lambda x: x.distance)
        
        return self.filter_matches(matches, k)
    
    def visualise_matches(self, kp1, img1, kp2, img2, matches):
        
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.axis('off')
        plt.show()
        
    def estimate_motion(self, kp1, kp2, k, matches, depth = None, max_depth = 3000):
        '''
        Estimate motion between two sets of keypoints using the essential matrix. This is done by
        computing the fundamental matrix and then decomposing it to get the rotation and translation
        matrices.
        
        Arguments:
        kp1 -- keypoints from first image
        kp2 -- keypoints from second image
        k -- intrinsic matrix for camera
        
        Optional Arguments:
        depth -- (np.array) depth map for left camera. Default is None
        max_depth -- (int) maximum depth for filtering. Default is 3000
        
        Returns:
        R -- rotation matrix
        t -- translation vector '''
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
        image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])
        

        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]
    
        object_points = np.zeros((0, 3))
        delete_indices = []
        
        for i, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            if z < max_depth:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                object_points = np.vstack((object_points, np.array([[x, y, z]])))
            else:
                delete_indices.append(i)
        image1_points = np.delete(image1_points, delete_indices, axis=0)
        image2_points = np.delete(image2_points, delete_indices, axis=0)
        
        _, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image1_points, k, None)  # this has the 3D points
        rmat = cv2.Rodrigues(rvec)[0]
        
        return rmat, tvec, image1_points, image2_points
    
    
    def visual_odometry(self, dataset_handler, matcher = 'bm', rgb = False, verbose = False, rectified = True):
        trajectory = np.zeros((dataset_handler.num_frames, 3, 4), np.float32) #! this the empty trajectory
        trajectory[0] = np.eye(4)
        
        imheight = dataset_handler.imheight
        imwidth = dataset_handler.imwidth
        
        k_left, r_left, t_left = self.decompose_projection_matrix(dataset_handler.P0)
        
