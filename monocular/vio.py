import cv2
import os
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"fifth"))
        # print(self.images)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
        self.predictor = self._init_panoptic_segmentation()

    def _init_panoptic_segmentation(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        # sort
        image_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def _get_panoptic_masks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        panoptic_seg, segments_info = self.predictor(rgb_image)["panoptic_seg"]
        
        things_mask = np.zeros_like(image, dtype=bool)
        stuff_mask = np.zeros_like(image, dtype=bool)
        for seg in segments_info:
            if seg["isthing"]:
                things_mask |= (panoptic_seg.cpu().numpy() == seg["id"])
            else:
                stuff_mask |= (panoptic_seg.cpu().numpy() == seg["id"])
                # cv2.imshow(stuff_mask)
        unknown_mask = ~(things_mask | stuff_mask)
        return things_mask, stuff_mask, unknown_mask
    
    def _filter_keypoints(self, kp, des, stuff_mask):
        static_kp, static_des = [], []
        for pt, desc in zip(kp, des):
            x, y = map(int, pt.pt)
            if stuff_mask[y, x]:
                static_kp.append(pt)
                static_des.append(desc)
        return static_kp, np.array(static_des)
    
    def _filter_dynamic_keypoints(self, kp, des, stuff_mask):
        dynamic_kp, dynamic_des = [], []
        for pt, desc in zip(kp, des):
            x, y = map(int, pt.pt)
            if not stuff_mask[y, x]:
                dynamic_kp.append(pt)
                dynamic_des.append(desc)
        return dynamic_kp, np.array(dynamic_des)
    

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        
        things_mask_curr, stuff_mask_curr, unknown_mask_curr = self._get_panoptic_masks(self.images[i])
        things_mask_prev, stuff_mask_prev, unknown_mask_prev = self._get_panoptic_masks(self.images[i-1])
        
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        
        kp1, des1 = self._filter_dynamic_keypoints(kp1, des1, things_mask_prev)
        kp2, des2 = self._filter_dynamic_keypoints(kp2, des2, things_mask_curr)
        
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2
    
    def get_matches_sift(self, i):
        """
        Detect and compute keypoints and descriptors using SIFT
        from the (i-1)'th and i'th images, filter them using FLANN,
        and return good matches.

        Parameters
        ----------
        i (int): The current frame index

        Returns
        -------
        q1 (ndarray): Good keypoints matches in (i-1)'th image
        q2 (ndarray): Good keypoints matches in i'th image
        """

        # Generate panoptic masks (optional for future filtering)
        things_mask_curr, stuff_mask_curr, unknown_mask_curr = self._get_panoptic_masks(self.images[i])
        things_mask_prev, stuff_mask_prev, unknown_mask_prev = self._get_panoptic_masks(self.images[i - 1])
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = sift.detectAndCompute(self.images[i], None)
        
        ic(len(kp1), len(kp2))
        ic(len(des1), len(des2))
        
        kp1_t, kp2_t = kp1, kp2
        des1_t, des2_t = des1, des2
        
        # Optional dynamic keypoint filtering (commented for now)
        kp1, des1 = self._filter_dynamic_keypoints(kp1, des1, things_mask_prev)
        kp2, des2 = self._filter_dynamic_keypoints(kp2, des2, things_mask_curr)

        if des1 is None or des2 is None:
            kp1, kp2 = kp1_t, kp2_t
            des1, des2 = des1_t, des2_t
        if len(des2) < len(des1):
            des1 = des1[:len(des2)]
        else:
            des2 = des2[:len(des1)]
            
        ic('after filter',len(kp1), len(kp2), i)
        # FLANN-based matcher for SIFT
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # Visualize matches
        draw_params = dict(matchColor=-1,
                        singlePointColor=None,
                        matchesMask=None,
                        flags=2)
        img3 = cv2.drawMatches(self.images[i], kp2, self.images[i - 1], kp1, good, None, **draw_params)
        cv2.imshow("SIFT Matches", img3)
        cv2.waitKey(200)

        # Extract good keypoints positions
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2


    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def main():
    data_dir = "test-1"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)


    gt_path = []
    # gt_path = np.load('gt_path.npy')
    # with_green = np.load('withgreen.npy')
    # without_green = np.load('without_green.npy')
    estimated_path = []
    count = len(os.listdir(os.path.join(data_dir, "image_l")))
    estimated_path = []
    ignore = [74]
    # Loop over all images in the dataset
    for i in tqdm(range(count), unit="pose"):
        if i == 0:
            cur_pose = np.eye(4)
        else:
            q1, q2 = vo.get_matches_sift(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    # plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

    # plt.plot([p[0] for p in gt_path], [p[1] for p in gt_path], label="GT path")
    # np.save('without_green', np.array(estimated_path))
    # plt.plot([p[0] for p in with_green], [p[1] for p in with_green], label="With green")
    # plt.plot([p[0] for p in without_green], [p[1] for p in without_green], label="Without green")
    try:
        gt_path = np.load('/home/mahesh/vio/test-1/gt_five_path.npy')
        plt.plot([p[0] for p in gt_path], [p[1] for p in gt_path], label="GT path")
    except FileNotFoundError:
        print("GT path not found")
    plt.plot([p[0] for p in estimated_path], [p[1] for p in estimated_path], label="Estimated path")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Visual Odometry")
    plt.legend()
    plt.show()

# def main():
#     data_dir = "test-1"  # or "KITTI_sequence_2" if needed
#     vo = VisualOdometry(data_dir)

#     count = len(os.listdir(os.path.join(data_dir, "image_l")))
#     estimated_path = []
#     ignore = [74]
#     # Loop over all images in the dataset
#     for i in tqdm(range(count), unit="pose"):
#         if i == 0:
#             # Initialize with the origin (identity matrix)
#             cur_pose = np.eye(4)
#         else:
#             if i in ignore:
#                 continue
#             q1, q2 = vo.get_matches_sift(i)
#             transf = vo.get_pose(q1, q2)
#             cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
#         # Save the (X, Z) position of the current pose
#         estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

#     # Plot the estimated path
#     plt.plot([p[0] for p in estimated_path], [p[1] for p in estimated_path], label="Estimated path")
#     plt.xlabel("X")
#     plt.ylabel("Z")
#     plt.title("Visual Odometry")
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    main()