import cv2
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch3d.transforms.so3 import so3_exp_map
from icecream import ic
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from icecream import ic
# from BA_utils import BundleAdjustment
from tqdm import tqdm

class PanopticSLAM:
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "forth"))
        
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
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ').reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses
    
    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in tqdm(image_paths, desc="Loading images")]

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
        things_mask_curr, stuff_mask_curr, unknown_mask_curr = self._get_panoptic_masks(self.images[i])
        things_mask_prev, stuff_mask_prev, unknown_mask_prev = self._get_panoptic_masks(self.images[i-1])
        
        # Apply histogram equalization and Gaussian blur
        self.images[i-1] = cv2.equalizeHist(self.images[i-1])
        self.images[i-1] = cv2.GaussianBlur(self.images[i-1], (5, 5), 0)
        self.images[i] = cv2.equalizeHist(self.images[i])
        self.images[i] = cv2.GaussianBlur(self.images[i], (5, 5), 0)
        kp1, des1 = self.orb.detectAndCompute(self.images[i-1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # ic('before filter keypoints', len(des1), len(des2))
        
        # kp1, des1 = self._filter_dynamic_keypoints(kp1, des1, things_mask_prev)
        # kp2, des2 = self._filter_dynamic_keypoints(kp2, des2, things_mask_curr)

        if len(des2) < len(des1):
            des1 = des1[:len(des2)]
        else:
            des2 = des2[:len(des1)]
        # ic(len(des1), len(des2))
        matches = self.flann.knnMatch(des1, des2, k=2)
        # ic(len(matches[0]), len(matches[1]))
        # ic(len(matches))
        
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.5 * n.distance:
                    good.append(m)
                    
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1], kp2, good, None, **draw_params)
        
        ic(f'{i}: {len(good)}')
        cv2.imshow("Matches", img3)
        cv2.waitKey(200)
        
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
        sift = cv2.SIFT_create(10000)

        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = sift.detectAndCompute(self.images[i], None)
        
        # ic(len(kp1), len(kp2))
        # ic(len(des1), len(des2))
        
        kp1_t, kp2_t = kp1, kp2
        des1_t, des2_t = des1, des2
        
        # Optional dynamic keypoint filtering (commented for now)
        # kp1, des1 = self._filter_dynamic_keypoints(kp1, des1, things_mask_prev)
        # kp2, des2 = self._filter_dynamic_keypoints(kp2, des2, things_mask_curr)

        if des1 is None or des2 is None:
            kp1, kp2 = kp1_t, kp2_t
            des1, des2 = des1_t, des2_t
        if len(des2) < len(des1):
            des1 = des1[:len(des2)]
        else:
            des2 = des2[:len(des1)]
            
        # ic('after filter',len(kp1), len(kp2), i)
        # FLANN-based matcher for SIFT
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio test
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # Visualize matches
        draw_params = dict(matchColor=-1,
                        singlePointColor=None,
                        matchesMask=None,
                        flags=2)
        img3 = cv2.drawMatches(self.images[i], kp2, self.images[i - 1], kp1, good, None, **draw_params)
        # cv2.imshow("SIFT Matches", img3)
        # cv2.waitKey(200)

        # Extract good keypoints positions
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2


    def get_pose(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)
        # here we get the rotation and translation from the essential matrix
        ls, tp1, tp2 = self.decomp_essential_mat(E, q1, q2)
        R, t = ls[0], ls[1]
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix, tp1, tp2

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            T = self._form_transf(R, t)
            P_proj = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            # these are the triangulated points
            hom_Q1 = cv2.triangulatePoints(self.P, P_proj, q1.T, q2.T)
            #These same points are transformed into the second camera's coordinate frame using the relative transformation T
            hom_Q2 = np.matmul(T, hom_Q1)
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
            if uhom_Q1.shape[1] < 2 or uhom_Q2.shape[1] < 2:
                print("Not enough valid points for scale computation.")
                relative_scale = 1.0  # Default scale
            else:
                denominator = np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1) + 1e-10
                relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) / denominator)

            # return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale, uhom_Q1, uhom_Q2 # edited to also return the triangulated points

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale, tp1, tp2 = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        return [R1, t], tp1, tp2 #added the triangulated points to the return value

    @staticmethod
    def _pose_matrix_to_params(T):
        R = T[:3, :3]
        t = T[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        return np.hstack((rvec.flatten(), t))
    
    def bundle_adjustment(self, camera_init_params, points_init, observations, num_iters=500, lr=1e-3):
        """
        Performs bundle adjustment with Lie algebra parametrization for camera poses.
        Each camera is represented as a 6D vector: first 3 elements for log-rotation (so(3)) 
        and the last 3 for translation.
        
        Parameters:
          camera_init_params: numpy array of shape (N_cam, 6)
          points_init: numpy array of shape (N_pts, 3)
          observations: list of dict; each entry:
                        { "pt_id": int, "observations": [(frame_idx, [u,v]), ...] }
          num_iters: iterations
          lr: learning rate.
          
        Returns:
          optimized_camera_params, optimized_points
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        K = torch.tensor(self.K, dtype=torch.double, device=device)
        # converting the camera parameters and points to torch tensors
        camera_params = torch.tensor(camera_init_params, dtype=torch.double, device=device, requires_grad=True)
        points = torch.tensor(points_init, dtype=torch.double, device=device, requires_grad=True)
        
        optimizer = optim.Adam([camera_params, points], lr=lr)
        # running over number of iterations 
        for itr in range(num_iters):
            # intial total loss of 0
            total_loss = torch.tensor(0.0, dtype=torch.double, device=device)
            for obs_entry in observations:
                pt_id = obs_entry["pt_id"]
                X = points[pt_id]
                for frame_idx, uv_obs in obs_entry["observations"]:
                    cam_param = camera_params[frame_idx]
                    log_rot = cam_param[:3]
                    t = cam_param[3:]
                    R = so3_exp_map(log_rot.unsqueeze(0))[0]
                    X_cam = R @ X + t
                    x_hom = K @ X_cam
                    x_proj = x_hom[:2] / x_hom[2]
                    uv_obs_t = torch.tensor(uv_obs, dtype=torch.double, device=device)
                    error = x_proj - uv_obs_t
                    total_loss = total_loss + torch.sum(error**2)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if itr % 50 == 0:
                ic(f"BA Iter {itr}/{num_iters}, Loss = {total_loss.item():.6f}")
        optimized_camera_params = camera_params.detach().cpu().numpy()
        optimized_points = points.detach().cpu().numpy()
        return optimized_camera_params, optimized_points

def get_camera_positions_from_poses(poses):
    return np.array([pose[:3, 3] for pose in poses])

def get_camera_positions_from_params(params):
    positions = []
    for cam in params:
        rvec = cam[:3]
        tvec = cam[3:]
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        # Get the camera center (inverse of transformation)
        cam_position = -R.T @ tvec
        positions.append(cam_position)
    return np.array(positions)

def plot_trajectory(estimated_positions, gt_positions=None):
    plt.figure(figsize=(10, 8))
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Path', color='blue')
    if gt_positions is not None:
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth Path', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Estimated Trajectory')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import cv2

def pose_vector_to_matrix(pose_vec):
    """Converts a 6D pose vector to a 4x4 transformation matrix."""
    rvec = pose_vec[:3]
    tvec = pose_vec[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def plot_optimized_camera_trajectory(optimized_camera_params, estimated_path):
    """
    Plots camera trajectory from optimized 6D pose parameters.
    """
    positions = []
    for cam_param in optimized_camera_params:
        T = pose_vector_to_matrix(cam_param.cpu().numpy() if hasattr(cam_param, 'cpu') else cam_param)
        cam_pos = T[:3, 3]
        positions.append([cam_pos[0], cam_pos[2]])  # x-z plane

    positions = np.array(positions)
    plt.figure(figsize=(10, 5))
    plt.plot(positions[:, 0], positions[:, 1], '-o', label="Optimized Camera Trajectory")
    plt.plot(estimated_path[:, 0], estimated_path[:, 1], label='Estimated Path', color='blue')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Optimized Camera Trajectory")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()

def main(data_dir):
    
    
    slam = PanopticSLAM(data_dir)
    num_frames = len(slam.images)
    estimated_path = []
    all_observations = []
    triangulated_points = []
    point_id_counter = 0
    cur_pose = np.eye(4, dtype=np.float64)

    # Start with identity pose for frame 0
    camera_init_params = [PanopticSLAM._pose_matrix_to_params(np.eye(4))]

    for i in tqdm(range(1, num_frames), unit="frames", desc="Processing frames"):
        q1, q2 = slam.get_matches(i)
        if len(q1) >= 8:
            transf, tp1, tp2 = slam.get_pose(q1, q2)
            if transf is not None:
                cur_pose = cur_pose @ transf

                estimated_path.append(cur_pose.copy())

                camera_init_params.append(PanopticSLAM._pose_matrix_to_params(cur_pose))

                # Transform triangulated points from camera frame (i-1) to world frame
                prev_pose = estimated_path[-2] if len(estimated_path) > 1 else np.eye(4)
                world_points = (prev_pose @ np.vstack((tp1, np.ones((1, tp1.shape[1])))))[:3, :]

                for j, (pt3d, uv1, uv2) in enumerate(zip(world_points.T, q1, q2)):
                    pt_id = point_id_counter
                    triangulated_points.append(pt3d)
                    point_id_counter += 1

                    obs = {"pt_id": pt_id, "observations": []}
                    obs["observations"].append((i - 1, uv1.tolist()))
                    obs["observations"].append((i, uv2.tolist()))
                    all_observations.append(obs)
            else:
                print(f"[Frame {i}] Transformation is None.")
        else:
            print(f"[Frame {i}] Not enough matches ({len(q1)}). Skipping.")

    # if len(estimated_path) == 0:
    #     print("No valid poses estimated. Exiting.")
    #     return

    # triangulated_points = np.array(triangulated_points, dtype=np.float32)
    # camera_init_params = np.array(camera_init_params, dtype=np.float32)

    # print(f"Running BA on {len(camera_init_params)} cameras and {len(triangulated_points)} points")


    
    # # Convert all_observations to tensor-compatible format
    # obs_tensor_data = []
    # for obs in all_observations:
    #     pt_idx = obs["pt_id"]
    #     for cam_idx, uv in obs["observations"]:
    #         obs_tensor_data.append([cam_idx, pt_idx, uv[0], uv[1]])

    # obs_tensor_data = np.array(obs_tensor_data, dtype=np.float32)
    
    # BA = BundleAdjustment(camera_init_params, triangulated_points, obs_tensor_data,slam.K )
    # optimized_cams, optimized_points, feature_weights = BA.optimize()


    # np.save("points",np.array(triangulated_points))
    # # Trajectory visualization
    estimated_positions = np.array([[pose[0, 3], pose[2, 3]] for pose in estimated_path])
    plot_trajectory(estimated_positions)

    # plot_optimized_camera_trajectory(optimized_cams, estimated_positions)
    # return optimized_cams, optimized_points
    



if __name__ == "__main__":
    data_dir = "/home/mahesh/visual_odometry/test-1"  
    main(data_dir)
