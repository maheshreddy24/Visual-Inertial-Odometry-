# import torch
# import torch.optim as optim
# from icecream import ic

# def so3_exp_map(log_rot):
#     """
#     Convert axis-angle (log_rot) to rotation matrix using Rodrigues' formula.
#     log_rot: (N, 3)
#     Returns: (N, 3, 3) rotation matrices
#     """
#     theta = torch.norm(log_rot, dim=1, keepdim=True) + 1e-8
#     axis = log_rot / theta
#     K = torch.zeros((log_rot.shape[0], 3, 3), device=log_rot.device)
#     K[:, 0, 1] = -axis[:, 2]
#     K[:, 0, 2] = axis[:, 1]
#     K[:, 1, 0] = axis[:, 2]
#     K[:, 1, 2] = -axis[:, 0]
#     K[:, 2, 0] = -axis[:, 1]
#     K[:, 2, 1] = axis[:, 0]

#     I = torch.eye(3, device=log_rot.device).unsqueeze(0)
#     R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
#     return R

# def project_points(camera_params, points, camera_indices, point_indices, K):
#     log_rot = camera_params[camera_indices, :3]
#     t = camera_params[camera_indices, 3:]
#     R = so3_exp_map(log_rot)  # (N_obs, 3, 3)
#     X = points[point_indices]  # (N_obs, 3)

#     # Transform points to camera frame
#     X_cam = torch.bmm(R, X.unsqueeze(-1)).squeeze(-1) + t

#     # Project to image plane
#     x_hom = torch.bmm(K.expand(len(X_cam), -1, -1), X_cam.unsqueeze(-1))
#     x_proj = x_hom.squeeze(-1)[:, :2] / x_hom.squeeze(-1)[:, 2:]
#     return x_proj

# def huber_loss(errors, delta=1.0):
#     abs_errors = errors.abs()
#     quadratic = torch.minimum(abs_errors, torch.tensor(delta, device=errors.device))
#     linear = abs_errors - quadratic
#     return 0.5 * quadratic ** 2 + delta * linear

# def compute_reprojection_error(camera_params, points, obs_data, K):
#     camera_indices = obs_data[:, 0].long()
#     point_indices = obs_data[:, 1].long()
#     uv_obs = obs_data[:, 2:4]

#     uv_proj = project_points(camera_params, points, camera_indices, point_indices, K)
#     errors = uv_proj - uv_obs
#     return huber_loss(errors).sum()

# def bundle_adjustment_lm(camera_params, points, obs_data, K, num_iters=20):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     camera_params = torch.tensor(camera_params, dtype=torch.float32, device=device, requires_grad=True)
#     points = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
#     obs_data = torch.tensor(obs_data, dtype=torch.float32, device=device)
#     K = torch.tensor(K, dtype=torch.float32, device=device)
#     camera_params = camera_params.to(device)
#     points = points.to(device)
#     obs_data = obs_data.to(device)
#     K = K.to(device)
#     camera_params.requires_grad = True
#     points.requires_grad = True
#     optimizer = optim.LBFGS([camera_params, points], lr=1e-3, max_iter=num_iters)

#     def closure():
#         optimizer.zero_grad()
#         loss = compute_reprojection_error(camera_params, points, obs_data, K)
#         loss.backward()
#         return loss

#     optimizer.step(closure)
#     return camera_params.detach().cpu(), points.detach().cpu()


# def bundle_adjustment_with_weights(camera_params, points, obs_data, K, num_iters=10, lambda_w=1.0):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # tensors for all variables
#     camera_params = torch.tensor(camera_params, dtype=torch.float32, device=device, requires_grad=True)
#     points = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
#     obs_data = torch.tensor(obs_data, dtype=torch.float32, device=device)
#     K = torch.tensor(K, dtype=torch.float32, device=device)

#     #total points
#     num_points = points.shape[0]
#     # weights for each poitns (when they reprojected if the error is high weights are high)
#     weights = torch.ones(num_points, device=device, requires_grad=False)

#     for iter in range(num_iters):
#         ic(iter)
#         camera_params.requires_grad_(True)
#         points.requires_grad_(True)

#         def closure():
#             optimizer.zero_grad()
#             # Extract indices
#             camera_indices = obs_data[:, 0].long()
#             point_indices = obs_data[:, 1].long()
#             uv_obs = obs_data[:, 2:4]

#             # this projects points to the camera frame 
#             uv_proj = project_points(camera_params, points, camera_indices, point_indices, K)
#             # difference between the projected points and the observed points
#             residuals = uv_proj - uv_obs

#             # Apply per-feature weights (DynaVINS-style)
#             weight_per_obs = weights[point_indices].sqrt().unsqueeze(1)
#             weighted_residuals = residuals * weight_per_obs

#             loss = huber_loss(weighted_residuals).sum()
#             loss.backward()
#             return loss

#         optimizer = optim.LBFGS([camera_params, points], lr=1e-3, max_iter=10)
#         optimizer.step(closure)

#         # --- Update Weights ---
#         with torch.no_grad():
#             camera_indices = obs_data[:, 0].long()
#             point_indices = obs_data[:, 1].long()
#             uv_obs = obs_data[:, 2:4]

#             uv_proj = project_points(camera_params, points, camera_indices, point_indices, K)
#             errors = uv_proj - uv_obs
#             squared_errors = errors.pow(2).sum(dim=1)  # per observation

#             # Accumulate reprojection error per point
#             error_sum = torch.zeros(num_points, device=device)
#             count = torch.zeros(num_points, device=device)
#             for i in range(obs_data.shape[0]):
#                 idx = point_indices[i]
#                 error_sum[idx] += squared_errors[i]
#                 count[idx] += 1
#             count = torch.clamp(count, min=1)
#             avg_error = error_sum / count

#             weights = lambda_w / (avg_error + lambda_w)
#             weights = torch.clamp(weights, 0.0, 1.0)

#     return camera_params.detach().cpu(), points.detach().cpu(), weights.detach().cpu()


import torch
import torch.nn as nn
from pytorch3d.transforms.so3 import so3_exp_map
from pytorch3d.transforms import axis_angle_to_matrix
import torch.optim as optim
from icecream import ic
import matplotlib.pyplot as plt

class BundleAdjustment:
    def __init__(self, camera_init, points, obs_data, K, ITERS=10, lambda_w=1.0, lambda_m=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        
        # Convert to tensors and enable gradients for optimization variables
        self.camera_params = torch.tensor(camera_init, dtype=torch.float32, device=self.device, requires_grad=True)
        self.points_3d = torch.tensor(points, dtype=torch.float32, device=self.device, requires_grad=True)
        self.obs_data = torch.tensor(obs_data, dtype=torch.float32, device=self.device)
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)
        
        self.ITERS = ITERS
        self.lambda_w = lambda_w
        self.lambda_m = lambda_m  # Momentum factor from paper
        
        # Initialize weights and previous weights
        self.num_points = self.points_3d.shape[0]
        self.weights = torch.ones(self.num_points, device=self.device, requires_grad=False)
        self.weights_prev = torch.ones_like(self.weights)
        
        # Optimizer only for camera and point parameters
        self.optimizer = optim.LBFGS([self.camera_params, self.points_3d], lr=1e-3, max_iter=10)


    def project_points(self, camera_params, points_3d, camera_indices, point_indices, K):
        # Extract rotation (Lie algebra) and translation
        log_rot = camera_params[camera_indices, :3]  # (N_obs, 3)
        t = camera_params[camera_indices, 3:]         # (N_obs, 3)

        # Convert Lie algebra to rotation matrices via exponential map
        R = so3_exp_map(log_rot)  # Correct conversion for Lie algebra parameters

        # Get corresponding 3D points
        X = points_3d[point_indices]  # (N_obs, 3)

        # Transform to camera coordinates
        X_cam = torch.bmm(R, X.unsqueeze(-1)).squeeze(-1) + t  # (N_obs, 3)

        # Project to image plane (remainder unchanged)
        K_batch = K.unsqueeze(0).expand(X_cam.shape[0], -1, -1)  # (N_obs, 3, 3)
        x_proj_hom = torch.bmm(K_batch, X_cam.unsqueeze(-1)).squeeze(-1)
        x_proj = x_proj_hom[:, :2] / x_proj_hom[:, 2].clamp(min=1e-6).unsqueeze(-1)
        
        return x_proj
    
    def compute_loss(self):
        camera_indices = self.obs_data[:, 0].long()
        point_indices = self.obs_data[:, 1].long()
        uv_obs = self.obs_data[:, 2:4]  # (x,y) coordinates
        
        # Project points
        uv_proj = self.project_points(
            self.camera_params, 
            self.points_3d, 
            camera_indices, 
            point_indices, 
            self.K
        )
        
        # Reprojection error
        residuals = uv_proj - uv_obs
        sq_errors = residuals.pow(2).sum(dim=1)  # (N_obs,)
        
        error_sum = torch.zeros_like(self.weights)
        error_sum = error_sum.index_add(0, point_indices, sq_errors)

        count = torch.zeros_like(self.weights)
        ones = torch.ones_like(sq_errors)
        count = count.index_add(0, point_indices, ones)
        count = count.clamp(min=1)

        
        # Weight update from paper Eq.7
        # ic(error_sum)
        with torch.no_grad():
            eps = 1e-8
            new_weights = self.lambda_w / (error_sum + self.lambda_w + eps)
            # Apply momentum from Eq.10
            new_weights = (new_weights + self.lambda_m * count * self.weights_prev) / (1 + self.lambda_m * count)
            new_weights = torch.clamp(new_weights, 0.0, 1.0)
            # percentage_above_0_5 = (new_weights > 0.1).float().mean().item() * 100
            # print(f"Percentage of weights > 0.5: {percentage_above_0_5:.2f}%")
            self.weights_prev = self.weights.clone()
            self.weights = new_weights
        
        # Get weights for each observation
        weights_per_obs = self.weights[point_indices]
        
        # Total loss: sum(w_j * r_p^ji^2) + lambda_w * (1 - w_j)^2 +   term
        visual_loss = (weights_per_obs * sq_errors).sum()
        reg_loss = self.lambda_w * ((1 - self.weights)**2).sum()
        momentum_loss = self.lambda_m * (count * (self.weights - self.weights_prev)**2).sum()
        
        total_loss = visual_loss + reg_loss + momentum_loss
        return total_loss

    # def optimize(self):
    #     loss_list = []
    #     for iter in range(self.ITERS):
    #         ic(f"Iteration {iter}")
            
    #         def closure():
    #             self.optimizer.zero_grad()
    #             loss = self.compute_loss()
    #             loss.backward()
    #             return loss
            
    #         self.optimizer.step(closure)
        
    #     # Return optimized parameters
    #     return self.camera_params.detach().cpu(), self.points_3d.detach().cpu(), self.weights.detach().cpu()
    

    def optimize(self):
        loss_list = []
        for iter in range(self.ITERS):
            ic(f"Iteration {iter}")
            
            def closure():
                self.optimizer.zero_grad()
                loss = self.compute_loss()
                loss.backward()
                return loss
            
            loss_val = self.optimizer.step(closure)
            loss_list.append(loss_val.item())
        
        # Plot the loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(loss_list, marker='o', linestyle='-')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Convergence")
        plt.grid(True)
        plt.show()
        
        # Return optimized parameters
        return self.camera_params.detach().cpu(), self.points_3d.detach().cpu(), self.weights.detach().cpu()
