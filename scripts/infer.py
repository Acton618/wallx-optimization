      
import os, sys
device='cuda'

import torch
import numpy as np
import yaml
import time
from tqdm import tqdm
import dataclasses
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor
import re
from PIL import Image
import threading
import copy

from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR
from qwen_vl_utils import process_vision_info

from qwen_vl_utils.vision_process import smart_resize
from transformers import BatchFeature
import cv2
import json
#from wall_x.data.utils import load_norm_stats
#from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
#from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from wall_x.data.backends.lerobot.utils import load_norm_stats
from wall_x.model.qact.qwen2_5 import Qwen2_5_VLMoEForAction
from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from robot_controller import RobotController

def move_to_cuda(obj, device=device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (dict, BatchFeature)):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj

def normalize(action, min_stat, delta):
    """
    Normalize action data using min-max normalization.
    """
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    x = (action - min_stat) / delta
    x = x * 2 - 1
    x = torch.clamp(x, -1, 1)
    return x


action_key_mapping = {
    "follow_left_ee_cartesian_pos": "follow1_pos[:3]",
    "follow_left_ee_rotation": "follow1_pos[3:6]",
    "follow_left_gripper": "follow1_pos[6:7]",
    "follow_right_ee_cartesian_pos": "follow2_pos[:3]",
    "follow_right_ee_rotation": "follow2_pos[3:6]",
    "follow_right_gripper": "follow2_pos[6:7]",
    "head_actions": "head_pos",
    "height": "lift",
    "velocity_decomposed": "velocity_decomposed",
    "follow_left_arm_joint_cur": "follow1_joints_cur[-1:]",
    "follow_right_arm_joint_cur": "follow2_joints_cur[-1:]",
    "follow_left_arm_joint_pos":"follow1_joints",
    "follow_right_arm_joint_pos":"follow2_joints",
}


_CAM_NAME_MAPPING = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
    "move1_view": "move view",
    "move2_view": "move view",
    "wall_view": "wall view",
    "top_view": "top view",
}

class VehiclePoseHandler:
    """车辆位姿和速度计算"""

    def __init__(self):
        self.current_pose = None
        self.previous_pose = None
        self.pose_history = deque(maxlen=10)

    def update_pose(self, new_pose):
        """更新车辆位姿"""
        if new_pose is not None:
            self.previous_pose = self.current_pose
            self.current_pose = np.array(new_pose)
            self.pose_history.append(self.current_pose.copy())
            print("current_pose", self.current_pose, flush=True)
        return self.current_pose

    def velocity_to_pose(self, vx_body, vy_body, vyaw, dt, start_pose=None):
        """将本体坐标系速度转换为全局坐标系位置"""
        if start_pose is None:
            if self.current_pose is not None:
                start_pose = self.current_pose.copy()
            else:
                start_pose = np.array([0.0, 0.0, 0.0])

        x, y, theta = start_pose

        # 本体坐标系速度转换为全局坐标系位移
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 坐标变换：本体坐标系 -> 全局坐标系
        dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
        dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
        dtheta = vyaw * dt

        # 计算新位置
        x_new = x + dx_global
        y_new = y + dy_global
        theta_new = theta + dtheta

        # 将角度限制在[-pi, pi]范围内
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, theta_new])

    def compute_body_velocities_from_poses(
        self, current_pose, previous_pose, dt=1 / 20
    ):
        """从位姿变化计算本体坐标系速度"""
        if current_pose is None or previous_pose is None:
            return np.array([0.0, 0.0, 0.0])

        # 计算全局坐标系下的位移
        dx_global = current_pose[0] - previous_pose[0]
        dy_global = current_pose[1] - previous_pose[1]
        dtheta = current_pose[2] - previous_pose[2]

        # 使用前一帧的角度进行坐标变换
        theta = previous_pose[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 全局坐标系位移转换为本体坐标系速度
        vx_body = (dx_global * cos_theta + dy_global * sin_theta) / dt
        vy_body = (-dx_global * sin_theta + dy_global * cos_theta) / dt
        vyaw = dtheta / dt

        return np.array([vx_body, vy_body, vyaw])


@dataclasses.dataclass
class WallxInferArgs:
    model_path: str = '/opt/data/zhuoziying/wall-x-model'
    config_path: str = '/opt/data/zhuoziying/wall-x/workspace/example/infer_config.yml'
    action_mode: str = "diffusion" # "ar" or "diffusion"
    instruction: str = "Pick up the cup and put it in the plate"
    repo_id: str = "/opt/data/zhuoziying/dataset/aloha_sim_transfer_cube_human"
    action_start_ratio: float = 0
    action_end_ratio: float = 0.2   # 有效推理步数，注意这一个部分
    action_dim: int = 20 # 目前只支持14(双臂动作) 或 20(全身动作)
    action_horizon: int = 32
    interpolate_action: bool = True
    interpolate_multiplier: int = 32


class WallxModelWrapper:
    def __init__(self, args: WallxInferArgs):
        self.args = args
        self.load_config(self.args.config_path)
        self.get_normalize_stats()

        self.action_tokenizer_path = None            
        self.obs_action_keys = ["follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper", 
                                "follow_right_ee_cartesian_pos","follow_right_ee_rotation","follow_right_gripper"]
        self.action_predict_mode = self.args.action_mode
        self.get_model_and_processor()
        self.vehicle_pose_handler = VehiclePoseHandler()  ##作为全身机器人/移动机器人中底盘相关的运动管理
        self.last_speed = [0, 0, 0]                       ##考虑到会将返回的动作做pad，目前并没有发挥作用


    def get_normalize_stats(self):                        ##根据前面数据集统计中获得的内容，采集对应的归一化用的信息
        norm_stats_path = self.config["norm_stats_path"]
        norm_stats = load_norm_stats(norm_stats_path,self.args.repo_id)
        self.action_min = norm_stats["action"].min
        self.action_delta = norm_stats["action"].delta
        self.state_min = norm_stats["state"].min
        self.state_delta = norm_stats["state"].delta
        print("action_min", self.action_min,flush=True)
        print("action_delta", self.action_delta,flush=True)
        print("state_min", self.state_min,flush=True)
        print("state_delta", self.state_delta,flush=True)


    def load_config(self, config_path):                   ##config中model-type必须放在第一行，供其获得("model_type")
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config["data"]["model_type"] = config.get("model_type")

        self.config = config

    def get_model_and_processor(self):
        
        # load model with customized robot config
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            self.args.model_path, train_config=self.config, action_tokenizer_path=self.action_tokenizer_path
        )
        model.eval()
        model = model.to(device)
        model = model.bfloat16()
        self.model = model
        #print(model.action_preprocessor.normalizer_action.min.x2,flush=True)
        #print(model.action_preprocessor.normalizer_action.delta.x2,flush=True)

        # self.normalizer_propri = copy.deepcopy(model.action_preprocessor.normalizer_propri).to("cpu")
        # print("normalizer_propri", self.normalizer_propri,flush=True)

        processor_path = self.config["pretrained_wallx_path"]
        action_tokenizer_path = self.config["action_tokenizer_path"]

        processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
        if self.config.get("padding_side", "left") == "left":
            processor.tokenizer.padding_side = "left"
        
        new_tokens = ["<|propri|>", "<|action|>"]
        processor.tokenizer.add_tokens(new_tokens)
        self.processor = processor

    def get_car_pose(self, base_velocity_pred):
        # 向量化积分速度为位置
        dt = 1 / 20
        current_pose = (
            self.vehicle_pose_handler.current_pose.copy()
            if self.vehicle_pose_handler.current_pose is not None
            else np.array([0.0, 0.0, 0.0])
        )

        # 批量积分计算位置
        poses_frames = []
        for i in range(len(base_velocity_pred)):
            current_pose = self.vehicle_pose_handler.velocity_to_pose(
                base_velocity_pred[i, 0],
                base_velocity_pred[i, 1],
                base_velocity_pred[i, 2],
                dt,
                current_pose,
            )
            poses_frames.append(current_pose.copy())

        return poses_frames

    def preprocess(self, state, views):
        # state: dict, keys: 
        #   follow1_pos, follow1_joints, follow2_pos, follow2_joints, follow1_joint_cur, follow2_joint_cur
        #   (optional) head_pos, lift, car_pose
        if "car_pose" in state:
            self.vehicle_pose_handler.update_pose(state["car_pose"])
            state["velocity_decomposed"] = self.last_speed

        agent_data = []
        agent_data_mask = []
        for key in self.obs_action_keys:
            state_key_str = action_key_mapping[key]
            value = None
            
            match = re.match(r'(\w+)\[(.*)\]', state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(':')
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = state[base_key][s]
            else:
                if state_key_str in state:
                    value = state[state_key_str]

            if value is not None:
                print("agent_data key", key, "value", value,flush=True)
                if isinstance(value, float) or isinstance(value, int):
                    value = [value]
                agent_data.append(np.asarray(value))
                agent_data_mask.append(np.ones_like(value))
            else:
                print("agent_data key", key, "value is None",flush=True)
                agent_data.append(np.zeros((dof_config[key],)))
                agent_data_mask.append(np.zeros((dof_config[key],)))
        print("agent_data", agent_data,flush=True)
        agent_data = np.concatenate(agent_data, axis=0)[None, None] # (1, 1, D)
        agent_pos_mask = np.concatenate(agent_data_mask, axis=0)[None, None] # (1, 1, D)
        dof_mask = agent_pos_mask.repeat(self.args.action_horizon, axis=1) # (1, H, D)
        print("agent_data", agent_data.shape,flush=True)
        print("agent_pos_mask", agent_pos_mask.shape,flush=True)
        print("dof_mask", dof_mask.shape,flush=True)

        observation = {
            "face_view":views['camera_front'][0], # (1, 480, 640, 3)
            "left_wrist_view":views['camera_left'][0], # (1, 480, 640, 3)
            "right_wrist_view":views['camera_right'][0], # (1, 480, 640, 3)
            "agent_pos":agent_data, # (1, 1, 14)
            "dof_mask":dof_mask,
            "agent_pos_mask":agent_pos_mask,
        }

        return observation

    def resize_images(self, observation):
        image_inputs = []
        for key in ["face_view", "left_wrist_view", "right_wrist_view"]:
            if key not in observation:
                continue
            # 1. 获取原始图像
            # current_obs = observation[key].permute(1, 2, 0)
            current_obs = observation[key]
            img_pil = Image.fromarray(current_obs)
            orig_width, orig_height = img_pil.size

            # 2. 应用分辨率限制（如果配置不为-1）
            # print(f"obs_data key: {key}, resolution dict: {data_config.resolution}")
            # target_size = data_config.resolution.get(cam_mapping[key], -1)
            target_size = 256
            if target_size != -1:
                # 保持宽高比的限制逻辑
                if orig_width > orig_height:  # 横向图像
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # 纵向图像
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. 应用智能缩放（qwen逻辑）
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            resized_img = img_pil.resize((resized_width, resized_height))

            image_inputs.append(resized_img)

        return image_inputs

    def get_text_flow(self, instruction, action_chunk_size, norm_state=None):
        role_start_symbol = "<|im_start|>"
        role_end_symbol = "<|im_end|>"
        vision_start_symbol = "<|vision_start|>"
        vision_end_symbol = "<|vision_end|>"
        image_pad_symbol = "<|image_pad|>"
        propri_symbol = "<|propri|>"
        action_symbol = "<|action|>"

        prologue = f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
        user_request = f"{role_start_symbol}user\nObservation:"
        for cam_name in ["face_view", "left_wrist_view", "right_wrist_view"]:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
        user_request += "\nInstruction:"
        propri = propri_symbol # 填占位符
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {propri}\n"
        user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
        assistant_message = f"{role_start_symbol}assistant\n"
        action = f"{action_symbol * action_chunk_size}"
        text = prologue + user_message + assistant_message + action
        print("text", text,flush=True)

        return text

    def _construct_input(self, observation, instruction):
        additional_inputs = {}

        agent_pos = torch.from_numpy(observation["agent_pos"])
        agent_pos_mask = torch.from_numpy(observation["agent_pos_mask"])
        # additional_inputs["dof_mask"] = torch.from_numpy(observation["dof_mask"])
        dof_mask = torch.from_numpy(observation["dof_mask"])
        agent_pos = normalize(agent_pos, self.state_min, self.state_delta)
        if agent_pos.shape[-1] != 20:
            agent_pos = torch.cat(
                [
                    agent_pos,
                    torch.zeros(
                        agent_pos.shape[0],
                        agent_pos.shape[1],
                        20 - agent_pos.shape[-1],
                    ),
                ],
                dim=-1,
            )
            agent_pos_mask = torch.cat(
                [
                    agent_pos_mask,
                    torch.zeros(
                        agent_pos_mask.shape[0],
                        agent_pos_mask.shape[1],
                        20 - agent_pos_mask.shape[-1],
                    ),
                ],
                dim=-1,
            )
            dof_mask = torch.cat(
                [
                    dof_mask,
                    torch.zeros(
                        dof_mask.shape[0],
                        dof_mask.shape[1],
                        20 - dof_mask.shape[-1],
                    ),
                ],
                dim=-1,
            )
        print("agent_pos", agent_pos,flush=True)
        print("agent_pos_mask", agent_pos_mask,flush=True)
        print("dof_mask", dof_mask,flush=True)
        additional_inputs["proprioception"] = agent_pos
        additional_inputs["agent_pos_mask"] = agent_pos_mask
        additional_inputs["dof_mask"] = dof_mask
        text = self.get_text_flow(instruction, self.args.action_horizon, agent_pos)
        text = [text]

        image_inputs = self.resize_images(observation)
        image_inputs = self.processor.image_processor(images=image_inputs, videos=None, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        # 处理文本中的图像占位符token
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    # 替换图像占位符为实际数量
                    text[i] = text[i].replace("<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        text_inputs = self.processor.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = BatchFeature(data={**text_inputs, **image_inputs})

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        additional_inputs["moe_token_types"] = (inputs.input_ids == action_token_id)
        additional_inputs["dataset_names"] = ["x2"]
        # 本体感觉数据 (agent_pos) + 掩码
        inputs.update(additional_inputs)
        inputs = move_to_cuda(inputs,device)
        return inputs

    def interpolate_trajectory(self, action_pred, target_length):
        interpolated = np.zeros((target_length, action_pred.shape[1]))
        original_indices = np.linspace(0, len(action_pred) - 1, len(action_pred))
        target_indices = np.linspace(0, len(action_pred) - 1, target_length)
        for i in range(action_pred.shape[1]):
            interpolated[:, i] = np.interp(target_indices, original_indices, action_pred[:, i])
        return interpolated

    def postprocess(self, action_pred, interpolate_multiplier=None):
        if interpolate_multiplier is None:
            interpolate_multiplier = self.args.interpolate_multiplier

        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.to(torch.float32).cpu().squeeze(0).numpy()
        
        if action_pred.shape[1] == 20:
            ## 处理 预测速度->世界坐标位置 转换
            last_car_pose = (
                self.vehicle_pose_handler.current_pose.copy()
                if self.vehicle_pose_handler.current_pose is not None
                else np.array([0.0, 0.0, 0.0])
            )
            self.last_speed = action_pred[-1, 17:20].copy().tolist()
            car_pose = np.array(
                self.get_car_pose(action_pred[:, 17:20])
            )  # (action_horizon, 3)
            action_pred[:, 17:20] = car_pose.tolist()

        start_frame = int(self.args.action_start_ratio * len(action_pred))
        end_frame = int(self.args.action_end_ratio * len(action_pred))
        action_pred = action_pred[start_frame:end_frame]

        target_length = interpolate_multiplier * len(action_pred)
        action_pred = self.interpolate_trajectory(
            action_pred, target_length
        )
        left_action_pred = action_pred[:,:7]
        right_action_pred = action_pred[:,7:14]
        head_pose = action_pred[:, 14:16]  # (32, 2)
        height = action_pred[:, 16:17]  # (32, 1)
        car_pose = action_pred[:, 17:20]  # (32, 3)

        print("left_action_pred", left_action_pred.shape, left_action_pred[-1],flush=True)
        print("right_action_pred", right_action_pred.shape, right_action_pred[-1],flush=True)


        left_action_pred = left_action_pred.tolist()
        right_action_pred = right_action_pred.tolist()

        
        # hard code for now
        if action_pred.shape[1] == 14:
            head_pose = [[-1,0] for i in range(len(left_action_pred))]
            height = [0.4 for i in range(len(left_action_pred))]
            car_pose = [last_car_pose.tolist() for i in range(len(left_action_pred))]
        elif action_pred.shape[1] == 20:
            print("head_pose", head_pose.shape,head_pose[-1],flush=True)
            print("height", height.shape,height[-1],flush=True)
            print("car_pose", car_pose.shape,car_pose[-1],flush=True)
            head_pose = head_pose.tolist()
            height = height.tolist()
            car_pose = car_pose.tolist()
        else:
            raise ValueError(f"action_pred.shape[1] is not 14 or 20: {action_pred.shape[1]}")

        serialized_actions = {
            "follow1_pos":left_action_pred,
            "follow2_pos":right_action_pred,
            "head_pos": head_pose,
            "lift": height,
            "car_pose": car_pose
        }
        
        return serialized_actions

    @torch.no_grad()
    def predict_action(self, state, views, instruction):
        observation = self.preprocess(state, views)
        inputs = self._construct_input(observation, instruction)
        while True:
            outputs = self.model(
                **inputs, 
                mode="predict", 
                action_dim=self.args.action_dim,
                pred_horizon=self.args.action_horizon, 
                predict_mode=self.action_predict_mode, 
                num_inference_timesteps=10
            )
            action_pred = outputs["predict_action"].detach().cpu()
            print("action_pred", action_pred.shape,flush=True)
            if action_pred is not None:
                break        
        action_pred = self.postprocess(action_pred)
        return action_pred

class WallxInfer:
    def __init__(self, args: WallxInferArgs):
        self.args = args
        self.instruction = args.instruction
        self.model_wrapper = WallxModelWrapper(args)
        self.robot_controller = RobotController(
                    robot_id = 10,
                    host='0.0.0.0',
                    port=30123
                )
        self.robot_controller.connect()
        
    def _recieve_from_robot(self):
        state = self.robot_controller.recv_action()    
        views = self.robot_controller.recv_image(['camera_left', 'camera_front', 'camera_right'])
        return state, views

    def run_infer(self):
        while True:
            state, views = self._recieve_from_robot()
            print("state", state,flush=True)
            t0=time.time()
            action_pred = self.model_wrapper.predict_action(state=state, views=views, instruction=self.instruction)
            t1=time.time()
            print(f"time cost in predict action: {t1-t0}s",flush=True)
            self.robot_controller.robot_comm.send_dict(action_pred)

        self.robot_controller.close()

    
if __name__ == "__main__":
    args = WallxInferArgs()
    Infer = WallxInfer(args)
    Infer.run_infer()
    

    