import gc
import glob
import os
import pickle
import sys
from copy import deepcopy
from datetime import datetime
sys.path.append("/kaiming-fast-vol-1/workspace/hand_teleop")

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from transforms3d.quaternions import quat2axangle

from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from hand_teleop.player.player import DcLawEnvPlayer, PickPlaceEnvPlayer
from hand_teleop.real_world import lab
from main.eval import apply_IK_get_real_action
from model import Agent


class Trainer:

    def __init__(self, args):
        self.args = args

        self.epoch_start = 0

        self.demos_train, self.demos_val = self.load_data(args)

        self.model = Agent(args).to(args.device)

        self.criterion = nn.MSELoss()
        if args.finetune_backbone:
            self.optimizer = optim.AdamW(self.model.parameters(),
                args.lr, weight_decay=args.wd_coef)
        else:
            self.optimizer = optim.AdamW(self.model.policy_net.parameters(),
                args.lr, weight_decay=args.wd_coef)

        self.start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.args.ckpt is None:
            self.log_dir = f"logs/{self.args.task}_{self.args.backbone}_{self.start_time}"
            os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = os.path.dirname(self.args.ckpt)

        if not args.wandb_off:
            wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
            wandb.login(key=wandb_cfg.key)
            wandb.init(
                project="hand-teleop",
                name=os.path.basename(self.log_dir),
                config=self.args
            )

    def load_data(self, args):
        demo_paths = sorted(glob.glob(os.path.join(args.demo_folder,
            "*.pickle")))
        demos = []
        for path in demo_paths:
            with open(path, "rb") as f:
                cur_demo = pickle.load(f)
                cur_demo["data"] = [{k: cur_demo["data"][k][i] for k in cur_demo["data"].keys()}
                    for i in range(len(cur_demo["data"]["simulation"]))]
                # TODO: fake data, fix it later
                for i in range(len(cur_demo["data"])):
                    cur_demo["data"][i]["action"] = np.random.random(size=(22, )).astype(np.float32)
                demos.append(cur_demo)
        train_idx, val_idx = train_test_split(list(range(len(demos))),
            test_size=args.val_pct, random_state=args.seed)
        demos_train = [demos[i] for i in train_idx]
        demos_val = [demos[i] for i in val_idx]

        return demos_train, demos_val

    def init_player(self, demo):
        meta_data = deepcopy(demo["meta_data"])
        # task_name = meta_data["env_kwargs"]["task_name"]
        # meta_data["env_kwargs"].pop("task_name")
        # meta_data["task_name"] = self.args.task
        robot_name = self.args.robot
        data = demo["data"]
        use_visual_obs = True
        if "finger_control_params" in meta_data.keys():
            finger_control_params = meta_data["finger_control_params"]
        if "root_rotation_control_params" in meta_data.keys():
            root_rotation_control_params = meta_data["root_rotation_control_params"]
        if "root_translation_control_params" in meta_data.keys():
            root_translation_control_params = meta_data["root_translation_control_params"]
        if "robot_arm_control_params" in meta_data.keys():
            robot_arm_control_params = meta_data["robot_arm_control_params"]            

        # Create env
        env_params = meta_data["env_kwargs"]
        if "task_name" in env_params:
            env_params.pop("task_name")
        env_params["robot_name"] = robot_name
        env_params["use_visual_obs"] = use_visual_obs
        env_params["use_gui"] = False

        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"

        if robot_name == "mano":
            env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
        else:
            env_params["zero_joint_pos"] = None

        if "init_obj_pos" in meta_data["env_kwargs"].keys():
            env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]

        if "init_target_pos" in meta_data["env_kwargs"].keys():
            env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]

        if self.args.task == "pick_place":
            env = PickPlaceRLEnv(**env_params)
        elif self.args.task == "dclaw":
            env = DClawRLEnv(**env_params)
        else:
            raise NotImplementedError

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        env.rl_step = env.simple_sim_step

        env.reset()

        real_camera_cfg = {
            "relocate_view": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT,
            fov=lab.fov, resolution=(224, 224))
        }
        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info}}
        env.setup_visual_obs_config(camera_info)

        # Player
        if self.args.task == "pick_place":
            player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        elif self.args.task == "dclaw":
            player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        else:
            raise NotImplementedError

        return player

    def replay_demo(self, player):
        demo_len = len(player.data)
        robot_pose = player.env.robot.get_pose()
        arm_dof = player.env.arm_dof

        player.scene.unpack(player.get_sim_data(0))
        for _ in range(player.env.frame_skip):
            player.scene.step()
        palm_pose = player.env.ee_link.get_pose()
        palm_pose = robot_pose.inv() * palm_pose
        hand_qpos = player.env.robot.get_drive_target()[arm_dof:]
        actions = []

        for i in range(1, demo_len):
            player.scene.unpack(player.get_sim_data(i))

            palm_pose_next = player.env.ee_link.get_pose()
            palm_pose_next = robot_pose.inv() * palm_pose_next
            palm_delta_pose = palm_pose.inv() * palm_pose_next
            delta_axis, delta_angle = quat2axangle(palm_delta_pose.q)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
                delta_axis = -delta_axis
            delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
            delta_pose = np.concatenate([
                palm_pose_next.p - palm_pose.p,
                delta_axis_world * delta_angle
            ])

            action = np.concatenate([delta_pose * 100, hand_qpos])
            actions.append(action)

            palm_pose = palm_pose_next
            hand_qpos = player.env.robot.get_drive_target()[arm_dof:]

        return actions

    def render_single_frame(self, player, idx):
        player.scene.unpack(player.get_sim_data(idx))
        image = player.env.get_observation()["relocate_view-rgb"]

        return image

    def validate_actions(self, player, actions, video_path):
        player.scene.unpack(player.get_sim_data(0))
        video = []

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        robot_arm_control_params = player.meta_data["finger_control_params"]
        finger_control_params = player.meta_data["robot_arm_control_params"]
        for joint in player.env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        player.env.rl_step = player.env.simple_sim_step

        for i in tqdm(range(len(actions))):
            real_action = apply_IK_get_real_action(actions[i], player.env,
                player.env.robot.get_qpos(), True)
            player.env.step(real_action)
            image = player.env.get_observation()["relocate_view-rgb"]
            video.append(image.cpu().numpy())

        video = (np.stack(video) * 255).astype(np.uint8)
        imageio.mimsave(video_path, video, fps=120)

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not ckpt_path.endswith("best.pth"):
            self.epoch_start = int(os.path.basename(ckpt_path)\
                .split(".")[0].split("_")[1]) - 1
        self.log_dir = os.path.dirname(ckpt_path)

    def save_checkpoint(self, epoch):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.log_dir, f"model_{epoch}.pth")
        torch.save(state_dict, save_path)

    def _train_epoch(self):
        loss_train = 0
        if self.args.finetune_backbone:
            self.model.policy_net.train()
        else:
            self.model.train()

        for i in tqdm(range(len(self.demos_train))):
            demo_len = len(self.demos_train[i]["data"])
            player = self.init_player(self.demos_train[i])

            beg = 0
            while beg < demo_len:
                end = min(beg + self.args.batch_size, demo_len)
                images = torch.stack([self.render_single_frame(player, t)
                    for t in range(beg, end)]).permute((0, 3, 1, 2))
                robot_qpos = torch.from_numpy(np.stack([
                    self.demos_train[i]["data"][t]["robot_qpos"]
                    for t in range(beg, end)])).to(self.args.device)
                actions = torch.from_numpy(np.stack([
                    self.demos_train[i]["data"][t]["action"]
                    for t in range(beg + self.args.window_size - 1, end)]))\
                    .to(self.args.device)

                # TODO: sim only here, fix it later
                actions_pred = self.model(images, robot_qpos, "sim")
                loss = self.criterion(actions_pred, actions)
                loss.backward()
                loss_train += loss.cpu().item()

                beg = end

            # gradient accumulation check
            if (i + 1) % self.args.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i >= 10 and self.args.debug:
                break

        loss_train /= len(self.demos_train)

        return loss_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_val = 0
        self.model.eval()

        for i in tqdm(range(len(self.demos_val))):
            demo_len = len(self.demos_val[i]["data"])
            player = self.init_player(self.demos_val[i])
            images = []
            beg = 0
            while beg < demo_len:
                end = min(beg + self.args.batch_size, demo_len)
                images = torch.stack([self.render_single_frame(player, t)
                    for t in range(beg, end)]).permute((0, 3, 1, 2))
                robot_qpos = torch.from_numpy(np.stack([
                    self.demos_val[i]["data"][t]["robot_qpos"]
                    for t in range(beg, end)])).to(self.args.device)
                actions = torch.from_numpy(np.stack([
                    self.demos_val[i]["data"][t]["action"]
                    for t in range(beg + self.args.window_size - 1, end)]))\
                    .to(self.args.device)

                # TODO: sim only here, fix it later
                actions_pred = self.model(images, robot_qpos, "sim")
                loss = self.criterion(actions_pred, actions)
                loss_val += loss.cpu().item()

                beg = end

            if i >= 10 and self.args.debug:
                break

        loss_val /= len(self.demos_val)
        return loss_val

    def train(self):
        best_success = 0

        for i in range(self.epoch_start, self.args.epochs):
            loss_train = self._train_epoch()
            loss_val = self._eval_epoch()
            metrics = {
                "loss/train": loss_train,
                "loss/val": loss_val,
                "epoch": i,
            }

            self.save_checkpoint("latest")

            # if (i + 1) % self.args.eval_freq == 0\
            #         and (i + 1) >= self.args.eval_beg:
            #     self.save_checkpoint(i + 1)
            #     avg_success = self.eval_in_env(self.args, i + 1,
            #         self.args.eval_x_steps, self.args.eval_y_steps)
            #     metrics.update({
            #         "avg_success": avg_success,
            #     })

            #     if avg_success > best_success:
            #         self.save_checkpoint("best")
            #         best_success = avg_success

            # if not self.args.wandb_off:
            #     wandb.log(metrics)
