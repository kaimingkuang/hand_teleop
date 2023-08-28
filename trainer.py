import gc
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

from data import HandTeleopDataset, collate_fn
from hand_teleop.player.player import PickPlaceEnvPlayer
from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from losses import DomainClfLoss
from hand_teleop.real_world import lab
from main.eval import apply_IK_get_real_action
from model import Agent


class Trainer:

    def __init__(self, args):
        self.args = args

        self.epoch_start = 0

        self.dl_train, self.dl_val = self.init_dataloaders(args)

        self.model = Agent(args).to(args.device)

        if args.grad_rev:
            self.criterion = DomainClfLoss(0.5)
        else:
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

    def init_dataloaders(self, args):
        with open(os.path.join(args.demo_folder,
                f"{args.backbone}_dataset.pickle"), "rb") as file:
            data = pickle.load(file)

        train_indices, val_indices = train_test_split(
            np.arange(len(data["obs"])), test_size=args.val_pct,
            random_state=args.seed)
        obs = np.stack(data["obs"])
        robot_qpos = np.stack(data["robot_qpos"])
        actions = np.stack(data["action"])
        domains = np.stack(data["sim_real_label"])
        del data
        gc.collect()
        data = {
            "obs": obs,
            "robot_qpos": robot_qpos,
            "actions": actions,
            "domains": domains,
        }
        ds_train = HandTeleopDataset(data, train_indices)
        ds_val = HandTeleopDataset(data, val_indices)
        dl_train = DataLoader(ds_train, args.batch_size, True,
            num_workers=args.n_workers, collate_fn=collate_fn)
        dl_val = DataLoader(ds_val, args.batch_size, False,
            num_workers=args.n_workers, collate_fn=collate_fn)

        return dl_train, dl_val

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
        if self.args.grad_rev:
            action_loss_train = 0
            domain_loss_train = 0
        self.model.train()

        for i, sample in enumerate(tqdm(self.dl_train)):
            obs = sample["obs"].to(self.args.device)
            robot_qpos = sample["robot_qpos"].to(self.args.device)
            actions = sample["action"].to(self.args.device)

            if self.args.grad_rev:
                actions_pred, domains_pred = self.model(robot_qpos, obs)
                domains = sample["domain"].to(self.args.device)
                loss, action_loss, domain_loss = self.criterion(actions_pred,
                    actions, domains_pred, domains)
            else:
                actions_pred = self.model(robot_qpos, obs)
                loss = self.criterion(actions_pred, actions)

            loss.backward()
            loss_train += loss.cpu().item()
            if self.args.grad_rev:
                action_loss_train += action_loss.cpu().item()
                domain_loss_train += domain_loss.cpu().item()

            # gradient accumulation check
            if (i + 1) % self.args.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i >= 5 and self.args.debug:
                break

        loss_train /= len(self.dl_train)
        if self.args.grad_rev:
            action_loss_train /= len(self.dl_train)
            domain_loss_train /= len(self.dl_train)
            return loss_train, action_loss_train, domain_loss_train
        else:
            return loss_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_val = 0
        if self.args.grad_rev:
            action_loss_val = 0
            domain_loss_val = 0
        self.model.eval()

        for i, sample in enumerate(tqdm(self.dl_val)):
            obs = sample["obs"].to(self.args.device)
            robot_qpos = sample["robot_qpos"].to(self.args.device)
            actions = sample["action"].to(self.args.device)

            if self.args.grad_rev:
                actions_pred, domains_pred = self.model(robot_qpos, obs)
                domains = sample["domain"].to(self.args.device)
                loss, action_loss, domain_loss = self.criterion(actions_pred,
                    actions, domains_pred, domains)
            else:
                actions_pred = self.model(robot_qpos, obs)
                loss = self.criterion(actions_pred, actions)

            loss_val += loss.cpu().item()
            if self.args.grad_rev:
                action_loss_val += action_loss.cpu().item()
                domain_loss_val += domain_loss.cpu().item()

            if i >= 5 and self.args.debug:
                break

        loss_val /= len(self.dl_val)
        if self.args.grad_rev:
            action_loss_val /= len(self.dl_train)
            domain_loss_val /= len(self.dl_train)
            return loss_val, action_loss_val, domain_loss_val
        else:
            return loss_val

    def train(self):
        best_success = 0

        for i in range(self.epoch_start, self.args.epochs):
            if self.args.grad_rev:
                loss_train, action_loss_train, domain_loss_train =\
                    self._train_epoch()
                loss_val, action_loss_val, domain_loss_val = self._eval_epoch()
                metrics = {
                    "loss/train": loss_train,
                    "loss/val": loss_val,
                    "action_loss/train": action_loss_train,
                    "action_loss/val": action_loss_val,
                    "domain_loss/train": domain_loss_train,
                    "domain_loss/val": domain_loss_val,
                    "epoch": i,
                }
            else:
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
