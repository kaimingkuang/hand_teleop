import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class HandTeleopDataset(Dataset):

    def __init__(self, data, indices):
        self.obs = data["obs"]
        self.robot_qpos = data["robot_qpos"]
        self.actions = data["actions"]
        self.domains = data["domains"]
        self.indices = indices

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, outer_idx):
        idx = self.indices[outer_idx]
        obs = torch.from_numpy(self.obs[idx])
        action = torch.from_numpy(self.actions[idx])
        robot_qpos = torch.from_numpy(self.robot_qpos[idx])
        domain = torch.tensor(self.domains[idx]).float()

        return {
            "obs": obs,
            "robot_qpos": robot_qpos,
            "action": action,
            "domain": domain
        }


def collate_fn(samples):
    obs = torch.stack([x["obs"] for x in samples])
    robot_qpos = torch.stack([x["robot_qpos"] for x in samples])
    action = torch.stack([x["action"] for x in samples])
    domain = torch.stack([x["domain"] for x in samples])

    return {
        "obs": obs,
        "robot_qpos": robot_qpos,
        "action": action,
        "domain": domain
    }
