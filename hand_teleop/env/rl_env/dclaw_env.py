from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.dclaw_env import DClawEnv
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.real_world import lab
from hand_teleop.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info


class DClawRLEnv(DClawEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", constant_object_state=False,
                 rotation_reward_weight=0, object_name="dclaw_3x", object_seed = 0, object_scale=1, randomness_scale=1, friction=1,
                 object_pose_noise=0.01, zero_joint_pos=None, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_name, object_scale, randomness_scale, friction, **renderer_kwargs)

        if robot_name != "mano":
            self.setup(robot_name)
            # NOTE: For using allegro without retargeting!
            # self.rl_step = self.simple_sim_step
        else:
            self.mano_setup(frame_skip, zero_joint_pos)
        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise

        self.object_angle = self.get_object_rotate_angle()
        # self.object_total_rotate_angle = 0

        # Parse link name
        if self.is_robot_free:
            if robot_name == "mano":
                info = generate_free_robot_hand_info()["mano_hand_free"]
            else:
                info = generate_free_robot_hand_info()[robot_name]
        else:
            info = generate_arm_robot_hand_info()[robot_name]
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

    def mano_setup(self, frame_skip, zero_joint_pos):
        self.robot_name = "mano"
        self.mano_robot = MANORobotHand(self.scene, self.renderer, init_joint_pos=zero_joint_pos,
                                        control_interval=frame_skip * self.scene.get_timestep(),
                                        scale=1)
        self.robot = self.mano_robot.robot
        self.is_robot_free = True
        if self.is_robot_free:
            info = generate_free_robot_hand_info()["mano_hand_free"]
            # velocity_limit = np.array([1.0] * 3 + [1.57] * 3 + [3.14] * (self.robot.dof - 6))
            # self.velocity_limit = np.stack([-velocity_limit, velocity_limit], axis=1)
            init_pose = sapien.Pose(np.array([-0.3, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
            self.robot.set_pose(init_pose)
            self.arm_dof = 0

        self.robot_info = info
        self.robot_collision_links = [link for link in self.robot.get_links() if len(link.get_collision_shapes()) > 0]
        self.control_time_step = self.scene.get_timestep() * self.frame_skip

        # Choose different step function
        if self.is_robot_free:
            self.rl_step = self.mano_sim_step

        # Scene light and obs
        if self.use_visual_obs:
            self.get_observation = self.get_visual_observation
            if not self.no_rgb:
                add_default_scene_light(self.scene, self.renderer)
        else:
            self.get_observation = self.get_oracle_state

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        # target_in_object = self.target_pose.p - object_pose.p
        # target_in_palm = self.target_pose.p - palm_pose.p
        object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        # theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
        # return np.concatenate(
        #     [object_pose_vec, object_in_palm, object_in_box, robot_qpos_vec, palm_pose.p, palm_pose.q])
        return np.concatenate(
            [object_pose_vec, robot_qpos_vec, palm_pose.p, palm_pose.q])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        ee_pose = self.ee_link.get_pose()
        jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(robot_qpos_vec[:self.arm_dof]).flatten()
        return np.concatenate([jacobian, robot_qpos_vec, ee_pose.p, ee_pose.q])

    def get_reward(self, action):
        reward = 0
        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # super().reset(seed=seed) helin
        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
            # init_pose = sapien.Pose(np.array([-0.55, 0, 0.17145]), transforms3d.euler.euler2quat(0, 0, 0))
        else:
            init_pose = sapien.Pose(np.array([-0.3, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))

        self.robot.set_pose(init_pose)
        self.reset_internal()
        # reset the is_object_lifted flag
        # NOTE: No randomness for now.
        # self.object_episode_init_pose = self.manipulated_object.get_pose()
        # random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        # random_pos = self.np_random.randn(3) * self.object_pose_noise
        # self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        self.object_angle = self.get_object_rotate_angle()
        # self.object_total_rotate_angle = 0

        return self.get_observation()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return False

    @cached_property
    def horizon(self):
        return 250

    def get_object_rotate_angle(self):
        # check if the object is in contact with the plate
        for link in self.manipulated_object.get_links():
            if "up" in link.get_name():
                rotate_object_qpos = link.get_pose().q
                x,y,z = transforms3d.euler.quat2euler(rotate_object_qpos)
                z_angle = z/np.pi*180

        return z_angle

    # def _is_object_rotated(self):
    #     # check the x-y position of the object against the target
    #     delta_angle = self.get_object_rotate_angle() - self.object_angle
    #     self.object_angle = self.get_object_rotate_angle()
    #     self.object_total_rotate_angle += np.fabs(delta_angle)

    #     return delta_angle != 0

    @property
    def object_total_rotate_angle(self):
        if self.object_angle < 0 and self.object_angle >= -135:
            return 0
        elif self.object_angle >= 0:
            return self.object_angle
        else:
            return self.object_angle + 360

    def _is_successful(self):
        return self.object_angle >= -180 and self.object_angle <= -135

    def get_info(self):
        return {
            # "is_object_rotated": self._is_object_rotated(),
            "object_total_rotate_angle": self.object_total_rotate_angle,
            "success": self._is_successful()
        }


def main_env():
    from time import time
    env = DClawRLEnv(use_gui=True, robot_name="allegro_hand_free",
                     object_name="tomato_soup_can", object_category="YCB", frame_skip=5,
                     use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()

    tic = time()
    env.reset()
    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    tic = time()
    for i in range(1000):
        action = np.random.rand(robot_dof) * 2 - 1
        action[2] = 0.1
        obs, reward, done, _ = env.step(action)
    tac = time()
    print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    env.reset()
    pose = env.palm_link.get_pose()
    for i in range(5000):
        action = np.zeros(robot_dof)
        action[0] = 0.1
        obs, reward, done, _ = env.step(action)
        env.render()
        if i == 200:
            pose_error = pose.inv() * env.palm_link.get_pose()
            print(pose_error)

    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
