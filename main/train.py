import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

import imageio
import numpy as np
import sapien.core as sapien
import torch
import wandb
from omegaconf import OmegaConf

import dataset.bc_dataset as bc_dataset
from main.policy.bc_agent import make_agent
#from att_agent import make_agent

from eval import apply_IK_get_real_action
from feature_extractor import generate_feature_extraction_model
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from logger import Logger
from tqdm import tqdm
from dataset.bc_dataset import RandomShiftsAug, argument_dependecy_checker, prepare_real_sim_data
from hand_teleop.real_world import lab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(agent, validation_loader, L, epoch):
    loss_val = 0
    for iter, data_batch in enumerate(validation_loader):
        valid_obs = data_batch[0].to(device)
        valid_actions = data_batch[-2].to(device)
        sim_real_label = data_batch[-1].to(device)
        if len(data_batch) == 6:
            valid_states = data_batch[2].to(device)
            loss = agent.validate(obs=valid_obs, state=valid_states, action=valid_actions, L=L, step=epoch, mode='eval') 
        else:
            valid_robot_qpos = data_batch[1].to(device)
            valid_states = None
            loss = agent.validate(concatenated_obs=valid_obs, action=valid_actions, robot_qpos=valid_robot_qpos, L=L, step=epoch, sim_real_label=sim_real_label, mode='eval')
        loss_val += loss

    loss_val /= len(validation_loader)

    return loss_val


def eval_in_env(args, agent, log_dir, epoch, x_steps, y_steps):
    with open("{}/{}_meta_data.pickle".format(args["dataset_folder"], args["backbone_type"].replace("/", "")),'rb') as file:
        meta_data = pickle.load(file)
    # --Create Env and Robot-- #
    robot_name = args["robot_name"]
    # task_name = meta_data['task_name']
    task_name = args["task"]
    if 'randomness_scale' in meta_data["env_kwargs"].keys():
        randomness_scale = meta_data["env_kwargs"]['randomness_scale']
    else:
        randomness_scale = 1
    rotation_reward_weight = 0
    use_visual_obs = args['use_visual_obs']
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params']            

    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = True
    env_params['use_gui'] = False

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]

    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        print('Found initial object pose')
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
        object_pos = meta_data["env_kwargs"]['init_obj_pos']

    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        print('Found initial target pose')
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
        target_pos = meta_data["env_kwargs"]['init_target_pos']

    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'dclaw':
        env = DClawRLEnv(**env_params)
    elif task_name == 'hammer':
        env = HammerRLEnv(**env_params)
    elif task_name == 'table_door':
        env = TableDoorRLEnv(**env_params)
    elif task_name == 'insert_object':
        env = InsertObjectRLEnv(**env_params)
    elif task_name == 'mug_flip':
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError
    env.seed(0)
    env.reset()

    if "free" in robot_name:
        for joint in env.robot.get_active_joints():
            name = joint.get_name()
            if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
            elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
            else:
                joint.set_drive_property(*(finger_control_params), mode="acceleration")
        env.rl_step = env.simple_sim_step
    elif "xarm" in robot_name:
        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        env.rl_step = env.simple_sim_step

    if args['use_visual_obs']:

        real_camera_cfg = {
            "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
        }
        
        if task_name == 'table_door':
            camera_cfg = {
            "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                    right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
            }           
        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)

    with open('{}/{}_dataset.pickle'.format(args["dataset_folder"], args["backbone_type"].replace("/", "")), 'rb') as file:
        print('dataset_folder: {}'.format(args["dataset_folder"]))
        dataset = pickle.load(file)
        print(dataset.keys())
        if 'state' in dataset.keys():
            init_robot_qpos = dataset['state'][0][-7-env.robot.dof:-7]
            state_shape = len(dataset['state'][0])
            concatenated_obs_shape = None
            # print('State shape: {}'.format(state_shape))
        else:
            init_robot_qpos = dataset['robot_qpos'][0][:env.robot.dof]
            concatenated_obs_shape = len(dataset['obs'][0])
            state_shape = None
        action_shape = len(dataset['action'][0])

    if concatenated_obs_shape != None:
        feature_extractor, preprocess = generate_feature_extraction_model(backbone_type=args['backbone_type'])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

    env.robot.set_qpos(init_robot_qpos)

    eval_idx = 0
    avg_success = 0
    avg_angle = 0
    progress = tqdm(total=x_steps * y_steps)

    # since in simulation, we always use simulated data, so sim_real_label is always 0
    sim_real_label = [0]

    for x in np.linspace(-0.08, 0.08, x_steps):        # -0.08 0.08 /// -0.05 0
        for y in np.linspace(0.22, 0.28, y_steps):  # 0.12 0.18 /// 0.12 0.32
            video = []
            idx = np.random.randint(len(meta_data['init_obj_poses']))
            sampled_pos = meta_data['init_obj_poses'][idx]
            object_p = np.array([x, y, sampled_pos.p[-1]])
            object_pos = sapien.Pose(p=object_p, q=sampled_pos.q)
            print('Object Pos: {}'.format(object_pos))
            env.reset()
            if "free" in robot_name:
                for joint in env.robot.get_active_joints():
                    name = joint.get_name()
                    if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                        joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                    elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                        joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                    else:
                        joint.set_drive_property(*(finger_control_params), mode="acceleration")
                env.rl_step = env.simple_sim_step
            elif "xarm" in robot_name:
                arm_joint_names = [f"joint{i}" for i in range(1, 8)]
                for joint in env.robot.get_active_joints():
                    name = joint.get_name()
                    if name in arm_joint_names:
                        joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                    else:
                        joint.set_drive_property(*(1 * finger_control_params), mode="force")
                env.rl_step = env.simple_sim_step                
            env.robot.set_qpos(init_robot_qpos)
            env.manipulated_object.set_pose(object_pos)
            for _ in range(10*env.frame_skip):
                env.scene.step()
            obs = env.get_observation()
            if args['use_visual_obs']:
                features = []
                robot_states = []
                next_robot_states = []
                rgb_imgs = []
                next_rgb_imgs = []
                stacked_robot_qpos = [] 
            else:
                oracle_obs = []
            success = False
            max_angle = 0
            for i in range(1700):
                video.append(obs["relocate_view-rgb"].cpu().numpy())
                if concatenated_obs_shape != None:
                    assert args['adapt'] == False
                    if args['use_visual_obs']:
                        features, robot_states, stacked_robot_qpos, obs, concatenate_robot_qpos = bc_dataset.get_stacked_data_from_obs(rgb_imgs=features, robot_states=robot_states, 
                                                                                                                stacked_robot_qpos=stacked_robot_qpos, obs=obs, i=i, concatenate=True, 
                                                                                                                   robot_qpos=np.concatenate([env.robot.get_qpos(),env.ee_link.get_pose().p,env.ee_link.get_pose().q]),
                                                                                                                   feature_extractor=feature_extractor, preprocess=preprocess)
                        obs = obs.reshape((1,-1))
                        #####reshape concatenate_robot_qpos to 1dim##############
                        concatenate_robot_qpos = concatenate_robot_qpos.reshape((1,-1))
                    else:
                        oracle_obs.append(obs)
                        j = len(oracle_obs)-1
                        if j==0:
                            stacked_obs = np.concatenate((oracle_obs[j],oracle_obs[j],oracle_obs[j],oracle_obs[j]))    
                        elif j==1:
                            stacked_obs = np.concatenate((oracle_obs[j-1],oracle_obs[j],oracle_obs[j],oracle_obs[j]))
                        elif j==2:
                            stacked_obs = np.concatenate((oracle_obs[j-2],oracle_obs[j-1],oracle_obs[j],oracle_obs[j]))         
                        else:
                            stacked_obs = np.concatenate((oracle_obs[j-3],oracle_obs[j-2],oracle_obs[j-1],oracle_obs[j]))
                        obs = torch.from_numpy(stacked_obs).to(device)
                        obs = obs.reshape((1,-1))
                # print('State shape: {}'.format(state_shape))
                if state_shape != None:
                    rgb_imgs, robot_states, stacked_imgs, stacked_states = bc_dataset.get_stacked_data_from_obs(rgb_imgs=rgb_imgs, robot_states=robot_states, obs=obs, i=i, concatenate=False)
                
                agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
                if concatenated_obs_shape != None:
                    action = agent.validate(concatenated_obs=obs, robot_qpos=concatenate_robot_qpos, sim_real_label=sim_real_label, mode='test')
                else:
                    action = agent.validate(obs=stacked_imgs, state=stacked_states,sim_real_label=sim_real_label, mode='test')
                action = action.cpu().numpy()
                # NOTE For new version, uncomment below!
                real_action = apply_IK_get_real_action(action, env, env.robot.get_qpos(), use_visual_obs=use_visual_obs)

                # next_obs, reward, done, _ = env.step(action)
                # NOTE For new version, uncomment below!
                next_obs, reward, done, info = env.step(real_action)
                if task_name == "pick_place":
                    if epoch != "best":
                        info_success = info["is_object_lifted"] and info["success"]
                    else:
                        info_success = info["success"]
                elif task_name == "dclaw":
                    info_success = info["success"]
                    max_angle = max(max_angle, info["object_total_rotate_angle"])

                success = success or info_success
                if success:
                    break
                # TODO: Check how this new action should be used in PAD!
                if args['adapt']:
                    action = torch.from_numpy(action).to(device)
                    next_rgb_imgs, next_robot_states, stacked_next_imgs, stacked_next_states = bc_dataset.get_stacked_data_from_obs(rgb_imgs=next_rgb_imgs, robot_states=next_robot_states, obs=next_obs, i=i, concatenate=False)
                    agent.train(train_visual_encoder=True, train_state_encoder=True, train_policy=False, train_inv=True)
                    agent.update_inv(h=stacked_imgs, s=stacked_states, next_h=stacked_next_imgs, next_s=stacked_next_states, action=action)

                # env.render()

                obs = deepcopy(next_obs)

            avg_success += int(success)
            video = (np.stack(video) * 255).astype(np.uint8)
            #If it did not lift the object, consider it as 0.25 success
            if task_name == "pick_place":
                if epoch != "best" and info["success"]:
                    avg_success += 0.25

                is_lifted = info["is_object_lifted"]
                video_path = os.path.join(log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}.mp4")
                #imageio version 2.28.1 imageio-ffmpeg version 0.4.8 scikit-image version 0.20.0
            elif task_name == "dclaw":
                video_path = os.path.join(log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{max_angle}.mp4")
                avg_angle += max_angle
            imageio.mimsave(video_path, video, fps=120)
            eval_idx += 1
            progress.update()

    avg_success /= eval_idx
    if task_name == "dclaw":
        avg_angle /= eval_idx
    progress.close()
    
    print("avg_success in epoch", epoch, ":", avg_success)
    if task_name == "pick_place":
        return avg_success
    elif task_name == "dclaw":
        return avg_success, avg_angle


def main(args):
    # read and prepare data 
    Prepared_Data = prepare_real_sim_data(args['dataset_folder'],args["backbone_type"],args['batch_size'],args['batch_size'],
                                 args['val_ratio'], seed = 20230806)
    it_per_epoch = Prepared_Data['it_per_epoch']
    bc_train_set = Prepared_Data['bc_train_set']
    bc_train_dataloader = Prepared_Data['bc_train_dataloader']
    bc_validation_dataloader = Prepared_Data['bc_validation_dataloader']
    print('Data prepared')
    if 'state' in bc_train_set.dummy_data.keys():
        state_shape = len(bc_train_set.dummy_data['state'])
        obs_shape = bc_train_set.dummy_data['obs'].shape
        concatenated_obs_shape = None
        print("State Shape: {}".format(state_shape))
        print("Observation Shape: {}".format(obs_shape))
    else:
        state_shape = None
        concatenated_obs_shape = len(bc_train_set.dummy_data['obs'])
        print("Concatenated Observation (State + Visual Obs) Shape: {}".format(concatenated_obs_shape))
    action_shape = len(bc_train_set.dummy_data['action'])
    robot_qpos_shape = len(bc_train_set.dummy_data['robot_qpos'])
    print("Action shape: {}".format(action_shape))
    print("robot_qpos shape: {}".format(robot_qpos_shape))
    # make agent
    agent = make_agent(
                       concatenated_obs_shape=concatenated_obs_shape, 
                       action_shape=action_shape, 
                       state_shape=state_shape, 
                       robot_qpos_shape=robot_qpos_shape,
                       args=args, 
                       frame_stack=args['frame_stack']
                       )
    L = Logger("{}_{}".format(args['model_name'],args['num_epochs']))

    if args['use_augmentation']:
        aug = RandomShiftsAug()
        aug = aug.to(device)

    it = 0
    # wandb.log({
    #     'dataset_folder': args['dataset_folder'],
    #     'batch_size': args['batch_size'],
    #     'backbone_type' : args['backbone_type'],
    #     'ss_visual_encoder_lr': args['ss_visual_encoder_lr'],
    #     'ss_state_encoder_lr': args['ss_state_encoder_lr'],
    #     'ss_inv_lr': args['ss_inv_lr'],
    #     'bc_lr': args['bc_lr'],
    #     'bc_beta': args['bc_beta']})
    
    if not args["eval_only"]:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", f"{args['task']}_{args['backbone_type']}_{cur_time}")
        wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
        wandb.login(key=wandb_cfg.key)
        wandb.init(
            project="hand-teleop",
            name=os.path.basename(log_dir),
            config=args
        )
        os.makedirs(log_dir, exist_ok=True)
        best_success = 0
        best_angle = 0
        for epoch in range(args['num_epochs']):
            print('  ','Epoch: ', epoch)
            loss_train = 0
            for _ in tqdm(range(it_per_epoch)):
                it += 1
                time_train_iter = time.time()
                data_batch = next(iter(bc_train_dataloader))
                obs_batch = data_batch[0].to(device)
                # next_obs_batch = data_batch[1].to(device)
                next_obs_batch = None
                action_batch = data_batch[-2].to(device)
                sim_real_label = data_batch[-1].to(device)
        
                if len(data_batch) == 6:
                    state_batch = data_batch[2].to(device)
                    next_state_batch = data_batch[3].to(device)
                else:
                    state_batch = None
                    next_state_batch = None
                    robot_qpos_batch = data_batch[1].to(device)

                if args['use_augmentation'] and len(obs_batch.shape)==4:
                    obs_batch = aug(obs_batch)

                if state_batch is not None:
                    loss = agent.update(obs=obs_batch, state=state_batch, next_obs=next_obs_batch, next_state=next_state_batch,  action=action_batch, L=L, step=epoch, concatenated_obs=None, concatenated_next_obs=None,sim_real_label=sim_real_label)
                else:
                    loss = agent.update(concatenated_obs=obs_batch, concatenated_next_obs=next_obs_batch, action=action_batch, robot_qpos=robot_qpos_batch, sim_real_label=sim_real_label, L=L, step=epoch)
                loss_train += loss
            loss_train /= it_per_epoch

            agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)

            loss_val = evaluate(agent, bc_validation_dataloader, L, epoch)

            metrics = {
                "loss/train": loss_train,
                "loss/val": loss_val,
                "epoch": epoch
            }

            if (epoch + 1) % args["eval_freq"] == 0 and (epoch+1) >= args["eval_start_epoch"]:
                #total_steps = x_steps * y_steps = 4 * 5 = 20
                if args["task"] == "pick_place":
                    avg_success = eval_in_env(args, agent, log_dir, epoch + 1, 4, 5)
                    metrics["avg_success"] = avg_success
                    if avg_success > best_success:
                        agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)
                elif args["task"] == "dclaw":
                    avg_success, avg_angle = eval_in_env(args, agent, log_dir, epoch + 1, 4, 5)
                    metrics["avg_success"] = avg_success
                    metrics["avg_angle"] = avg_angle
                    if avg_angle > best_angle:
                        agent.save(os.path.join(log_dir, f"epoch_best.pt"), args)
                
                agent.save(os.path.join(log_dir, f"epoch_{epoch + 1}.pt"), args)

            agent.train(train_visual_encoder=args['train_visual_encoder'],
                        train_state_encoder=args['train_state_encoder'], 
                        train_policy=args['train_policy'], 
                        train_inv=args['train_inv'])

            wandb.log(metrics)

        agent.load(os.path.join(log_dir, "epoch_best.pt"))
        agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
        if args["task"] == "pick_place":
            final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)
            wandb.log({"final_success": final_success})
            print(f"Final success rate: {final_success:.4f}")
        elif args["task"] == "dclaw":
            final_success, final_angle = eval_in_env(args, agent, log_dir, "best", 10, 10)
            wandb.log({"final_success": final_success, "final_angle": final_angle})
            print(f"Final success rate: {final_success:.4f}")
            print(f"Final angle: {final_angle:.4f}")

        wandb.finish()
    else:
        log_dir = os.path.dirname(args["ckpt"])
        wandb.init(
            project="hand-teleop",
            name=log_dir+args["ckpt"],
            config=args
        )
        os.makedirs(log_dir, exist_ok=True)
        agent.load(args["ckpt"])
        agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
        if args["task"] == "pick_place":
            final_success = eval_in_env(args, agent, log_dir, "best", 10, 10)
            wandb.log({"final_success": final_success})
            print(f"Final success rate: {final_success:.4f}")
        elif args["task"] == "dclaw":
            final_success, final_angle = eval_in_env(args, agent, log_dir, "best", 10, 10)
            wandb.log({"final_success": final_success, "final_angle": final_angle})
            print(f"Final success rate: {final_success:.4f}")
            print(f"Final angle: {final_angle:.4f}")

        wandb.finish()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--demo-folder", required=True)
    parser.add_argument("--backbone-type", default="regnet_y_3_2gf")
    parser.add_argument("--eval-freq", default=200, type=int)
    parser.add_argument("--eval-start-epoch", default=400, type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--batch-size", default=65536, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--wd-coef", default=1e-2, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    args = {
        'dataset_folder': args.demo_folder,
        'batch_size': 65536,
        'val_ratio': 0.1,
        'bc_lr': args.lr,
        'num_epochs': 1600,
        'weight_decay': args.wd_coef,
        'model_name': '',
        'resume': False,
        'load_model_from': None,
        'use_augmentation': False,
        'save': True,
        'save_freq': 100,
        'use_visual_encoder': False,
        'use_visual_backbone': False,
        'backbone_type' : args.backbone_type,
        'frame_stack' : 4,
        'embedding_name' : 'fixed',
        'hidden_dim': 1200,
        'num_layers': 4,
        'num_filters': 32,
        'in_channels': 3,
        'num_shared_layers': 1,
        'train_visual_encoder': False,
        'num_frozen_layers': 0,
        'use_ss': False,
        'use_state_encoder': False,
        'state_hidden_dim': 100,
        'state_encoder_output_dim': 250,
        'train_policy': True,
        'train_state_encoder': False,
        'train_inv': False,
        'ss_visual_encoder_lr':3e-4,
        'ss_state_encoder_lr': 3e-4,
        'ss_inv_lr': 3e-4,
        'bc_beta': 0.99,
        "task": os.path.basename(args.demo_folder),
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        "eval_freq": args.eval_freq,
        "eval_start_epoch": args.eval_start_epoch,
        "eval_only": args.eval_only,
        "ckpt": args.ckpt
    }
    args = argument_dependecy_checker(args)

    main(args)
