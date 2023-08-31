import os
import numpy as np
import pickle
import imageio
from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as T

from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv

from hand_teleop.player.player import *
from hand_teleop.player.player_augmentation import *
from hand_teleop.player.randomization_utils import *
from hand_teleop.real_world import lab

from main.feature_extractor import generate_features, generate_feature_extraction_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play_multiple_sim_visual(args):
    demo_files = []
    # dataset_folder = '{}_{}_{}'.format(args['sim_demo_folder'].split('/')[-3],args['sim_demo_folder'].split('/')[-2],args['sim_demo_folder'].split('/')[-1])
    # os.makedirs("./sim/baked_data/{}".format(dataset_folder), exist_ok=True)
    dataset_folder = args["out_folder"]
    os.makedirs(dataset_folder, exist_ok=True)
    # shutil.rmtree("./sim/baked_data/{}".format(dataset_folder),ignore_errors=True)
    if args['with_features']:
        assert args['backbone_type'] != None
        model, preprocess = generate_feature_extraction_model(args["backbone_type"])
        model = model.to("cuda:0")
        model.eval()

    for file_name in os.listdir(args['sim_demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['sim_demo_folder'], file_name))
    print('Replaying the sim demos and creating the dataset:')
    print('---------------------')
    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[], sim_real_label=[])
    init_obj_poses = []
    demo_files = sorted(demo_files)

    for demo_id, file_name in enumerate(tqdm(demo_files)):
        print(file_name)
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
            visual_baked, meta_data = play_one_real_sim_visual_demo(demo=demo, robot_name=args['robot_name'], domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], retarget=args['retarget'])
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
            # visual_baked_demos.append(visual_baked)
        visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model=model, preprocess=preprocess)
        # since here we are using sim data, we set sim_real_label = 0
        
    visual_training_set['sim_real_label'] = [0 for _ in range(len(visual_training_set['obs']))]

    if visual_training_set['obs'] and visual_training_set['action'] and visual_training_set['robot_qpos']:
        assert len(visual_training_set['obs']) == len(visual_training_set['action'])
        assert len(visual_training_set['obs']) == len(visual_training_set['robot_qpos'])

        print("Dataset ready:")
        print('----------------------')
        print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
        print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
        print("Shape of Robot_qpos: {}".format(visual_training_set['robot_qpos'][0].shape))
        print("Action dimension: {}".format(len(visual_training_set['action'][0])))
        dataset_path = "{}/{}_dataset.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
        with open(dataset_path,'wb') as file:
            pickle.dump(visual_training_set, file)
    print('dataset is saved in the folder: ./sim/baked_data/{}'.format(dataset_folder))
    meta_data_path = "{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
    meta_data['init_obj_poses'] = init_obj_poses
    with open(meta_data_path,'wb') as file:
        pickle.dump(meta_data, file)

def play_multiple_real_visual(args):  
    demo_files = []
    dataset_folder = args["out_folder"]
    os.makedirs(dataset_folder, exist_ok=True)
    # shutil.rmtree("./sim/baked_data/{}".format(dataset_folder),ignore_errors=True)
    if args['with_features']:
        assert args['backbone_type'] != None
        model, preprocess = generate_feature_extraction_model(args['backbone_type'])
        model = model.to(device)
        model.eval()
    
    for file_name in os.listdir(args['real_demo_folder']):
        if ".pkl" in file_name:
            demo_files.append(os.path.join(args['real_demo_folder'], file_name))

    print('Replaying the real demos and creating the dataset:')
    print('---------------------')
    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[], sim_real_label=[])
    init_obj_poses = []
    for demo_id, file_name in enumerate(demo_files):

        print("file_name: ", file_name)
        image_file = file_name.strip(".pkl")
        print("image_file: ", image_file)

        with open(file_name, 'rb') as file:
            real_demo = pickle.load(file)
            #path = "./sim/raw_data/xarm/less_random/pick_place_mustard_bottle/mustard_bottle_0004.pickle"
            task_name = args['real_demo_folder'].split('/')[-1]
            path = "./sim/raw_data/xarm/less_random/{}/{}_{}_0004.pickle".format(task_name,task_name.split('_')[-2],task_name.split('_')[-1])
            demo = np.load(path, allow_pickle=True)

            visual_baked, meta_data = play_one_real_sim_visual_demo(demo=demo, real_demo=real_demo, real_images=image_file, robot_name=args['robot_name'], 
                                                                    domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], 
                                                                    retarget=args['retarget'],using_real_data=True)
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
            # visual_baked_demos.append(visual_baked)
        visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model = model, preprocess = preprocess)
        # since here we are using real data, we set sim_real_label = 1
    
    visual_training_set['sim_real_label'] = [1 for _ in range(len(visual_training_set['obs']))]

    if visual_training_set['obs'] and visual_training_set['action'] and visual_training_set['robot_qpos']:
        assert len(visual_training_set['obs']) == len(visual_training_set['action'])
        assert len(visual_training_set['obs']) == len(visual_training_set['robot_qpos'])
        
        print("Dataset ready:")
        print('----------------------')
        print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
        print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
        print("Action dimension: {}".format(len(visual_training_set['action'][0])))
        print("Robot_qpos dimension: {}".format(len(visual_training_set['robot_qpos'][0])))
        dataset_path = "{}/{}_dataset.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
        with open(dataset_path,'wb') as file:
            pickle.dump(visual_training_set, file)
    print('dataset is saved in the folder: ./sim/baked_data/{}'.format(dataset_folder))
    meta_data_path = "{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
    meta_data['init_obj_poses'] = init_obj_poses
    with open(meta_data_path,'wb') as file:
        pickle.dump(meta_data, file)

def play_multiple_sim_real_visual(args):  
    demo_files = []
    dataset_folder = args["out_folder"]
    os.makedirs(dataset_folder, exist_ok=True)

    if args['with_features']:
        assert args['backbone_type'] != None
        model, preprocess = generate_feature_extraction_model(args['backbone_type'])
        model = model.to(device)
        model.eval()
    
    for file_name in os.listdir(args['real_demo_folder']):
        if ".pkl" in file_name:
            demo_files.append(os.path.join(args['real_demo_folder'], file_name))

    print('Replaying the real demos and creating the dataset:')
    print('---------------------')
    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[], sim_real_label=[])
    init_obj_poses = []
    for demo_id, file_name in enumerate(demo_files):

        print("file_name: ", file_name)
        image_file = file_name.strip(".pkl")
        print("image_file: ", image_file)

        with open(file_name, 'rb') as file:
            real_demo = pickle.load(file)
            task_name = args['real_demo_folder'].split('/')[-1]
            path = "./sim/raw_data/xarm/less_random/{}/{}_{}_0004.pickle".format(task_name,task_name.split('_')[-2],task_name.split('_')[-1])
            demo = np.load(path, allow_pickle=True)
            visual_baked, meta_data = play_one_real_sim_visual_demo(demo=demo, real_demo=real_demo, real_images=image_file, robot_name=args['robot_name'], 
                                                                    domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], 
                                                                    retarget=args['retarget'],using_real_data=True)
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
            # visual_baked_demos.append(visual_baked)
            visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model = model, preprocess = preprocess)

    real_demo_length = len(visual_training_set['obs'])
    
    demo_files = []
    for file_name in os.listdir(args['sim_demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['sim_demo_folder'], file_name))
    print('Replaying the sim demos and creating the dataset:')
    print('---------------------')
    for demo_id, file_name in enumerate(demo_files):
        print(file_name)
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
            visual_baked, meta_data = play_one_real_sim_visual_demo(demo=demo, robot_name=args['robot_name'], domain_randomization=args['domain_randomization'], randomization_prob=args['randomization_prob'], retarget=args['retarget'])
            init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
            # visual_baked_demos.append(visual_baked)
            visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model=model, preprocess=preprocess)
    
    sim_demo_length = len(visual_training_set['obs']) - real_demo_length
     # since here we are using real data, we set sim_real_label = 1
    visual_training_set['sim_real_label'] = [1 for _ in range(real_demo_length)]
    # since here we are using sim data, we set sim_real_label = 0
    visual_training_set['sim_real_label'].extend([0 for _ in range(sim_demo_length)])

    if visual_training_set['obs'] and visual_training_set['action'] and visual_training_set['robot_qpos']:
        assert len(visual_training_set['obs']) == len(visual_training_set['action'])
        print("Dataset ready:")
        print('----------------------')
        print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
        print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
        print("Action dimension: {}".format(len(visual_training_set['action'][0])))
        print("Robot_qpos dimension: {}".format(len(visual_training_set['robot_qpos'][0])))
        print("Real_Demos: {}".format(real_demo_length))
        print("Sim_Demos: {}".format(sim_demo_length))
        dataset_path = "{}/{}_dataset.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
        with open(dataset_path,'wb') as file:
            pickle.dump(visual_training_set, file)
    print('dataset is saved in the folder: ./real_sim_mix/baked_data/{}'.format(dataset_folder))
    meta_data_path = "{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
    meta_data['init_obj_poses'] = init_obj_poses
    with open(meta_data_path,'wb') as file:
        pickle.dump(meta_data, file)
        
def play_multiple_sim_aug(args):
    ################Using Augmented Sim Data################
    dataset_folder = args["out_folder"]
    visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[], sim_real_label=[])
    init_obj_poses = []
    demo_files = []
    for file_name in os.listdir(args['sim_demo_folder']):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args['sim_demo_folder'], file_name))
    print('Augmenting sim demos and creating the dataset:')
    print('---------------------')
    np.random.seed(20220824)
    for demo_id, file_name in enumerate(demo_files):
        print(file_name)
        num_test = 0
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)
            for i in tqdm(range(args['kinematic_aug'])):
                x = np.random.uniform(-0.11,0.11)
                y = np.random.uniform(-0.11,0.11)
                
                if np.fabs(x) <= 0.01 and np.fabs(y) <= 0.01:
                    continue

                info_success, visual_baked, meta_data = bake_visual_demonstration_test_augmented(all_data=demo, init_pose_aug=sapien.Pose([x, y, 0], [1, 0, 0, 0]), retarget=args['retarget'])

                if info_success:
                    print("##############SUCCESS##############")
                    num_test += 1
                    print("##########This is {}th try and {}th success##########".format(i+1,num_test))
                    init_obj_poses.append(meta_data['env_kwargs']['init_obj_pos'])
                    # visual_baked_demos.append(visual_baked)
                    visual_training_set = stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model=model, preprocess=preprocess)
        
        sim_demo_length = len(visual_training_set['obs'])
        # since here we are using real data, we set sim_real_label = 1
        visual_training_set['sim_real_label'] = [0 for _ in range(sim_demo_length)]

        if visual_training_set['obs'] and visual_training_set['action'] and visual_training_set['robot_qpos']:
            assert len(visual_training_set['obs']) == len(visual_training_set['action'])
            print(f"Augment sim dataset for demo_{demo_id} ready:")
            print('----------------------')
            print("Number of datapoints: {}".format(len(visual_training_set['obs'])))
            print("Shape of observations: {}".format(visual_training_set['obs'][0].shape))
            print("Action dimension: {}".format(len(visual_training_set['action'][0])))
            print("Robot_qpos dimension: {}".format(len(visual_training_set['robot_qpos'][0])))
            dataset_path = "{}/{}_dataset_demo_{}_aug.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""),demo_id)

            with open(dataset_path,'wb') as file:
                pickle.dump(visual_training_set, file)
                
        print('dataset is saved in the folder: ./real_sim_mix/baked_data/{}'.format(dataset_folder))
        meta_data_path = "{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"].replace("/", ""))
        meta_data['init_obj_poses'] = init_obj_poses
        with open(meta_data_path,'wb') as file:
            pickle.dump(meta_data, file)


def average_angle_handqpos(hand_qpos):
    delta_angles = []
    for i in range(0,len(hand_qpos),4):
        qpos = hand_qpos[i:i+4]
        delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(qpos)
        if delta_angle > np.pi:
            delta_angle = 2 * np.pi - delta_angle
        delta_angles.append(delta_angle)
    return np.mean(delta_angles)

def play_one_real_sim_visual_demo(demo, robot_name, domain_randomization, randomization_prob, 
                               retarget=False, real_demo=None, real_images=None, using_real_data = False):
    if robot_name == 'mano':
        assert retarget == False
    # Get env params
    meta_data = demo["meta_data"]
    if not retarget:    
        assert robot_name == meta_data["robot_name"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    meta_data["task_name"] = task_name
    if 'randomness_scale' in meta_data["env_kwargs"].keys():
        randomness_scale = meta_data["env_kwargs"]['randomness_scale']
    else:
        randomness_scale = 1
    data = demo["data"]
    rotation_reward_weight = 0
    use_visual_obs = True
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params']            

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = False

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None

    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        print('Found initial object pose')
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']

    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        print('Found initial target pose')
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']

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
    
    if domain_randomization:
        p = np.random.rand(1)[0]
        if p <= randomization_prob:
            env = randomize_env_colors(task_name=task_name, env=env)

    if not retarget:
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

    env.reset()

    # real_camera_cfg = {
    #     "relocate_view": dict( pose= lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(47.4), resolution=(320, 240))
    # }

    real_camera_cfg = {
        "relocate_view": dict( pose= lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(320, 240))
    }

    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }   

    env.setup_camera_from_config(real_camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'dclaw':
        player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

     # Retargeting
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    elif using_real_data:
        baked_data = real_demo
    else:
        baked_data = player.bake_demonstration()
    
    visual_baked = dict(obs=[], action=[],robot_qpos=[])

    
    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    if using_real_data:
        env.robot.set_qpos(baked_data[0]["teleop_cmd"])
        print("init_qpos: ",baked_data[0]["teleop_cmd"])
    else:
        env.robot.set_qpos(baked_data["robot_qpos"][0])
        if baked_data["robot_qvel"] != []:
            env.robot.set_qvel(baked_data["robot_qvel"][0])

    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))

    if using_real_data:
        ee_pose = baked_data[0]["ee_pose"]
        hand_qpos_prev = baked_data[0]["teleop_cmd"][env.arm_dof:]
    else:
        ee_pose = baked_data["ee_pose"][0]
        hand_qpos_prev = baked_data["action"][0][env.arm_dof:]

    if using_real_data:
        for idx in range(len(baked_data)):
            
            # NOTE: robot.get_qpos() version
            if idx != len(baked_data)-1:
                ee_pose_next = np.array(baked_data[idx + 1]["ee_pose"])
                ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
                hand_qpos = baked_data[idx+1]["teleop_cmd"][env.arm_dof:]

                delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos

                if ee_pose_delta < 0.0005 and average_angle_handqpos(delta_hand_qpos)*np.pi/180<= 1:
                    continue
                else:

                    ee_pose = ee_pose_next
                    hand_qpos_prev = hand_qpos

                    palm_pose = env.ee_link.get_pose()
                    palm_pose = robot_pose.inv() * palm_pose
                    
                    palm_next_pose = sapien.Pose(ee_pose_next[0:3],ee_pose_next[3:7])
                    palm_next_pose = robot_pose.inv() * palm_next_pose

                    palm_delta_pose = palm_pose.inv() * palm_next_pose
                    delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
                    if delta_angle > np.pi:
                        delta_angle = 2 * np.pi - delta_angle
                        delta_axis = -delta_axis
                    delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                    delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
                    arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]

                    observation = env.get_observation()
                    rgb_pic = torchvision.io.read_image(path = os.path.join('.'+real_images, "frame%04i.png" % idx), mode=torchvision.io.ImageReadMode.RGB)
                    rgb_pic = rgb_pic.permute(1,2,0)
                    #print("rgb_pic: ", rgb_pic.shape)
                    rgb_pic = (rgb_pic / 255.0).type(torch.float32)
                    observation["relocate_view-rgb"] = rgb_pic

                    visual_baked["obs"].append(observation)
                    action = np.concatenate([delta_pose*100, hand_qpos])
                    visual_baked["action"].append(action.astype(np.float32))
                    # Using robot qpos version
                    visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                      env.ee_link.get_pose().p,env.ee_link.get_pose().q]))

                    target_qpos = np.concatenate([arm_qpos, hand_qpos])
                    env.step(target_qpos)
    else:

        for idx, (obs, qpos, action, ee_pose) in enumerate(zip(baked_data["obs"], baked_data["robot_qpos"],
                                        baked_data["action"], baked_data["ee_pose"])):
            # NOTE: robot.get_qpos() version
            if idx != len(baked_data['obs'])-1:
                ee_pose_next = baked_data["ee_pose"][idx + 1]
                ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
                hand_qpos = baked_data["action"][idx][env.arm_dof:]
                delta_hand_qpos = hand_qpos - hand_qpos_prev
                #if ee_pose_delta > 0.0005 and (average_angle_handqpos(hand_qpos))/np.pi*180 >= 1:
                if ee_pose_delta <= args['delta_ee_pose_bound'] and (average_angle_handqpos(delta_hand_qpos))/np.pi*180 <= 1:
                    continue
                else:
                    ee_pose = ee_pose_next
                    
                    hand_qpos_prev = hand_qpos      
                    palm_pose = env.ee_link.get_pose()
                    palm_pose = robot_pose.inv() * palm_pose
    
                    palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                    palm_next_pose = robot_pose.inv() * palm_next_pose

                    palm_delta_pose = palm_pose.inv() * palm_next_pose
                    delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
                    if delta_angle > np.pi:
                        delta_angle = 2 * np.pi - delta_angle
                        delta_axis = -delta_axis
                    delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                    delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
                    arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]
                    target_qpos = np.concatenate([arm_qpos, hand_qpos])

                    observation = env.get_observation()
                    #print("observation: ", observation.keys())
                    # rgb = observation["relocate_view-rgb"]
                    # rgb_pic = (rgb * 255).astype(np.uint8)
                    # print("rgb_pic: ", observation["relocate_view-rgb"].shape)

                    visual_baked["obs"].append(observation)
                    visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                    # Using robot qpos version
                    visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                      env.ee_link.get_pose().p,env.ee_link.get_pose().q]))

                
                    env.step(target_qpos)
               

    # # For visual obs debugging    
    # for i in range(len(visual_baked["obs"])):
    #     rgb = visual_baked["obs"][i]["relocate_view-rgb"]
    #     rgb_pic = (rgb * 255).astype(np.uint8)
    #     imageio.imsave("./temp/demos/single/relocate-rgb_{}.png".format(i), rgb_pic)
    #     rgb_pic = imageio.imread(, pilmode="RGB")

    return visual_baked, meta_data

def stack_and_save_frames(visual_baked, visual_training_set, demo_id, dataset_folder, args, model, preprocess):
    # 0 menas sim, 1 means real here
    if args['with_features']:
        if 'state' in visual_training_set.keys():
            visual_training_set.pop('state')
            visual_training_set.pop('next_state')
        visual_demo_with_features = generate_features(visual_baked=visual_baked, backbone_type=args['backbone_type'], stack_frames=args['stack_frames'],
                                                      num_data_aug=args['num_data_aug'],augmenter=args['image_augmenter'], model=model, preprocess=preprocess)
        stacked_next_obs = visual_demo_with_features['obs'][1:]
        stacked_obs = visual_demo_with_features['obs'][:-1]
        actions = visual_demo_with_features['action'][:-1]
        stacked_robot_qpos = visual_demo_with_features['robot_qpos'][:-1]
        visual_training_set['obs'].extend(stacked_obs)
        visual_training_set['next_obs'].extend(stacked_next_obs)
        visual_training_set['action'].extend(actions)
        visual_training_set['robot_qpos'].extend(stacked_robot_qpos)
    else:
        if 'robot_qpos' in visual_training_set.keys():
            visual_training_set.pop('robot_qpos')
        rgb_imgs = []
        robot_states = []
        target_actions = []
        for act, obs in zip(visual_baked["action"], visual_baked["obs"]):
            rgb_imgs.append(obs["relocate_view-rgb"])
            robot_states.append(obs["state"])
            target_actions.append(act)
        stacked_frames = []
        stacked_states = []
        for i in range(len(rgb_imgs)):
            if i==0:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i],robot_states[i],robot_states[i],robot_states[i])))         
            elif i==1:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i-1],robot_states[i],robot_states[i],robot_states[i])))         
            elif i==2:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0))
                stacked_states.append(np.concatenate((robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i])))         
            else:
                stacked_frames.append(np.moveaxis(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1), -1, 0))        
                stacked_states.append(np.concatenate((robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i])))         
        assert len(rgb_imgs) == len(stacked_frames)
        assert len(robot_states) == len(stacked_states)
        stacked_next_frames = stacked_frames[1:]
        stacked_frames = stacked_frames[:-1]
        stacked_next_states = stacked_states[1:]
        stacked_states = stacked_states[:-1]
        target_actions = target_actions[:-1]
        visual_training_set['obs'].extend(stacked_frames)
        visual_training_set['next_obs'].extend(stacked_next_frames)
        visual_training_set['state'].extend(stacked_states)
        visual_training_set['next_state'].extend(stacked_next_states)
        visual_training_set['action'].extend(target_actions)

    if args['save_each_frame']:
        dataset_folder = './sim/baked_data/{}'.format(dataset_folder)
        os.makedirs(dataset_folder,exist_ok=True)
        for idx in range(len(visual_training_set['obs'])):
            frame = {}
            frame['obs'] = visual_training_set['obs'][idx]
            frame['next_obs'] = visual_training_set['next_obs'][idx]
            frame['action'] = visual_training_set['action'][idx]
            if 'state' in visual_training_set.keys():
                frame['state'] = visual_training_set['state'][idx]
                frame['next_state'] = visual_training_set['next_state'][idx]
            with open(os.path.join(dataset_folder,'{}_{}.pickle'.format(demo_id, idx)),'wb') as file:
                pickle.dump(frame, file)
        visual_training_set = dict(obs=[], next_obs=[], state=[], next_state=[], action=[], robot_qpos=[])
    
    return visual_training_set


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--backbone-type", required=True)
    parser.add_argument("--delta-ee-pose-bound", default=0.0005, type=float)
    parser.add_argument("--sim-folder", default=None)
    parser.add_argument("--real-folder", default=None)
    parser.add_argument("--out-folder", required=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Since we are doing online retargeting currently, 'retargeting' argument should be False all the time in this file.
    # It might be better to save each frame one by one if you need the images itself. you can save it all as one file if you are just using the features.
    args = parse_args()

    args = {
        #'demo_folder' : './sim/raw_data/xarm/less_random/pick_place_mustard_bottle_0.73',
        #'sim_demo_folder': None,
        'sim_demo_folder' : args.sim_folder,
        #'sim_demo_folder' : './sim/raw_data/xarm/less_random/pick_place_tomato_soup_can',
        #'sim_demo_folder' : './sim/raw_data/xarm/less_random/pick_place_sugar_box',

        #'real_demo_folder': None,
        'real_demo_folder' : args.real_folder,
        #'real_demo_folder' : './real/raw_data/pick_place_tomato_soup_can',
        #'real_demo_folder' : './real/raw_data/pick_place_sugar_box',
    
        "robot_name": "xarm6_allegro_modified_finger",
        'with_features' : True,
        'backbone_type' : args.backbone_type,
        'stack_frames' : True,
        'retarget' : False,
        'save_each_frame' : False,
        'domain_randomization': False,
        'randomization_prob': 0.2,
        'num_data_aug': 5,
        'image_augmenter': T.AugMix(),
        'kinematic_aug': 400,
        'delta_ee_pose_bound': args.delta_ee_pose_bound,
        'out_folder': args.out_folder
    }

    if args['sim_demo_folder'] is not None and args['real_demo_folder'] is None:
        print("##########################Using Only Sim##################################")
        play_multiple_sim_visual(args)
    elif args['sim_demo_folder'] is None and args['real_demo_folder'] is not None:
        print("##########################Using Only Real##################################")
        play_multiple_real_visual(args)
    elif args['sim_demo_folder'] is not None and args['real_demo_folder'] is not None:
        print("##########################Using Sim and Real##################################")
        play_multiple_sim_real_visual(args)
    elif args['kinematic_aug'] > 0:
        play_multiple_sim_aug(args)