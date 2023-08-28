import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self):
		super().__init__()
		self.pad = int(224/21)

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class BCDataset(Dataset):
    def __init__(self, data_paths=None, data=None):
        super().__init__()
        self.paths = None
        if data_paths != None:
            assert data == None
            self.paths = data_paths
            with open(self.paths[0], 'rb') as dummy_file:
                self.dummy_data = pickle.load(dummy_file)
        elif data != None:
            self.keys = data.keys()
            self.obs = torch.from_numpy(data['obs'])
            # self.next_obs = torch.from_numpy(np.array(data['next_obs']))
            self.robot_qpos = torch.from_numpy(data['robot_qpos'])
            self.state = None
            self.next_state = None
            if 'state' in data.keys():
                self.state = torch.from_numpy(data['state'])
                self.next_state = torch.from_numpy(data['next_state'])
            self.action = torch.from_numpy(data['action'])
            self.label = torch.from_numpy(data['sim_real_label'])
            self.dummy_data = {}
            for key in data.keys():
                self.dummy_data[key] = data[key][0]                
        else:
            raise NotImplementedError

    def __len__(self):
        if self.paths != None:
            return len(self.paths)
        else:
            return len(self.action)
    
    def __getitem__(self, idx):
        if self.paths != None:
            with open(self.paths[idx],'rb') as file:
                data = pickle.load(file)
            obs = torch.from_numpy(np.array(data['obs']))
            # next_obs = torch.from_numpy(np.array(data['next_obs']))
            action = torch.from_numpy(np.array(data['action']))
            robot_qpos = torch.from_numpy(np.array(data['robot_qpos']))
            label = torch.from_numpy(np.array(data['sim_real_label']))
            if 'state' in data.keys():
                state = torch.from_numpy(np.array(data['state']))
                next_state = torch.from_numpy(np.array(data['next_state']))
                # return obs, next_obs, state, next_state, action, label
                return obs, state, next_state, action, label
        else:
            obs = self.obs[idx]
            # next_obs = self.next_obs[idx]
            state = None
            next_state = None
            action = self.action[idx]
            robot_qpos = self.robot_qpos[idx]
            label = self.label[idx]
            if 'state' in self.keys:
                state = self.state[idx]
                next_state = self.next_state[idx]
                # return obs, next_obs, state, next_state, action, label
                return obs, state, next_state, action, label
        # return obs, next_obs, robot_qpos, action, label
        return obs, robot_qpos, action, label

def prepare_real_sim_data(dataset_folder, backbone_type, real_batch_size, sim_batch_size, val_ratio = 0.1, seed = 0, using_real_sim = False):
    print('=== Loading trajectories ===')
    with open('{}/{}_dataset.pickle'.format(dataset_folder, backbone_type.replace("/", "")),'rb') as file:
        data = pickle.load(file)

    if using_real_sim:
        for i in range(len(data["sim_real_label"])):
            if data["sim_real_label"][i] == 1:
                real_demo_length = i + 1      
        print("=== real_demo_length: ===")
        print(real_demo_length)
        print("=== sim_demo_length: ===")
        print(len(data["sim_real_label"])-real_demo_length)

        print("=== preparing real data: ===")
        real_data = {'obs': data['obs'][:real_demo_length], 'next_obs': data['next_obs'][:real_demo_length], 'robot_qpos': data['robot_qpos'][:real_demo_length], 'action': data['action'][:real_demo_length], 'sim_real_label': data['sim_real_label'][:real_demo_length]}
        it_per_epoch_real, bc_train_set_real, bc_train_dataloader_real, bc_validation_dataloader_real = prepare_data(real_data, real_batch_size, val_ratio, seed)
        print("=== preparing sim data: ===")
        sim_data = {'obs': data['obs'][real_demo_length:], 'next_obs': data['next_obs'][real_demo_length:], 'robot_qpos': data['robot_qpos'][real_demo_length:], 'action': data['action'][real_demo_length:], 'sim_real_label': data['sim_real_label'][real_demo_length:]}
        it_per_epoch_sim, bc_train_set_sim, bc_train_dataloader_sim, bc_validation_dataloader_sim = prepare_data(sim_data, sim_batch_size, val_ratio, seed)
        Prepared_Data = {"it_per_epoch_real": it_per_epoch_real, "bc_train_set_real": bc_train_set_real, 
                     "bc_train_dataloader_real": bc_train_dataloader_real, "bc_validation_dataloader_real": bc_validation_dataloader_real,
                     "it_per_epoch_sim": it_per_epoch_sim, "bc_train_set_sim": bc_train_set_sim, "bc_train_dataloader_sim": bc_train_dataloader_sim, 
                     "bc_validation_dataloader_sim": bc_validation_dataloader_sim}
    else:
        print("=== preparing only sim or real data: ===")
        print("=== demo_length: ===")
        # print(len(data["sim_real_label"]))
        data_type = "sim" if dataset_folder.split("/")[0] == "sim" else "real"
        it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader = prepare_data(data, real_batch_size, val_ratio, seed, data_type)
        Prepared_Data = {"it_per_epoch": it_per_epoch, "bc_train_set": bc_train_set, "bc_train_dataloader": bc_train_dataloader, 
                     "bc_validation_dataloader": bc_validation_dataloader}

    return Prepared_Data
    
    


def prepare_data(data, batch_size , val_ratio = 0.1, seed = 0,
        data_type="sim_real"):
    
    np.random.seed(seed)
    random_order = np.random.permutation(len(data["action"]))
    # random_order = random_order[:50000]

    obs = np.stack([data["obs"][i] for i in random_order])
    del data["obs"]
    gc.collect()
    # next_obs = np.stack([data["next_obs"][i] for i in random_order])
    del data["next_obs"]
    # gc.collect()
    robot_qpos = np.stack([data["robot_qpos"][i] for i in random_order])
    del data["robot_qpos"]
    gc.collect()
    targets = np.stack([data["action"][i] for i in random_order])
    del data["action"]
    gc.collect()
    # obs = np.stack([data['obs']])
    # obs = obs[random_order]
    # next_obs = np.stack(data['next_obs'])
    # next_obs = next_obs[random_order]
    # robot_qpos = np.stack(data['robot_qpos'])
    # robot_qpos = robot_qpos[random_order]
    # targets = np.stack(data["action"])
    # targets = targets[random_order]
    if data_type == "sim":
        labels = np.zeros(obs.shape[0])
    elif data_type == "real":
        labels = np.ones(obs.shape[0])
    else:
        labels = np.stack(data["sim_real_label"])
        labels = labels[random_order]
    cutoff = int(len(obs) * val_ratio)
    # train_data = dict(obs=obs[cutoff:], next_obs=next_obs[cutoff:], action=targets[cutoff:], robot_qpos=robot_qpos[cutoff:], sim_real_label=labels[cutoff:])
    # validation_data = dict(obs=obs[:cutoff], next_obs=next_obs[:cutoff], action=targets[:cutoff], robot_qpos=robot_qpos[:cutoff], sim_real_label=labels[:cutoff])
    train_data = dict(obs=obs[cutoff:], action=targets[cutoff:], robot_qpos=robot_qpos[cutoff:], sim_real_label=labels[cutoff:])
    validation_data = dict(obs=obs[:cutoff], action=targets[:cutoff], robot_qpos=robot_qpos[:cutoff], sim_real_label=labels[:cutoff])

    bc_train_set = BCDataset(data_paths=None, data=train_data)
    bc_validation_set = BCDataset(data_paths=None, data=validation_data)
    bc_train_dataloader = DataLoader(bc_train_set, batch_size=batch_size, shuffle=True)
    bc_validation_dataloader = DataLoader(bc_validation_set, batch_size=len(bc_validation_set), shuffle=False)
    it_per_epoch = max(len(bc_train_set) // batch_size, 1)
    print('  ', 'total number of training samples', len(bc_train_set))
    print('  ', 'total number of validation samples', len(bc_validation_set))
    print('  ', 'number of iters per epoch', it_per_epoch)
    return it_per_epoch, bc_train_set, bc_train_dataloader, bc_validation_dataloader

def get_stacked_data_from_obs(rgb_imgs, stacked_robot_qpos, robot_qpos, robot_states, obs, i, concatenate=False, feature_extractor=None, preprocess=None):
    if concatenate:
        assert feature_extractor != None
        assert preprocess != None
        robot_state = obs["state"]
        img = obs["relocate_view-rgb"]
        # img = torch.from_numpy(np.moveaxis(img,-1,0)[None, ...])
        img = img.permute((2, 0, 1))[None, ...]
        img = preprocess(img)
        img = img.to(device)
        with torch.no_grad():
            feature = feature_extractor(img)
        feature = feature.cpu().detach().numpy().reshape(-1)
        ################rgb_imgs here is actually features################
        rgb_imgs.append(feature)
        robot_states.append(robot_state)
        stacked_robot_qpos.append(robot_qpos)
        if i == 0:
            obs = np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        elif i == 1:
            obs = np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-1],stacked_robot_qpos[i],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        elif i == 2:
            obs = np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-2],stacked_robot_qpos[i-1],stacked_robot_qpos[i],stacked_robot_qpos[i]))
        else:
            obs = np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]))
            concatenate_robot_qpos = np.concatenate((stacked_robot_qpos[i-3],stacked_robot_qpos[i-2],stacked_robot_qpos[i-1],stacked_robot_qpos[i]))
        obs = torch.from_numpy(obs).to(device) 
        concatenate_robot_qpos = torch.from_numpy(concatenate_robot_qpos).to(device)
        return rgb_imgs, robot_states, stacked_robot_qpos, obs, concatenate_robot_qpos
    else:
        robot_states.append(obs["state"])
        rgb_imgs.append(obs["relocate_view-rgb"])

        # Save the frames - for debugging
        # rgb_pic = (obs["relocate_view-rgb"] * 255).astype(np.uint8)
        # imageio.imsave("./temp/eval/relocate-rgb_{}.png".format(i), rgb_pic)

        if i==0:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i],robot_states[i],robot_states[i],robot_states[i]))        
        elif i==1:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i-1],robot_states[i],robot_states[i],robot_states[i]))         
        elif i==2:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
            stacked_states = np.concatenate((robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i]))       
        else:
            stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1), -1, 0)       
            stacked_states = np.concatenate((robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i]))        

        stacked_imgs = torch.from_numpy(stacked_imgs).to(device)
        stacked_imgs = torch.unsqueeze(stacked_imgs, 0)
        stacked_states = torch.from_numpy(stacked_states).to(device)
        stacked_states = torch.unsqueeze(stacked_states, 0)        
        return rgb_imgs, robot_states, stacked_imgs, stacked_states

def argument_dependecy_checker(args):
    if args['resume']:
        assert args['load_model_from'] != None
    if args['use_visual_backbone']:
        assert args['backbone_type'] != None
    else:
        assert args['embedding_name'] != None
    if args['embedding_name'] == 'flexible':
        assert args['num_layers'] > 0
    if args['use_ss']:
        assert args['use_visual_encoder'] == True
        assert args['train_visual_encoder'] == True
        assert args['train_inv'] == True
        assert args['num_shared_layers'] > 0
    if args['use_state_encoder']:
        assert args['state_hidden_dim'] != None
        
    if args['use_visual_backbone']:
        args['model_name'] = '{}'.format(args['backbone_type'])
    elif args['use_visual_encoder']:
        args['model_name'] = '{}'.format(args['embedding_name'])
    else:
        args['model_name'] = 'policy_only'
    if args['use_state_encoder']:
        args['model_name'] = '{}_using_se'.format(args['model_name'])
    if args['use_ss']:
        args['model_name'] = '{}_with_pad'.format(args['model_name'])
    args['model_name'] = '{}_{}'.format(args['model_name'], args['dataset_folder'])

    return args
