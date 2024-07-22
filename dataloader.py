import torch
import os
import h5py
import cv2
import time
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import pyquaternion as pyq
from sklearn.cluster import dbscan

import IPython
e = IPython.embed


class NormProcessor:
    def __init__(
        self,
        device,
        data_sources
    ):
        self.device = device
        self.data_sources = data_sources
        self.image_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.dict = {
            'pos': self.norm_pos,
            'images': self.norm_image,
            'depths': self.norm_depth,
            'voxs': self.norm_vox,
        }

        self.qpos_mean, self.qpos_std = None, None
        self.action_mean, self.action_std = None, None

    def load_state(self, norm_dir):
        stats_path = os.path.join(norm_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.qpos_mean = torch.tensor(stats['qpos_mean'], requires_grad=False).to(self.device)
        self.qpos_std = torch.tensor(stats['qpos_std'], requires_grad=False).to(self.device)
        self.action_mean = torch.tensor(stats['action_mean'], requires_grad=False).to(self.device)
        self.action_std = torch.tensor(stats['action_std'], requires_grad=False).to(self.device)

    def infer_norm(self, qpos, all_cond, task):
        qpos = self.norm_pos(qpos)
        normed_cond = []
        for i, sources in enumerate(self.data_sources):
            cond = all_cond[i]
            normed_cond.append(self.dict[sources](cond))

        return qpos, normed_cond, task

    def train_norm(self, qpos, all_cond, task, action, is_pad):
        qpos = self.norm_pos(qpos)
        normed_cond = []
        for i, sources in enumerate(self.data_sources):
            cond = all_cond[i]
            normed_cond.append(self.dict[sources](cond))
        action = self.norm_action(action)

        return qpos, normed_cond, task, action, is_pad

    @torch.no_grad()
    def norm_image(self, images):
        return self.image_normalize(images)

    def norm_depth(self, depth):
        print('no implement norm depth')
        exit(0)

    def norm_vox(self, vox):
        vox = - vox + 3
        vox[vox == 3] = 0
        vox = vox / 2
        return vox

    @torch.no_grad()
    def norm_pos(self, qpos):
        return (qpos - self.qpos_mean) / self.qpos_std

    @torch.no_grad()
    def norm_action(self, action):
        return (action - self.action_mean) / self.action_std

    def post_action(self, pred_action):
        pred_action = pred_action * self.action_std + self.action_mean

        return pred_action.squeeze(0).cpu().numpy()


class FormatProcessor:
    def __init__(
        self,
        data_sources,
        cam_names,
        vox_robot_base,
        vox_dict,
        voxel_size,
        occugrid_ori,
        occugrid_range,
        device,
        chunck_size,
    ):
        self.data_sources = data_sources
        self.cam_names = cam_names
        self.device = device
        self.chunck_size = chunck_size

        self.vox_robot_base = vox_robot_base
        self.vox_dict = vox_dict

        self.voxel_size = voxel_size
        self.occugrid_ori = occugrid_ori
        self.occugrid_range = occugrid_range

        self.dict = {
            'images': self.process_image,
            'depths': self.process_depth,
            'voxs': self.process_vox,
        }

    # ------------------------------------------------- train  ---------------------------------------------------
    def read_data_all(self, dataset_base, episode_ids, max_steps, chunck_size):
        all_obs = []
        all_lens = []
        for episode_id in episode_ids:
            dataset_path = os.path.join(dataset_base, f'episode_{episode_id}.hdf5')

            obs = {}
            print('loading  dataset_path')
            with h5py.File(dataset_path, 'r') as root:
                episode_len, state_dim = root['/action'].shape

                # get observation at start_ts only
                obs['qpos'] = root['/observations/qpos'][()]

                for sources in self.data_sources:
                    obs[sources] = {}
                    for cam_name in self.cam_names:
                        obs[sources][cam_name] = root[f'/observations/{sources}/{cam_name}'][()]

                action = root['/action'][()].astype(float)

                obs['action'] = np.zeros((episode_len + chunck_size - 1, state_dim), dtype=np.float32)
                obs['action'][:episode_len] = action

                obs['pad'] = np.zeros(episode_len + chunck_size - 1)
                obs['pad'][episode_len:] = 1

                obs_list = []
                for ind in range(episode_len):
                    each_obs = {}
                    each_obs['qpos'] = obs['qpos'][ind]

                    for sources in self.data_sources:
                        each_obs[sources] = {}
                        for cam_name in self.cam_names:
                            each_obs[sources][cam_name] = obs[sources][cam_name][ind]

                    each_obs['action'] = obs['action'][ind:ind + chunck_size]
                    each_obs['pad'] = obs['pad'][ind:ind + chunck_size]

                    obs_list.append(each_obs)

                all_obs.append(obs_list)
                all_lens.append(episode_len)

        return all_obs, all_lens

    def read_data_old(self, dataset_path, max_steps):
        sample_full_episode = False
        obs = {}

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len, state_dim = root['/action'].shape

            start_ts = 0 if sample_full_episode else np.random.choice(episode_len)

            # get observation at start_ts only
            obs['qpos'] = root['/observations/qpos'][start_ts]

            for sources in self.data_sources:
                obs[sources] = {}
                for cam_name in self.cam_names:
                    obs[sources][cam_name] = root[f'/observations/{sources}/{cam_name}'][start_ts]

            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:episode_len].astype(float)
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        obs['action'] = np.zeros((max_steps, state_dim), dtype=np.float32)
        obs['action'][:action_len] = action

        obs['pad'] = np.zeros(max_steps)
        obs['pad'][action_len:] = 1

        return obs

    def read_data(self, dataset_path, max_steps):
        obs = {}

        with h5py.File(dataset_path, 'r') as root:
            episode_len, state_dim = root['/action'].shape

            start_ts = np.random.choice(episode_len)

            # get observation at start_ts only
            obs['qpos'] = root['/observations/qpos'][start_ts]

            if 'task' in root:
                obs['task'] = int(root['/task'][start_ts])
            else:
                obs['task'] = 0

            for sources in self.data_sources:
                obs[sources] = {}
                for cam_name in self.cam_names:
                    obs[sources][cam_name] = root[f'/observations/{sources}/{cam_name}'][start_ts]

            action = root['/action'][start_ts:episode_len].astype(float)
            action_len = episode_len - start_ts

            extra_len = self.chunck_size - action_len
            if extra_len > 0:
                action_extra = action[-1:, :].repeat(extra_len, axis=0)
                action_extra = np.concatenate([action, action_extra], axis=0)
            else:
                action_extra = action[:self.chunck_size, :]

            # pad = np.array([0] * action_len + [1] * extra_len)
            pad = np.zeros(self.chunck_size)

        obs['action'] = action_extra
        obs['pad'] = pad

        return obs

    def train_process(self, qpos, all_cond, task, action, pad):
        qpos = self.process_pos(qpos, infer=False)
        task = self.process_task(task, infer=False)

        # ---- hack here
        # mask = torch.ones_like(qpos)
        # mask[3:7] = 0
        # mask[17:21] = 0
        # qpos = mask * qpos
        # ---- hack here

        processed_cond = []
        for i, sources in enumerate(self.data_sources):
            cond = all_cond[i]
            processed_cond.append(self.dict[sources](cond, infer=False))

        action = self.process_action(action)
        pad = self.process_pad(pad)

        return qpos, processed_cond, task, action, pad

    # -------------------------------------------------  test  ---------------------------------------------------
    def change_format(self, obs, infer):
        qpos = np.array(obs['qpos'])
        task = np.array(obs['task'])

        all_cond = []
        for sources in self.data_sources:
            cond = []
            for cam_name in self.cam_names:
                if sources == 'voxs' and infer:
                    vox = get_occugrid(
                        obs['segment'][cam_name],
                        obs['depths'][cam_name],
                        obs['k'][cam_name],
                        obs['T'][cam_name],
                        self.vox_robot_base,
                        self.vox_dict,
                        self.voxel_size,
                        self.occugrid_ori,
                        self.occugrid_range,
                    )
                    cond.append(vox)
                else:
                    cond.append(obs[sources][cam_name])

            all_cond.append(cond)

        if infer:
            image = obs['images']
            return qpos, all_cond, task, image
        else:
            action = obs['action']
            pad = obs['pad']
            return qpos, all_cond, task, action, pad

    def infer_process(self, qpos, all_cond, task):
        qpos = self.process_pos(qpos, infer=True)
        task = self.process_task(task, infer=True)

        # ---- hack here
        # mask = torch.ones_like(qpos)
        # mask[0, 3:7] = 0
        # mask[0, 17:21] = 0
        # qpos = mask * qpos
        # --------------------

        processed_cond = []
        for i, sources in enumerate(self.data_sources):
            cond = all_cond[i]
            processed_cond.append(self.dict[sources](cond, infer=True))

        return qpos, processed_cond, task

    def to_device(self, qpos, all_cond, task, action, pad):
        qpos = qpos.to(self.device)
        task = task.to(self.device)
        action = action.to(self.device)
        pad = pad.to(self.device)
        cond_device = []
        for cond in all_cond:
            cond_device.append(cond.to(self.device))

        return qpos, cond_device, task, action, pad

    def process_pos(self, ori_poss, infer):
        poss = torch.from_numpy(ori_poss).float()
        poss = poss.unsqueeze(0).to(self.device) if infer else poss
        return poss

    def process_image(self, ori_images, infer):
        """
        ori_images list of image(h, w, c) and len n
        images torch tensor with (n, c, h, w)
        """
        images = np.stack(ori_images, axis=0)
        images = torch.from_numpy(images).float()
        images = images / 255.0

        images = torch.einsum('k h w c -> k c h w', images)

        images = images.unsqueeze(0).to(self.device) if infer else images

        return images

        # images = self.normalize(images)
        
    def process_depth(self, ori_depths, infer):
        depths = np.stack(ori_depths[:, : None], axis=0)
        depths = torch.from_numpy(depths).float()
        depths = depths / 1000.0

        depths = torch.einsum('k h w c -> k c h w', depths)

        depths = depths.unsqueeze(0).to(self.device) if infer else depths
        
        return depths

    def process_vox(self, ori_vox, infer):
        voxs = np.stack(ori_vox, axis=0)
        voxs = torch.from_numpy(voxs).float()
        voxs = voxs.unsqueeze(0).to(self.device) if infer else voxs

        return voxs

    def process_task(self, ori_task, infer):
        # task = np.stack(ori_task, axis=0)
        task = torch.from_numpy(ori_task) # .float()
        task = task.unsqueeze(0).to(self.device) if infer else task

        return task

    @staticmethod
    def process_pad(pads):
        return torch.from_numpy(pads).bool()

    @staticmethod
    def process_action(actions):
        return torch.from_numpy(actions).float()


# -------------------------------------------------------------train val------------------------------------------------
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, processor, episode_ids, dataset_dir, max_steps):
        super(EpisodicDataset).__init__()
        self.processor = processor
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.max_steps = max_steps

        self.is_sim = None

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        obs = self.processor.read_data(dataset_path, self.max_steps)  # read data

        qpos, all_cond, task, action, pad = self.processor.change_format(obs, infer=False)  # decode data
        qpos, all_cond, task, action, pad = self.processor.train_process(qpos, all_cond, task, action, pad)

        return qpos, all_cond, task, action, pad


class EpisodicFullDataset(torch.utils.data.Dataset):
    def __init__(self, processor, episode_ids, dataset_dir, max_steps, chunck_size=50):
        super(EpisodicFullDataset).__init__()
        self.processor = processor
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.max_steps = max_steps
        self.chunck_size = chunck_size

        self.data_sources = self.processor.data_sources
        self.cam_names = self.processor.cam_names

        self.is_sim = None

        self.datas, self.data_lens = self.processor.read_data_all(self.dataset_dir, episode_ids, max_steps, chunck_size)
        print('-------------------------------')

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        data = self.datas[index]
        episode_len = self.data_lens[index]

        ind = np.random.choice(episode_len)

        obs = data[ind]

        """
        for sources in self.data_sources:
            obs[sources] = {}
            for cam_name in self.cam_names:
                obs[sources][cam_name] = data[sources][cam_name][ind]

        obs['action'] = data['action'][ind:ind + self.chunck_size]
        obs['pad'] = data['pad'][ind:ind + self.chunck_size]"""

        # obs = self.processor.read_data(dataset_path, self.max_steps)  # read data
        qpos, all_cond, action, pad = self.processor.change_format(obs, infer=False)  # decode data
        qpos, all_cond, action, pad = self.processor.train_process(qpos, all_cond, action, pad)

        return qpos, all_cond, action, pad


def save_norm_states(dataset_dir, num_episodes, ckpt_dir):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]

        all_qpos_data.append(torch.from_numpy(qpos.astype('float32')))
        all_action_data.append(torch.from_numpy(action.astype('float32')))

    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)


def build_loader(processor, dataset_dir, num_episodes, max_steps, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    train_ratio = 0.8

    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(processor, train_indices, dataset_dir, max_steps)
    val_dataset = EpisodicDataset(processor, val_indices, dataset_dir, max_steps)

    # train_dataset = EpisodicFullDataset(processor, train_indices, dataset_dir, max_steps)
    # val_dataset = EpisodicFullDataset(processor, val_indices, dataset_dir, max_steps)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=False,
        num_workers=1,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
    )

    return train_dataloader, val_dataloader, train_dataset.is_sim


def cluster(u, v):
    points = np.stack([u, v], 1)
    core, ind = dbscan(points, eps=2, min_samples=10)
    u = u[ind >= 0]
    v = v[ind >= 0]

    return u, v


def depth2xyz_label(depth, K, labels):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth

    label_2d = labels.reshape(z.shape[0], z.shape[1])
    index = label_2d * z > 0

    # index = z > 0

    x = ((w - cx) * z / fx)[index]
    y = ((h - cy) * z / fy)[index]
    z = z[index]
    labels = labels[index.ravel()]
    xyz = np.dstack((x, y, z)).reshape(-1, 3)

    return xyz, labels


def get_occugrid(seg, depth, K, T, robot_base, trans_dict, vs, occ_ori, occ_range):
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    shape = seg.shape
    h, w = shape[0], shape[1]

    voxel_size = np.array(vs)
    occugrid_ori = np.array(occ_ori)
    occugrid_range = np.array(occ_range)

    labels_new = np.zeros(h * w)

    for key, value in trans_dict.items():
        obj_seg = np.any(seg[..., None] == value, 2)
        v, u = np.where(obj_seg)

        if len(v) < 10:
            continue

        u, v = cluster(u, v)
        labels_new[v * w + u] = key

    points, labels = depth2xyz_label(depth, K, labels_new)

    if not robot_base:
        T = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

    # print(T)

    # q = [-0.398, 0.000, -0.000, 0.917]
    # trans = [0.012, -0.458, 0.204]

    # T_new = get_T(q, trans)
    # print('new:')
    # print(T_new)

    points_world = points@T[:3, :3].T + T[:3, 3]

    points_grid = (points_world - occugrid_ori) // voxel_size

    occusize = (occugrid_range / voxel_size).astype(int)

    index_x = np.logical_and(points_grid[:, 0] < occusize[0], points_grid[:, 0] >= 0)
    index_y = np.logical_and(points_grid[:, 1] < occusize[1], points_grid[:, 1] >= 0)
    index_z = np.logical_and(points_grid[:, 2] < occusize[2], points_grid[:, 2] >= 0)
    index = index_x & index_y & index_z

    points_grid = points_grid[index].astype(int)
    labels = labels[index]

    occugrid = np.zeros(occusize)
    for k, v in trans_dict.items():
        x, y, z = np.unique(points_grid[labels == k], axis=0).T
        occugrid[x, y, z] = k

    return occugrid


def get_T(q, t):
    from scipy.spatial.transform import Rotation as R
    rotate_mat = R.from_quat(q).as_matrix()

    T0 = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ])

    T1 = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0]
    ])

    b1 = np.array([-0.0027, -0.0008, 1.068])

    R = T0 @ rotate_mat
    t = T1 @ t + b1

    T = np.concatenate([R, t[:, None]], axis=1)
    T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

    return T

