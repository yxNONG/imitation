import torch
import numpy as np
import os
import copy
import torch.nn as nn

import matplotlib.pyplot as plt
import time
import argparse
import h5py

from conf.task import SIM_TASK_CONFIGS
from Env.rdk.rdk import make_rdk
from scripted_rdk import gen_grasp_policy, gen_transfer_policy, gen_multi_grasp_policy, gen_grasp_real_policy, gen_grasp_test_policy
import Env.rdk_ee.GrabSim_pb2 as GrabSim_pb2


def make_policy(task_name):
    if task_name == 'rdk_grasp':
        policy = gen_grasp_policy
    elif task_name == 'rdk_grasp_new':
        policy = gen_grasp_policy
    elif task_name == 'rdk_transfer':
        policy = gen_transfer_policy
    elif task_name == 'rdk_grasp_real':
        policy = gen_grasp_real_policy
    elif task_name == 'rdk_multi_grasp':
        policy = gen_multi_grasp_policy
    elif task_name == 'rdk_grasp_test':
        policy = gen_grasp_test_policy

    return policy


class ResBlock(nn.Module):
    def __init__(self, c, r=2):
        super().__init__()
        self.l1 = nn.Linear(c, c // r)
        self.l2 = nn.Linear(c // r, c)

        self.act = nn.ReLU()

    def forward(self, x):
        inp = x
        x = self.act(self.l1(x))
        return self.act(self.l2(x) + inp)


class ActionNet(nn.Module):
    def __init__(self, i_dim, o_dim, f_act, c=128, nb=4):
        super().__init__()
        self.stem = nn.Linear(i_dim, c)
        self.block = []
        for n in range(nb):
            self.block.append(ResBlock(c))

        self.block = nn.ModuleList(self.block)

        self.out = nn.Linear(c, o_dim)
        self.act = nn.ReLU()
        self.f_act = f_act

    def forward(self, qpos):
        x = self.act(self.stem(qpos))
        for block in self.block:
            x = block(x)

        x = self.out(x)

        return torch.exp(-x) if self.f_act else x


class ConvResBlock(nn.Module):
    def __init__(self, c, k=1):
        super().__init__()
        self.l1 = nn.Conv1d(c, c // 2, stride=1, kernel_size=k, padding=k // 2)
        self.l2 = nn.Conv1d(c // 2, c, stride=1, kernel_size=k, padding=k // 2)

        self.act = nn.ReLU()

    def forward(self, x):
        inp = x
        x = self.act(self.l1(x))
        return self.act(self.l2(x) + inp)


class ConvActionNet(nn.Module):
    def __init__(self, i_dim, o_dim, len=7, c=16, nb=3):
        super().__init__()
        self.len = len
        self.stem = nn.Conv1d(i_dim, c, stride=1, kernel_size=1)
        self.block = []
        for n in range(nb):
            self.block.append(ConvResBlock(c))

        self.block = nn.ModuleList(self.block)

        self.out = nn.Conv1d(c, o_dim, stride=1, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, qpos):
        b, l, dim = qpos.shape
        qpos = qpos.reshape(b * l, -1)
        qpos = torch.cat((qpos[:, :self.len][:, None, :], qpos[:, self.len:][:, None, :]), dim=1)

        x = self.stem(qpos)
        for block in self.block:
            x = block(x)

        x = self.out(x)
        x = x.reshape(b, l, -1)

        return x


class Pid:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.diff = None
        self.sum_diff = None
        self.d_diff = None

    def reset(self, init):
        self.diff = np.zeros_like(np.array(init))
        self.sum_diff = np.zeros_like(self.diff)
        self.d_diff = np.zeros_like(self.diff)

    def __call__(self, qpos, now_qpos, target_qpos, update=True):
        if update:
            diff = np.array(qpos) - np.array(now_qpos)
            diff[21:] = 0

            self.sum_diff = self.sum_diff + diff
            self.d_diff = diff - self.diff
            self.diff = diff

        action = np.array(target_qpos) + self.diff * self.kp + self.sum_diff * self.ki + self.d_diff * self.kd

        return action


class Trainer:
    def __init__(self, t_type='', task_name='', data_type='', device='cuda:0', dpg=False):
        self.s = 7
        self.e = 21
        self.joint_len = self.e - self.s

        self.save_path = self.build_folder('ckpt_action/rdk/{}_{}'.format(task_name, data_type))
        self.dataset_path = self.build_folder('/home/yuxuan/workspace/mmwork/train_data/action_{}_{}/'.format(task_name, data_type))

        self.device = device
        self.t_type = t_type
        self.data_type = data_type

        self.act_pid = [0.4, 0.01, 0.3]
        self.pid = [0.3, 0.9, 0.2]
        self.sleep_time = SIM_TASK_CONFIGS[task_name]['DT_model']

        if t_type == 'save_joint':
            self.num_roll_out = 300  # 135
            self.replay_num = 1

            self.env = make_rdk(task_name)
            self.policy = make_policy(task_name)
        elif t_type == 'save':
            self.dataset_dir = SIM_TASK_CONFIGS[task_name]['dataset_dir']
            self.num_episodes = SIM_TASK_CONFIGS[task_name]['num_episodes']
            self.episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
            self.camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

            self.env = make_rdk(task_name)
            self.policy = make_policy(task_name)
        else:
            self.net = ActionNet(self.joint_len * 2, self.joint_len, f_act=False).to(device)
            # self.net = ConvActionNet(2, 1, len=self.joint_len).to(device)

            self.value_net = ActionNet(21, 1, f_act=True).to(device)

            if t_type == 'train':
                q_param_dicts = [{"params": [p for n, p in self.value_net.named_parameters() if p.requires_grad]}]
                self.value_optimizer = torch.optim.AdamW(
                    q_param_dicts,
                    lr=0.001,
                    weight_decay=1e-4
                )

                param_dicts = [{"params": [p for n, p in self.net.named_parameters() if p.requires_grad]}]
                self.optimizer = torch.optim.AdamW(
                    param_dicts,
                    lr=0.001,
                    weight_decay=1e-4
                )

                self.total_num = 300  # 1350
                self.train_bs = 256  # 256
                self.train_bn = self.total_num // self.train_bs
                self.train_num = self.train_bn * self.train_bs

                self.total_epoch = 10000

                self.best_epoch = -1
                self.best_loss = 1e6
                self.state_dict = None

                self.value_best_epoch = -1
                self.value_best_loss = 1e6
                self.value_state_dict = None

                self.dpg = dpg
                if dpg:
                    self.load_model(value=True)

                self.train_action, self.train_qpos, self.train_target, \
                    self.val_action, self.val_qpos, self.val_target = self.build_dataset()

            elif t_type == 'test':
                self.env = make_rdk(task_name)
                self.policy = make_policy(task_name)

                self.load_model()

    @staticmethod
    def build_folder(f_path):
        if not os.path.exists(f_path):
            os.mkdir(f_path)

        return f_path

    @staticmethod
    def build_pid(n=-1):
        all_params = []

        """
        kps = [0.2, 0.3]
        kis = [0.8, 0.9]
        kds = [0.1, 0.2, 0.3]
        """

        kps = [0.3]
        kis = [0.9]
        kds = [0.2]

        if n > 0:
            for i in range(n):
                pid = [
                    np.random.randint(0, 10) / 10,
                    np.random.randint(0, 10) / 10,
                    np.random.randint(0, 6) / 10,
                ]
                all_params.append(pid)
        else:
            for kp in kps:
                for ki in kis:
                    for kd in kds:
                        all_params.append([kp, ki, kd])

        return all_params

    def build_dataset(self):
        all_action, all_qpos, all_target = [], [], []
        for i in np.random.permutation(self.total_num):
            action = np.load(self.dataset_path + 'action_{}.npy'.format(i + 1))[:, self.s:self.e]
            qpos = np.load(self.dataset_path + 'qpos_{}.npy'.format(i + 1))[:, self.s:self.e]
            target = np.load(self.dataset_path + 'target_{}.npy'.format(i + 1))[:, self.s:self.e]

            now_qpos = qpos[:-1, :]
            target_qpos = qpos[1:, :]

            qpos = np.concatenate([now_qpos, target_qpos], axis=1)

            all_action.append(action[None, :, :])
            all_qpos.append(qpos[None, :, :])
            all_target.append(target[None, :, :])

        all_action = torch.from_numpy(np.concatenate(all_action, axis=0)).float().to(self.device)
        all_qpos = torch.from_numpy(np.concatenate(all_qpos, axis=0)).float().to(self.device)
        all_target = torch.from_numpy(np.concatenate(all_target, axis=0)).float().to(self.device)

        return all_action[:self.train_num], all_qpos[:self.train_num], all_target[:self.train_num], \
            all_action[self.train_num:], all_qpos[self.train_num:], all_target[self.train_num:]

    def save_model(self, value=False):
        if value:
            ckpt_path = os.path.join(self.save_path, 'value_best.pt')
            torch.save(self.value_state_dict, ckpt_path)
        else:
            ckpt_path = os.path.join(self.save_path, 'best.pt')
            torch.save(self.state_dict, ckpt_path)

    def load_model(self, value=False):
        if value:
            ckpt_path = os.path.join(self.save_path, 'value_best.pt')
            self.value_net.load_state_dict(torch.load(ckpt_path))
            self.value_net.eval()
            print('value net loaded')
        else:
            ckpt_path = os.path.join(self.save_path, 'best.pt')
            self.net.load_state_dict(torch.load(ckpt_path))
            self.net.eval()

    def block_time(self, t0):
        while True:
            t1 = time.time()
            if t1 - t0 > self.sleep_time:
                return
            time.sleep(0.001)

    def train_net(self, epoch):
        self.net.train()
        total_loss = 0
        inds = np.random.permutation(self.train_qpos.shape[0])

        for bn in range(self.train_bn):
            ind = inds[bn * self.train_bs:(bn + 1) * self.train_bs]

            qpos = self.train_qpos[ind]
            action = self.train_action[ind]
            target = self.train_target[ind]

            if self.dpg:
                inp = torch.cat([qpos[:, :, :self.joint_len], target], dim=-1)
                pred_action = self.net(inp)

                inp_gt = torch.cat([inp, action], dim=-1)
                reward = self.value_net(inp_gt)

                print('gt_reward:', reward.sum())

                inp = torch.cat([inp, pred_action], dim=-1)
                reward = self.value_net(inp)
                print('reward', reward.sum())

                # exit(0)

                loss = - reward.sum()
                print('los:', loss)

            else:
                pred_action = self.net(qpos)
                loss = (pred_action - action) ** 2
                loss = torch.sum(loss, dim=[1, 2]).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.train_bn

    def eval_net(self, epoch):
        self.net.eval()
        with torch.no_grad():
            if self.dpg:
                inp = torch.cat([self.val_qpos[:, :, :7], self.val_target], dim=-1)
                pred_action = self.net(inp)
                inp = torch.cat([inp, pred_action], dim=-1)
                reward = self.value_net(inp)
                loss = - reward.sum()
            else:
                pred_action = self.net(self.val_qpos)
                loss = (pred_action - self.val_action) ** 2
                loss = torch.sum(loss, dim=[1, 2]).mean()

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_epoch = epoch
            self.state_dict = self.net.state_dict()

        return loss.item()

    def train_value_net(self, epoch):
        self.value_net.train()
        total_loss = 0
        inds = np.random.permutation(self.train_qpos.shape[0])

        for bn in range(self.train_bn):
            ind = inds[bn * self.train_bs:(bn + 1) * self.train_bs]

            qpos = self.train_qpos[ind]
            action = self.train_action[ind]
            target = self.train_target[ind]

            target_r = - torch.sum(torch.abs(target - qpos[:, :, 7:]), dim=-1)
            inp = torch.cat([qpos[:, :, :7], target, action], dim=-1)

            r = self.value_net(inp)[:, :, 0]

            loss = ((target_r - r) ** 2).mean()

            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()

            total_loss += loss.item()

        print('train epoch {}, loss: {}'.format(epoch, total_loss / self.train_bn))

    def eval_value_net(self, epoch):
        self.value_net.eval()
        with torch.no_grad():
            target_r = - torch.sum(torch.abs(self.val_target - self.val_qpos[:, :, 7:]), dim=-1)
            inp = torch.cat([self.val_qpos[:, :, :7], self.val_target, self.val_action], dim=-1)

            r = self.value_net(inp)[:, :, 0]
            loss = ((target_r - r) ** 2).mean()

        if loss.item() < self.value_best_loss:
            self.value_best_loss = loss.item()
            self.value_best_epoch = epoch
            self.value_state_dict = self.value_net.state_dict()

        print('val epoch {}, loss: {}'.format(epoch, loss.item()))
        print('best epoch {}, loss:{}'.format(self.value_best_epoch, self.value_best_loss))

    def train(self):
        for epoch in range(self.total_epoch):
            train_loss = self.train_net(epoch)
            eval_loss = self.eval_net(epoch)

            print('epoch: {}\n'
                  'train_loss: {}\n'
                  'val_loss: {}\n'
                  'best epoch {}, best_loss:{}\n'.format(
                epoch, train_loss, eval_loss, self.best_epoch, self.best_loss),
                end='\r'
            )
        self.save_model()

    def train_qnet(self):
        for epoch in range(self.total_epoch):
            self.train_value_net(epoch)
            self.eval_value_net(epoch)
        self.save_model(value=True)

    def ee_play(self):
        _ = self.env.reset(activate=True, random=True)
        x, y, z, yaw = self.env.get_target()
        all_actions = self.policy(x, y, z, yaw)

        all_reward = []
        all_joint_actions = []

        for i in range(len(all_actions)):
            action = all_actions[i]

            ts = self.env.mocap_step(**action)
            all_reward.append(ts.reward)

            # set target
            qpos = ts.observation['qpos']

            # left
            qpos[24:28] = action['fingers'][3:7]
            qpos[21] = action['fingers'][2]

            # right
            qpos[31:35] = action['fingers'][10:14]
            qpos[28] = action['fingers'][9]

            all_joint_actions.append(np.array(qpos))

        max_reward = np.max(all_reward)
        success = 1 if max_reward == self.env.max_reward else 0

        return all_joint_actions, success

    def replay_pid(self, all_target_qpos, pid, use_action=True, visual=False):
        kp, ki, kd = pid
        ts = self.env.reset(activate=False, random=False)

        episodes = [ts]
        all_qpos = [ts.observation['qpos']]
        all_actions = []

        pid_net = Pid(kp, ki, kd)
        pid_net.reset(ts.observation['qpos'])

        if use_action:
            qpos = all_target_qpos[0]
            qpos[5] = 35 / 180 * np.pi
        else:
            qpos = all_qpos[0]

        for i, target_qpos in enumerate(all_target_qpos):
            t0 = time.time()

            target_qpos[5] = 35 / 180 * np.pi

            action = pid_net(qpos, all_qpos[-1], copy.deepcopy(target_qpos), update=(i != 0))

            self.block_time(t0)

            # ts = self.env.step(target_qpos)
            ts = self.env.step(action)

            episodes.append(ts)
            all_actions.append(action)
            all_qpos.append(ts.observation['qpos'])

            if use_action:
                qpos = action * 1
            else:
                qpos = target_qpos

        target_qpos = np.vstack(all_target_qpos)
        all_actions = np.vstack(all_actions)
        all_qpos = np.vstack(all_qpos)

        dif = target_qpos[:, self.s:self.e] - all_qpos[1:, self.s:self.e]
        all_dif = np.abs(dif)
        max_dif = np.max(all_dif, axis=0)
        mean_dif = np.mean(all_dif, axis=0)

        if use_action:
            print('-- replay pid: {} use action replay'.format(pid))
        else:
            print('-- replay pid: {} not use action replay'.format(pid))
        print('max_dif:  {}\nmean_dif: {}'.format(
            round(max_dif.max(), 6),
            round(mean_dif.mean(), 6)
        ))
        print('each joint max dif:\n', max_dif)
        print('each joint mean dif:\n', mean_dif)

        if visual:
            self.visual_replay(dif, save_name='vis_pid/{}_{}_{}_{}.png'.format(use_action, kp, ki, kd))

        max_reward = np.max([ts.reward for ts in episodes[1:]])
        success = 1 if max_reward == self.env.max_reward else 0

        return episodes, all_actions, all_qpos, success

    def replay_random(self, all_target_qpos, visual=False):
        ts = self.env.reset(activate=False, random=False)

        episodes = [ts]
        all_qpos = [ts.observation['qpos']]
        all_actions = []

        for i, target_qpos in enumerate(all_target_qpos):
            t0 = time.time()

            target_qpos[5] = 35 / 180 * np.pi
            noise = np.random.randn(target_qpos.shape[0]) * 0.01

            action = target_qpos + noise

            self.block_time(t0)
            ts = self.env.step(action)

            episodes.append(ts)
            all_actions.append(action)
            all_qpos.append(ts.observation['qpos'])

        target_qpos = np.vstack(all_target_qpos)
        all_actions = np.vstack(all_actions)
        all_qpos = np.vstack(all_qpos)

        dif = target_qpos[:, self.s:self.e] - all_qpos[1:, self.s:self.e]
        all_dif = np.abs(dif)
        max_dif = np.max(all_dif, axis=0)
        mean_dif = np.mean(all_dif, axis=0)

        print('-- noise replay')
        print('max_dif:  {}\nmean_dif: {}'.format(
            round(max_dif.max(), 6),
            round(mean_dif.mean(), 6)
        ))
        print('each joint max dif:\n', max_dif)
        print('each joint mean dif:\n', mean_dif)

        if visual:
            self.visual_replay(dif, save_name='vis_pid/random.png')

        return episodes, all_actions, all_qpos

    def replay_model(self, all_target_qpos, visual=False):
        ts = self.env.reset(activate=False, random=False)

        episodes = [ts]
        all_qpos = [ts.observation['qpos']]
        all_actions = []

        for i, target_qpos in enumerate(all_target_qpos):
            t0 = time.time()

            action = copy.deepcopy(target_qpos)
            action[5] = 35 / 180 * np.pi

            qpos = torch.from_numpy(np.concatenate([all_qpos[-1][self.s:self.e], target_qpos[self.s:self.e]], axis=0)).float().to(self.device)
            with torch.no_grad():
                pred_action = self.net(qpos).cpu().numpy()
            action[self.s:self.e] = pred_action

            self.block_time(t0)

            ts = self.env.step(action)

            episodes.append(ts)
            all_actions.append(action)
            all_qpos.append(ts.observation['qpos'])

        target_qpos = np.vstack(all_target_qpos)
        all_actions = np.vstack(all_actions)
        all_qpos = np.vstack(all_qpos)

        dif = target_qpos[:, self.s:self.e] - all_qpos[1:, self.s:self.e]
        all_dif = np.abs(dif)
        max_dif = np.max(all_dif, axis=0)
        mean_dif = np.mean(all_dif, axis=0)

        print('-- model replay:')
        print('max_dif:  {}\nmean_dif: {}'.format(
            round(max_dif.max(), 6),
            round(mean_dif.mean(), 6)
        ))
        print('each joint max dif:\n', max_dif)
        print('each joint mean dif:\n', mean_dif)

        if visual:
            self.visual_replay(dif, save_name='vis_pid/model.png')

        return episodes, all_actions, all_qpos

    @staticmethod
    def visual_replay(dif, save_name):
        c_dict = {
            1: 'r',
            2: 'g',
            3: 'b',
            4: 'c',
            5: 'm',
            6: 'y',
            7: 'k'
        }

        plt.figure()
        plt.title('example')
        plt.ylim((-0.045, 0.045))
        # plt.xlim((0, 120))
        for i in range(7):
            x, = plt.plot(dif[:, i], c_dict[i + 1])

        plt.savefig(save_name)
        plt.close()

    def test(self):
        _ = self.env.reset(activate=True, random=True)
        x, y, z, yaw = self.env.get_target()
        all_actions = self.policy(x, y, z, yaw)

        all_target_qpos, _ = self.ee_play(all_actions)

        # all_pids = self.build_pid()


        # ori
        # self.replay_pid(all_target_qpos, [0, 0, 0], use_action=True, visual=True)

        # pid old
        # self.replay_pid(all_target_qpos, self.act_pid, use_action=True, visual=True)

        # pid
        self.replay_pid(all_target_qpos, [0.2, 0.5, 0.3], use_action=False, visual=True)
        self.replay_pid(all_target_qpos, [0.2, 0.6, 0.3], use_action=False, visual=True)
        self.replay_pid(all_target_qpos, [0.2, 0.7, 0.3], use_action=False, visual=True)

        # model
        # self.replay_model(all_target_qpos, visual=True)

    def save_joint(self):
        count = 0
        if self.data_type == 'pid':
            all_pids = self.build_pid()
        for n in range(self.num_roll_out):
            if self.data_type == 'randpid' or self.data_type == 'noise':
                all_pids = self.build_pid(self.replay_num)

            _ = self.env.reset(activate=True, random=True)
            x, y, z, yaw = self.env.get_target()
            all_actions = self.policy(x, y, z, yaw)

            all_target_qpos, all_reward = self.ee_play(all_actions)

            for pid in all_pids:
                if self.data_type in ['randpid', 'pid']:
                    print(count, pid)
                    _, all_actions, all_qpos = self.replay_pid(all_target_qpos, pid, use_action=False)
                else:
                    print(count)
                    _, all_actions, all_qpos = self.replay_random(all_target_qpos)

                count += 1
                np.save(self.dataset_path + '/action_{}.npy'.format(count), all_actions)
                np.save(self.dataset_path + '/qpos_{}.npy'.format(count), all_qpos)
                np.save(self.dataset_path + '/target_{}'.format(count), np.vstack(all_target_qpos))

    def save_data(self, joint_traj, episode_replay, episode_idx, dim, save_vox=False):
        data_dict = {
            '/observations/qpos': [],
            '/action': [],
            '/task': [],
        }
        for cam_name in self.camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
            if save_vox:
                data_dict[f'/observations/voxs/{cam_name}'] = []

        max_timesteps = len(joint_traj)

        # while joint_traj:
        #     action = joint_traj.pop(0)

        for action in joint_traj:
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/action'].append(action)
            data_dict['/task'].append(ts.observation['task'])

            for cam_name in self.camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                if save_vox:
                    vox = get_occugrid(
                        ts.observation['segment'][cam_name],
                        ts.observation['depths'][cam_name],
                        ts.observation['k'][cam_name],
                        ts.observation['T'][cam_name],
                        SIM_TASK_CONFIGS[self.task_name]['vox_robot_base'],
                        SIM_TASK_CONFIGS[self.task_name]['vox_dict'],
                        SIM_TASK_CONFIGS[self.task_name]['voxel_size'],
                        SIM_TASK_CONFIGS[self.task_name]['occugrid_ori'],
                        SIM_TASK_CONFIGS[self.task_name]['occugrid_range']
                    )

                    data_dict[f'/observations/voxs/{cam_name}'].append(vox)

        print('task_id:', data_dict['/task'][0])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(self.dataset_dir, 'episode_{}.hdf5'.format(episode_idx))
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            if save_vox:
                vox = obs.create_group('voxs')

            for cam_name in self.camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3),
                                         dtype='uint8', chunks=(1, 480, 640, 3),
                                         compression="gzip", compression_opts=2)
                if save_vox:
                    _ = vox.create_dataset(cam_name, (max_timesteps, 30, 80, 40),
                                           dtype='float32', chunks=(1, 30, 80, 40))
            qpos = obs.create_dataset('qpos', (max_timesteps, dim))
            action = root.create_dataset('action', (max_timesteps, dim))
            task = root.create_dataset('task', (max_timesteps, ))

            print('done')

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'dSaving: {time.time() - t0:.1f} secs\n')

    def save(self):
        ee_fail = 0
        replay_fail = 0
        episodes_idx = 0
        while episodes_idx < self.num_episodes:
            all_target_qpos, success = self.ee_play()
            if success:
                print(f"{episodes_idx} Successful")
            else:
                ee_fail += 1
                print(f"{episodes_idx} Failed")
                continue

            ts, all_actions, all_qpos, success = self.replay_pid(all_target_qpos, [0.2, 0.5, 0.3], use_action=False)
            if success:
                self.save_data(all_actions, ts, episodes_idx, dim=35)
                print(f"replay {episodes_idx} Successful")
                episodes_idx += 1
            else:
                replay_fail += 1
                print(f"replay {episodes_idx} Failed, retry")

            print(episodes_idx, ee_fail, replay_fail)

    def start(self):
        if self.t_type == 'train':
            self.train()
        elif self.t_type == 'test':
            for i in range(4):
                print('-----{}-----'.format(i))
                self.test()
        elif self.t_type == 'save_joint':
            self.save_joint()
        else:  # save data
            self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', default='rdk_multi_grasp', type=str, help='task_name')
    parser.add_argument('--type', action='store', default='train', type=str, help='task_name')
    parser.add_argument('--data_type', action='store', default='randpid', type=str, help='task_name')

    args = vars(parser.parse_args())
    print(args)

    assert args['type'] in ['save_joint', 'train', 'test', 'save']
    assert args['data_type'] in ['randpid', 'pid', 'noise']
    server = Trainer(t_type=args['type'], task_name=args['task_name'], data_type=args['data_type'])
    server.start()

    exit(0)

