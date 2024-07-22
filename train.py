import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from Env import build_env
from conf.task import SIM_TASK_CONFIGS
from dataloader import save_norm_states, build_loader  # train_loader
from dataloader import FormatProcessor  # test

from utils import InferLogger, ScreenRender
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions

import IPython
e = IPython.embed


def get_vox_cv(occ, i):
    c_dict = {
        1: 'b',
        2: 'r',
        3: 'g',
    }
    fig = plt.figure()

    ax3d = plt.axes(projection='3d')
    ax3d.set_xlim(0, 30)
    ax3d.set_ylim(0, 80)
    ax3d.set_zlim(0, 40)
    print(occ.shape)
    print(occ)

    for j in [1, 2, 3]:
        x, y, z = np.where(occ == j)
        c = c_dict[j]
        ax3d.scatter3D(x, y, z, cmap=c)

    plt.draw()
    plt.savefig('vi/occ_{}.jpg'.format(i))

    # img = cv2.imread('temp.jpg')

    plt.close()

    # return img


def build_policy(p_type):
    if p_type == 'vae':
        from conf.cfg_vae import vae as param
        from Policy import VaePolicy
        Policy = VaePolicy
    elif p_type == 'vae_exp':
        from conf.cfg_vae_exp import vae_exp as param
        from Policy import VaeExpPolicy
        Policy = VaeExpPolicy
    elif p_type == 'gaussian':
        from conf.cfg_gaussian import gaussian as param
        from Policy import GaussianPolicy
        Policy = GaussianPolicy
    else:  # p_type == 'diffusion':
        from conf.cfg_diffusion import diffusion as param
        from Policy import DiffusionPolicy
        Policy = DiffusionPolicy

    return Policy, param


class TrainServer:
    def __init__(self, policy, args, test):
        # task
        args = dict(args, **SIM_TASK_CONFIGS[args['task_name']])
        args['episode_len'] += args['ex_episode_len']

        self.infer_step = args['episode_len']
        self.camera_names = args['camera_names']
        print('args', args)

        # utils
        self.seed = args['seed']
        set_seed(self.seed)
        self.device = args['device']
        self.ckpt_dir = '{}/{}/{}_{}_{}'.format(
            args['ckpt_base'],
            args['env'],
            args['policy'],
            args['task_name'],
            args['data_sources'][0]
        )
        print(self.ckpt_dir)

        save_norm_states(args['dataset_dir'], args['num_episodes'], self.ckpt_dir)

        # model
        self.policy = policy(args)
        self.policy.to(args['device'])

        self.Processor = FormatProcessor(
            args['data_sources'],
            args['camera_names'],
            args['vox_robot_base'],
            args['vox_dict'],
            args['voxel_size'],
            args['occugrid_ori'],
            args['occugrid_range'],
            args['device'],
            args['chunk_size']
        )

        if test:  # test
            self.DT = args['DT']
            self.DT_model = args['DT_model']

            self.onscreen_render = ScreenRender(
                args['onscreen_render'],
                args['onscreen_cam'],
                args['DT']
            )

            self.env, self.env_max_reward = build_env(args['env'], args['task_name'])
            self.logger = InferLogger(self.env_max_reward, args['chunk_size'], args['draw_index'])
            self.num_roll_outs = args['num_roll_outs']
            self.infer_step = args['episode_len']

            self.policy.load_model()
            self.visual_vox = True if 'voxs' in args['data_sources'] else False
        else:  # train
            self.num_epochs = args['num_epochs']
            self.save_per_epoch = args['save_per_epoch']

            # loader
            self.train_data_loader, self.val_data_loader, _ = build_loader(
                processor=self.Processor,
                dataset_dir=args['dataset_dir'],
                num_episodes=args['num_episodes'],
                max_steps=args['episode_len'],
                batch_size_train=args['batch_size_train'],
                batch_size_val=args['batch_size_val']
            )

            # log
            self.train_history = []
            self.val_history = []
            self.min_val_loss = np.inf
            self.best_epoch = -1

        self.policy.processor.load_state(self.ckpt_dir)

    @staticmethod
    def save_result():
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards))

    @staticmethod
    def print_loss(epoch_summary, train=False):
        loss = epoch_summary['loss']
        l_type = 'Train' if train else 'Val'

        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '

        print(f'{l_type} loss: {loss:.5f}')
        print(summary_string)

    def block_time(self, t0):
        while True:
            t1 = time.time()
            if t1 - t0 > self.DT_model:
                return
            time.sleep(0.001)

    def plot_history(self, num_epochs):
        for key in self.train_history[0]:
            plot_path = os.path.join(self.ckpt_dir, f'train_val_{key}_seed_{self.seed}.png')
            plt.figure()
            train_values = [summary[key].item() for summary in self.train_history]
            val_values = [summary[key].item() for summary in self.val_history]
            plt.plot(np.linspace(0, num_epochs - 1, len(self.train_history)), train_values, label='train')
            plt.plot(np.linspace(0, num_epochs - 1, len(self.val_history)), val_values, label='validation')
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f'Saved plots to {self.ckpt_dir}')

    def eval_one(self, epoch, save):
        self.policy.eval()
        epoch_dicts = []

        with torch.inference_mode():
            for batch_idx, data in enumerate(self.val_data_loader):
                qpos, all_cond, task, action, pad = data
                qpos, all_cond, task, action, pad = self.Processor.to_device(qpos, all_cond, task, action, pad)

                forward_dict = self.policy(
                    qpos=qpos,
                    all_cond=all_cond,
                    task=task,
                    actions=action,
                    is_pad=pad,
                    update=False
                )
                epoch_dicts.append(forward_dict)

        epoch_summary = compute_dict_mean(epoch_dicts)
        self.print_loss(epoch_summary, train=False)
        self.val_history.append(epoch_summary)

        # save best
        if save and (epoch_summary['loss'] < self.min_val_loss):
            self.min_val_loss = epoch_summary['loss']
            self.policy.save_model('policy_best.ckpt')
            self.best_epoch = epoch

    def train_one(self, epoch):
        self.policy.train()
        batch_idx = 0
        for batch_idx, data in enumerate(self.train_data_loader):
            qpos, all_cond, task, action, pad = data
            qpos, all_cond, task, action, pad = self.Processor.to_device(qpos, all_cond, task, action, pad)

            forward_dict = self.policy(
                qpos=qpos,
                all_cond=all_cond,
                task=task,
                actions=action,
                is_pad=pad,
                update=True
            )

            self.train_history.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(self.train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        self.print_loss(epoch_summary, train=True)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            print('\nEpoch {}'.format(epoch))

            self.eval_one(epoch, epoch > self.num_epochs / 2)
            self.train_one(epoch)

            self.policy.step()

            # self.plot_history(epoch)
            if epoch % self.save_per_epoch == 0:
                self.plot_history(epoch)
                self.policy.save_model('policy_epoch_{}_seed_{}.ckpt'.format(epoch, self.seed))

        self.plot_history(self.num_epochs - 1)

        print(f'Training finished:\nSeed {self.seed}, val loss {self.min_val_loss:.6f} at epoch {self.best_epoch}')

    def test(self, t_type):
        self.policy.set_model_type(t_type)
        for roll_out_id in range(self.num_roll_outs):
            print('inferring ... : {}'.format(roll_out_id))

            self.infer_one()
            # self.infer_one_data()
            # exit(0)
            # self.infer_temp()

        self.logger.show_result()

        # self.save_result()

    def infer_one(self):
        ts = self.env.reset()
        self.onscreen_render.init(self.env)
        with torch.inference_mode():
            for t in range(int(self.infer_step)):
                t0 = time.time()

                self.onscreen_render.update(self.env)

                ori_qpos, cond, task, image = self.Processor.change_format(ts.observation, infer=True)
                qpos, cond, task = self.Processor.infer_process(ori_qpos, cond, task)

                # inference and control
                target_qpos = self.policy.infer(qpos, cond, task, t)

                self.block_time(t0)

                ts = self.env.step(target_qpos)

                # update log
                self.logger.update(image, ori_qpos, target_qpos, ts.reward, cond)
            self.onscreen_render.end()
            torch.cuda.empty_cache()

        self.logger.update_episode(self.DT, self.ckpt_dir)

    def infer_temp(self):
        import collections
        with torch.inference_mode():
            for t in range(int(self.infer_step)):

                obs = collections.OrderedDict()

                obs['qpos'] = np.load('temp/qpos_{}.npy'.format(t))
                obs['images'] = {'head': np.load('temp/images_{}.npy'.format(t))}
                obs['depths'] = {'head': np.load('temp/depths_{}.npy'.format(t))}
                obs['segment'] = {'head': np.load('temp/segment_{}.npy'.format(t))}
                obs['T'] = {'head': np.load('temp/T_{}.npy'.format(t))}
                obs['k'] = {'head': np.load('temp/k_{}.npy'.format(t))}

                ori_qpos, cond, image = self.Processor.change_format(obs, infer=True)
                qpos, cond = self.Processor.infer_process(ori_qpos, cond)

                # inference and control
                target_qpos = self.policy.infer(qpos, cond, t)

                r_target_pos = np.load('temp/target_qpos_{}.npy'.format(t))

                print('frame:{}, difference:{}'.format(t, np.sum(target_qpos - r_target_pos)))

    def infer_one_data(self):
        import h5py
        root = h5py.File('/home/yuxuan/workspace/mmwork/train_data/grasp_cwq_new/episode_1.hdf5', 'r')
        re_qpos = root['/observations/qpos']
        action = root['/action']

        preds, states = [], []
        gt_preds, gt_states = [], []
        # ss, ee = 28, 35

        ss, ee = 14, 21
        total_time = 120
        freq = 50

        ts = self.env.reset()
        self.onscreen_render.init(self.env)
        for i in range(total_time):  # action:
            self.onscreen_render.update(self.env)

            print('---------- {}'.format(i))

            with torch.inference_mode():
                t0 = time.time()
                qpos, cond, image = self.Processor.change_format(ts.observation, infer=True)

                state = qpos[ss:ee]
                print(state)

                qpos, cond = self.Processor.infer_process(qpos, cond)

                # inference
                target_qpos = self.policy.infer(qpos, cond, i)

                pred = target_qpos[ss:ee]

            gt_pred = action[i][ss:ee]
            gt_state = re_qpos[i][ss:ee]

            states.append(state[None, :])
            preds.append(pred[None, :])
            gt_states.append(gt_state[None, :])
            gt_preds.append(gt_pred[None, :])

            print('state:', state)
            print('prd:', pred)
            print('gt_state:', gt_state)
            print('gt_pred:', gt_pred)

            # wait and control
            self.block_time(t0)
            ts = self.env.step(target_qpos)
            # ts = self.env.step(action[i])

            t1 = time.time()
            print(t1 - t0)
            # assert t1 - t0 < 0.3

            self.logger.update(image, qpos, target_qpos, ts.reward, cond)

        states = np.concatenate(states, axis=0)
        preds = np.concatenate(preds, axis=0)
        gt_states = np.concatenate(gt_states, axis=0)
        gt_preds = np.concatenate(gt_preds, axis=0)

        c_dict = {
            1: 'r',
            2: 'g',
            3: 'b',
            4: 'c',
            5: 'm',
            6: 'y',
            7: 'k'
        }

        print(states)
        print(preds)

        plt.title('example')
        for i in range(1, 7):
            x, = plt.plot(preds[:, i], c_dict[i+1])
            x, = plt.plot(states[:, i], c_dict[i+1], linestyle=':')
            x, = plt.plot(gt_preds[:, i], c_dict[i + 1], linestyle='--')
            x, = plt.plot(gt_states[:, i], c_dict[i + 1], linestyle='-.')

        for i in range(total_time // freq):
            plt.axvline(i * freq, linestyle=':', linewidth=1)
            plt.axvline(i * freq + freq - 1, linestyle=':', linewidth=1)
        plt.show()

        self.onscreen_render.end()
        torch.cuda.empty_cache()

        self.logger.update_episode(self.DT, self.ckpt_dir)
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--policy', action='store', type=str, help='policy model used', required=True)
    parser.add_argument('--type', action='store', type=str, help='policy model used',  default='torch')

    par = vars(parser.parse_args())

    policies = ['vae', 'diffusion', 'vae_exp', 'gaussian']
    test_type = ['torch', 'onnx', 'tensorrt']
    assert par['type'] in test_type, 'type must in {}'.format(test_type)
    assert par['policy'] in policies, 'policy must in {}'.format(policies)

    policy, param = build_policy(par['policy'])
    server = TrainServer(policy, param, test=par['test'])

    if par['test']:
        server.test(par['type'])
    else:
        server.train()


