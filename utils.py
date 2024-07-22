import numpy as np
import torch
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FFMpegWriter

import cv2
import pyquaternion as pyq

import IPython
e = IPython.embed


# ------------------------------------------------------infer ----------------------------------------------------------
class InferLogger:
    def __init__(
        self,
        env_max_reward,
        chunk_size,
        draw_index,
    ):
        # total
        self.episode_returns = []
        self.highest_rewards = []
        self.env_max_reward = env_max_reward
        self.chunk_size = chunk_size
        self.draw_index = draw_index

        self.vox = True

        # epoisode
        self.image_list = []  # for visualization
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []

        self.cond_list = []

        self.count = 0
        self.success = 0

    def reset_episode(self, success):
        self.image_list = []  # for visualization
        self.qpos_list = []
        self.target_qpos_list = []
        self.rewards = []
        self.cond_list = []

        self.count += 1
        self.success += int(success)

    def show_result(self):
        print('\n------ \nsuccess rate: {} %'.format(round(self.success / self.count * 100, 3)))

    def update(self, ori_image, ori_qpos, target_qpos, reward, cond):
        self.image_list.append(ori_image)
        self.qpos_list.append(ori_qpos)
        self.target_qpos_list.append(target_qpos)
        self.rewards.append(reward)
        self.cond_list.append(cond[0][0][0].cpu().numpy())

    def update_episode(self, DT, ckpt_dir):
        rewards = np.array(self.rewards)
        episode_return = np.sum(rewards[rewards != None])
        self.episode_returns.append(episode_return)

        episode_highest_reward = np.max(rewards)
        self.highest_rewards.append(episode_highest_reward)
        print('Rollout {}\n{}, {}, {}, Success: {}'.format(
            self.count,
            episode_return,
            episode_highest_reward,
            self.env_max_reward,
            episode_highest_reward == self.env_max_reward
        ))

        save_videos(self.image_list, self.cond_list, DT,
                    video_path=os.path.join(ckpt_dir, f'video{self.count}.mp4'),
                    cat_cond=self.vox)

        draw_qpos(
            self.qpos_list,
            self.target_qpos_list,
            draw_index=self.draw_index,
            freq=self.chunk_size,
            save_path=os.path.join(ckpt_dir, f'qpos_target_plot_{self.count}.png'))

        self.reset_episode(episode_highest_reward == self.env_max_reward)

    def get_matrix(self):
        success_rate = np.mean(np.array(self.highest_rewards) == env_max_reward)
        avg_return = np.mean(self.episode_returns)
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
        for r in range(self.env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

        print(summary_str)


class ScreenRender:
    def __init__(
        self,
        on,
        cam_name,
        DT,
    ):
        self.on = on
        self.cam_name = cam_name
        self.DT = DT
        self.plt_img = None

    def init(self, env):
        if self.on:
            ax = plt.subplot()
            self.plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=self.cam_name))
            plt.ion()

    def update(self, env):
        if self.on:
            image = env._physics.render(height=480, width=640, camera_id=self.cam_name)
            self.plt_img.set_data(image)
            plt.pause(self.DT)

    def end(self):
        plt.close()
        self.plt_img = None


# ------------------------------------------------ env utils  ---------------------------------------------------------
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# ---------------------------------------------------- helper---------------------------------------------------------
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_pose(x_range, y_range, z_range):
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([0, 0, 0, 0])

    return np.concatenate([cube_position, cube_quat])


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def save_videos(video, cond, dt, video_path=None, cat_cond=False):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        w_len = len(cam_names) if not cat_cond else len(cam_names) + 1
        h, w, _ = video[0][cam_names[0]].shape
        v_w = w * w_len
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (v_w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                if cam_name == 'screen':
                    image = cv2.resize(image, (w, h))

                images.append(image)

            if cat_cond:
                vox = get_vox_cv(cond[ts])
                images.append(vox)

            images = np.concatenate(images, axis=1)
            cv2.imwrite('frame.jpg', images)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')


    """
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        # cam_names.pop(0)
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')"""

"""
def save_vox(vox_list, dt, video_path):
    c_dict = {
        1: 'b',
        2: 'r',
        3: 'g',
    }

    fps = int(1 / dt)
    writer = FFMpegWriter(fps=fps)

    fig = plt.figure()

    with writer.saving(fig, video_path, 120):
        for i, occ in enumerate(vox_list):
            occ = occ[0, 0].cpu().numpy()
            print(i)
            # s = np.sum(occ == 1)
            # d = np.sum(occ == 2)
            # h = np.sum(occ == 3)

            # print('i:{}, s:{}, d:{}, h:{}'.format(i, s, d, h))

            # fig = plt.figure()
            ax3d = plt.axes(projection='3d')
            ax3d.set_xlim(0, 30)
            ax3d.set_ylim(0, 80)
            ax3d.set_zlim(0, 40)

            for j in [1, 2, 3]:
                x, y, z = np.where(occ == j)
                c = c_dict[j]
                ax3d.scatter3D(x, y, z, cmap=c)

            plt.draw()
            writer.grab_frame()
            plt.pause(0.01)"""


def get_vox_cv(occ):
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

    for j in [1, 2, 3]:
        x, y, z = np.where(occ == j)
        c = c_dict[j]
        ax3d.scatter3D(x, y, z, cmap=c)

    plt.draw()
    plt.savefig('temp.jpg')

    img = cv2.imread('temp.jpg')

    plt.close()

    return img


def draw_qpos(qpos_list, target_qpos_list, draw_index, freq, save_path):
    states = np.vstack(qpos_list)
    preds = np.vstack(target_qpos_list)

    total_time = states.shape[0]
    s_index, e_index = draw_index

    states = states[:, s_index: e_index]
    preds = preds[:, s_index: e_index]

    c_dict = {
        1: 'r',
        2: 'g',
        3: 'b',
        4: 'c',
        5: 'm',
        6: 'y',
        7: 'k'
    }

    plt.title('example')
    for i in range((e_index - s_index)):
        x, = plt.plot(preds[:, i], c_dict[i + 1])
        x, = plt.plot(states[:, i], c_dict[i + 1], linestyle=':')
        # x, = plt.plot(gt_preds[:, i], c_dict[i + 1], linestyle='--')
        # x, = plt.plot(gt_states[:, i], c_dict[i + 1], linestyle='-.')

    for i in range(total_time // freq):
        plt.axvline(i * freq, linestyle=':', linewidth=1)
        plt.axvline(i * freq + freq - 1, linestyle=':', linewidth=1)
    # plt.show()
    plt.savefig(save_path)

