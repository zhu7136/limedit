"""
翻转动作文件：将蹲下动作转为站起动作，并添加末尾保持
"""

import numpy as np
import sys
from pathlib import Path


def reverse_motion(input_path: str, output_path: str, hold_frames: int = 1000):
    """
    翻转动作文件

    Args:
        input_path: 输入NPZ文件路径
        output_path: 输出NPZ文件路径
        hold_frames: 末尾保持帧数
    """
    # 加载数据
    data = np.load(input_path, allow_pickle=True)

    # 翻转数据
    reversed_data = {
        'fps': data['fps'],
        'joint_pos': data['joint_pos'][::-1],
        'joint_vel': -data['joint_vel'][::-1],
        'body_pos_w': data['body_pos_w'][::-1],
        'body_quat_w': data['body_quat_w'][::-1],
        'body_lin_vel_w': -data['body_lin_vel_w'][::-1],
        'body_ang_vel_w': -data['body_ang_vel_w'][::-1],
    }

    # 添加末尾hold
    hold_data = {}
    for key in reversed_data.keys():
        if key == 'fps':
            hold_data[key] = reversed_data[key]
        elif 'vel' in key:
            # 速度补0
            shape = (hold_frames,) + reversed_data[key].shape[1:]
            hold_data[key] = np.zeros(shape, dtype=reversed_data[key].dtype)
        else:
            # 位置/姿态重复最后一帧
            last_frame = reversed_data[key][-1:]
            hold_data[key] = np.repeat(last_frame, hold_frames, axis=0)

    # 合并数据
    final_data = {}
    for key in reversed_data.keys():
        if key == 'fps':
            final_data[key] = reversed_data[key]
        else:
            final_data[key] = np.concatenate([reversed_data[key], hold_data[key]], axis=0)

    # 保存
    np.savez(output_path, **final_data)

    # 打印信息
    fps = final_data['fps'][0]
    total_frames = len(final_data['joint_pos'])
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"原始帧数: {len(data['joint_pos'])}")
    print(f"翻转帧数: {len(reversed_data['joint_pos'])}")
    print(f"Hold帧数: {hold_frames}")
    print(f"总帧数: {total_frames}")
    print(f"总时长: {total_frames/fps:.2f}秒")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python reverse_motion.py <input.npz> [output.npz] [hold_frames]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.npz', '_reversed.npz')
    hold_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    reverse_motion(input_path, output_path, hold_frames)
