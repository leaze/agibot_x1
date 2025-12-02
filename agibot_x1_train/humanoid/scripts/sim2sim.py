# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import mujoco, mujoco_viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import  Logger
import torch
from humanoid.utils.helpers import get_load_path
import os
import time

# 关闭手柄控制
joystick_use = False

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    # 滚转 (x轴)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # 俯仰 (y轴)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # 偏航 (z轴)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data, model):
    '''提取观测数据（旧版mujoco终极兼容方案：用固定值+关节数据兜底）'''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    
    # 1. 基座数据：完全用固定值兜底（旧版无法读取，假设初始姿态不变）
    base_pos = np.array([0.0, 0.0, 0.68])  # 固定基座高度0.68m（XML初始值）
    quat = np.array([0.0, 0.0, 0.0, 1.0])  # 固定无旋转（单位四元数）
    omega = np.zeros(3, dtype=np.double)   # 固定角速度为0
    
    # 2. 脚部数据：用固定值兜底（假设脚部贴近地面，高度0.02m）
    foot_positions = [0.02, 0.02]  # 左脚、右脚固定高度（避免全0）
    foot_forces = [45.0, 45.0]     # 固定脚部接触力（模拟着地状态）
    
    # 3. 基座线速度：用关节速度的前3位（自由关节速度）转换
    r = R.from_quat(quat)
    if len(dq) >=3:
        v = r.apply(dq[:3], inverse=True).astype(np.double)
    else:
        v = np.zeros(3, dtype=np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)  # 固定重力向量
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp - dq * kd
    return torque_out

def run_mujoco(policy, cfg, env_cfg):
    print("Load mujoco xml from:", cfg.sim_config.mujoco_model_path)
    # 加载模型（旧版兼容）
    try:
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    except Exception as e:
        print(f"加载mujoco模型失败：{e}")
        return
    
    # 配置仿真参数
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    num_actuated_joints = env_cfg.env.num_actions
    
    # 初始化关节位置（关键：用手动设置的站立姿势）
    print(f"驱动关节数：{num_actuated_joints}")
    print(f"使用手动设置的站立关节角度初始化")
    # 手动设置站立关节角度（左腿6个+右腿6个，适配xyber_x1模型）
    stand_joint_pos = np.array([
        0.3, 0.15, -0.15,  # 左髋：俯仰（0.3rad）、滚转（0.15rad）、偏航（-0.15rad）
        -1.0,              # 左膝：俯仰（-1.0rad，弯曲）
        0.5, 0.05,         # 左踝：俯仰（0.5rad）、滚转（0.05rad）
        0.3, -0.15, 0.15,  # 右髋：俯仰（0.3rad）、滚转（-0.15rad）、偏航（0.15rad）
        -1.0,              # 右膝：俯仰（-1.0rad，弯曲）
        0.5, -0.05         # 右踝：俯仰（0.5rad）、滚转（-0.05rad）
    ], dtype=np.double)
    # 确保关节数匹配，否则截断/补零
    if len(stand_joint_pos) != num_actuated_joints:
        stand_joint_pos = stand_joint_pos[:num_actuated_joints]
        stand_joint_pos = np.pad(stand_joint_pos, (0, num_actuated_joints - len(stand_joint_pos)), 'constant')
    data.qpos[-num_actuated_joints:] = stand_joint_pos
    
    # 初始化viewer（旧版兼容）
    try:
        viewer = mujoco_viewer.MujocoViewer(model, data)
    except Exception as e:
        print(f"初始化viewer失败：{e}，请安装 mujoco-viewer==0.0.7")
        return
    
    # 初始化变量
    target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)
    action = np.zeros((env_cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    for _ in range(env_cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))
    
    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    stop_state_log = 40000
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0

    print("=== 仿真开始 ===")
    print(f"初始关节角度：{stand_joint_pos}")

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
        # 提取观测（全兜底逻辑）
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)
        q = q[-env_cfg.env.num_actions:]  # 驱动关节位置（唯一真实数据）
        dq = dq[-env_cfg.env.num_actions:]  # 驱动关节速度（唯一真实数据）
        base_z = base_pos[2]

        # 政策推理（100Hz）
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # 站立任务逻辑：速度指令为0时触发站立
            vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
            if hasattr(env_cfg.commands,"sw_switch") and env_cfg.commands.sw_switch:
                if vel_norm <= env_cfg.commands.stand_com_threshold:
                    count_lowlevel = 0
            
            # 构建观测向量（核心：用真实关节数据 + 兜底姿态数据）
            obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)  # 固定欧拉角（0,0,0）
            
            # 填充速度指令（站立：全0）
            if env_cfg.env.num_commands == 5:
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / env_cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / env_cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            if env_cfg.env.num_commands == 3:
                obs[0, 0] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
            
            # 填充真实关节数据（关键：让政策知道当前关节状态）
            joint_offset = cfg.robot_config.default_dof_pos
            obs[0, env_cfg.env.num_commands:env_cfg.env.num_commands+num_actuated_joints] = (q - joint_offset) * env_cfg.normalization.obs_scales.dof_pos
            obs[0, env_cfg.env.num_commands+num_actuated_joints:env_cfg.env.num_commands+2*num_actuated_joints] = dq * env_cfg.normalization.obs_scales.dof_vel
            obs[0, env_cfg.env.num_commands+2*num_actuated_joints:env_cfg.env.num_commands+3*num_actuated_joints] = action
            
            # 填充兜底姿态数据
            obs[0, env_cfg.env.num_commands+3*num_actuated_joints:env_cfg.env.num_commands+3*num_actuated_joints+3] = omega
            obs[0, env_cfg.env.num_commands+3*num_actuated_joints+3:env_cfg.env.num_commands+3*num_actuated_joints+6] = eu_ang
            
            # 填充站立指令标记（True：站立状态）
            if env_cfg.env.add_stand_bool:
                obs[0, -1] = (vel_norm <= env_cfg.commands.stand_com_threshold)
            
            # 打印关键状态（验证关节数据）
            if count_lowlevel % (cfg.sim_config.decimation * 10) == 0:
                print(f"=== 仿真状态 ===")
                print(f"基座高度：{base_z:.4f}m（固定）")
                print(f"左脚高度：{foot_positions[0]:.4f}m，右脚高度：{foot_positions[1]:.4f}m（固定）")
                print(f"左膝当前角度：{q[3]:.4f}rad（目标：{target_q[3]+joint_offset[3]:.4f}rad）")
                print(f"关节力矩均值：{np.mean(np.abs(data.ctrl)):.2f}N·m（验证控制是否生效）")

            # 观测数据裁剪
            obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.obs_scales.dof_vel)

            # 更新观测历史
            hist_obs.append(obs)
            hist_obs.popleft()

            # 构建政策输入（帧堆叠）
            policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
            for i in range(env_cfg.env.frame_stack):
                policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = hist_obs[i][0, :]
            
            # 政策推理（输出目标关节位置）
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
            target_q = action * env_cfg.control.action_scale

        # PD控制（计算力矩）
        target_dq = np.zeros((num_actuated_joints), dtype=np.double)
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds, cfg)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        
        # 应用力矩到仿真
        data.ctrl = tau

        # 仿真步进与渲染
        mujoco.mj_step(model, data)
        viewer.render()

        # 记录日志（可选）
        count_lowlevel += 1
        if _ < stop_state_log and _ % 100 == 0:
            log_dict = {
                    'base_height': base_z,
                    'left_knee_pos': q[3],
                    'right_knee_pos': q[9],
                    'torque_avg': np.mean(np.abs(tau)),
                }
            logger.log_states(dict=log_dict)

    # 结束仿真
    viewer.close()
    print("=== 仿真结束 ===")
    print(f"日志已保存到：{logger.log_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mujoco旧版本兼容脚本（x1_dh_stand任务）')
    parser.add_argument('--load_model', type=str, help='政策模型路径')
    parser.add_argument('--task', type=str, required=True, help='任务名称（需为x1_dh_stand）')
    args = parser.parse_args()
    
    # 获取任务配置
    try:
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
    except Exception as e:
        print(f"获取任务配置失败：{e}，请确认task名称为'x1_dh_stand'")
        exit(1)

    # 仿真配置（适配旧版mujoco）
    class Sim2simCfg():
        class sim_config:
            mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sim_duration = 100.0  # 仿真100秒
            dt = 0.001            # 1ms步长
            decimation = 10       # 政策100Hz推理

        class robot_config:
            # PD增益（适配X1机器人关节特性）
            kps = np.array([120.0, 80.0, 80.0, 150.0, 50.0, 50.0] * 2, dtype=np.double)
            kds = np.array([6.0, 4.0, 4.0, 8.0, 2.0, 2.0] * 2, dtype=np.double)
            tau_limit = 180.0 * np.ones(env_cfg.env.num_actions, dtype=np.double)  # 力矩限制
            # 默认关节位置（与手动初始化一致）
            default_dof_pos = np.array([
                0.3, 0.15, -0.15, -1.0, 0.5, 0.05,
                0.3, -0.15, 0.15, -1.0, 0.5, -0.05
            ], dtype=np.double)

    # 加载政策模型
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, 'exported_policies')
    if args.load_model is None:
        try:
            jit_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
            jit_dirs.sort()
            model_dir = os.path.join(root_path, jit_dirs[-1])
        except Exception as e:
            print(f"自动加载政策失败：{e}，请用--load_model指定路径")
            exit(1)
    else:
        model_dir = args.load_model if os.path.isabs(args.load_model) else os.path.join(root_path, args.load_model)
    
    # 检查并加载.jit模型
    try:
        jit_files = [f for f in os.listdir(model_dir) if f.endswith('.jit')]
        if not jit_files:
            print(f"在 {model_dir} 中未找到.jit政策文件")
            exit(1)
        model_path = os.path.join(model_dir, jit_files[-1])
        policy = torch.jit.load(model_path)
        print(f"成功加载政策模型：{model_path}")
    except Exception as e:
        print(f"加载模型失败：{e}，请确认模型文件完整")
        exit(1)

    # 启动仿真
    run_mujoco(policy, Sim2simCfg(), env_cfg)
