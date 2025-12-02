# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# ...
# (header kept from original)

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

"""Configuration class for the XBotL humanoid robot."""
class X1DHStandCfg(LeggedRobotCfg):
    """Configuration class for the XBotL humanoid robot."""
    class env(LeggedRobotCfg.env):
        frame_stack = 66      # All history obs num
        short_frame_stack = 5  # Short history step
        c_frame_stack = 3      # All history privileged obs num
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # Episode length in seconds
        use_ref_actions = False
        num_commands = 5       # sin_pos cos_pos vx vy vz

        # -------------------
        # 新增 / 修改：状态与命令相关配置
        # -------------------
        # 当连续 inactivity_threshold 秒内未收到有效命令 -> 进入 idle-march 状态
        inactivity_threshold = 10.0  # 单位 s
        # 单次命令被视为"有效"的阈值（任一线速度或角速度超过该阈值）
        command_threshold = 0.01
        # 用于原地踏步（idle）的默认命令值（vx, vy, yaw）
        default_idle_command = [0.0, 0.0, 0.0]
        # send_timeouts 如果为 True，会把 timeout 信息放入 extras
        send_timeouts = True

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/A1.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'
        name = "x1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = False
        measure_heights = False
        static_friction =1.5
        dynamic_friction =1.2
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20
        num_cols = 20
        max_init_terrain_level = 5
        platform = 3.
        terrain_dict = {
            "flat": 0.0, "rough flat": 0.0, "slope up": 0.0, "slope down": 0.0,
            "rough slope up": 0.0, "rough slope down": 0.0, "stairs up": 0.,
            "stairs down": 0.0,
             "discrete": 1.0,
            "wave": 0.0,
           "gap": 0.0,
             "pit": 0.0,
             "gravel": 0.0,
            "grass": 0.0,
            "rugged_ruins": 0.0,
            "sandy": 0.0,
             "icy": 0.0,
            "narrow_passage": 0.0,
           "urban_terrain": 0.0,
            "stair_terrain": 0.0,

        }
        terrain_proportions = list(terrain_dict.values())
        rough_flat_range = [0.005, 0.01]
        slope_range = [0, 0.1]
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.1

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7]
        
        # Symmetric initial joint angles
        default_joint_angles = {
            'left_hip_pitch': -0.1,
    'left_hip_roll': 0.0,
    'left_hip_yaw': 0.0,
    'left_knee_pitch': 0.2,
    'left_ankle_pitch': -0.1,
    'left_ankle_roll': 0.0,

    'right_hip_pitch': -0.1,
    'right_hip_roll': 0.0,
    'right_hip_yaw': 0.0,
    'right_knee_pitch': 0.2,
    'right_ankle_pitch': -0.1,
    'right_ankle_roll': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        
        # Symmetric joint stiffness/damping
        stiffness = {
            'left_hip_pitch': 40, 'left_hip_roll': 50, 'left_hip_yaw': 45,
            'left_knee_pitch': 120, 'left_ankle_pitch': 45, 'left_ankle_roll': 45,
            'right_hip_pitch': 40, 'right_hip_roll': 50, 'right_hip_yaw': 45,
            'right_knee_pitch': 120, 'right_ankle_pitch': 45, 'right_ankle_roll': 45
        }
        damping = {
            'left_hip_pitch': 4, 'left_hip_roll': 4.0, 'left_hip_yaw': 5,
            'left_knee_pitch': 12, 'left_ankle_pitch': 1.0, 'left_ankle_roll': 1.0,
            'right_hip_pitch': 4, 'right_hip_roll': 4.0, 'right_hip_yaw': 5,
            'right_knee_pitch': 12, 'right_ankle_pitch': 1.0, 'right_ankle_roll': 1.0
        }

        action_scale = 0.6
        decimation = 8

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        push_robots = True
        push_interval_s = 4
        update_step = 2000 * 24
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3]
        
        # Limit lateral COM displacement to reduce right tilt
        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.03, 0.03],  # Narrow Y range
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]
        
        # Joint friction by index (1-12)
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.15]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 25.
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"]
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}

        heading_command = False
        stand_com_threshold = 0.2
        sw_switch = False
        command_timeout = 3.0  # 与状态管理的超时分开，保持原有命令超时配置

        class ranges:
            lin_vel_x = [-0.8, 1.5]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61
        foot_min_dist = 0.2
        foot_max_dist = 1.0

        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
        target_feet_height = 0.03 
        target_feet_height_max = 0.06
        feet_to_ankle_distance = 0.041
        cycle_time = 0.7
        only_positive_rewards = True
        tracking_sigma = 5 
        max_contact_force = 700
        
        class scales:
            ref_joint_pos = 2.2
            feet_clearance = 1.
            feet_contact_number = 3.0
            feet_air_time = 1.2
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            feet_contact_forces = -0.01
            tracking_lin_vel = 3.0
            tracking_ang_vel = 2.0
            vel_mismatch_exp = 0.5
            low_speed = -0.1
            track_vel_hard = 0.5
            default_joint_pos = 1.0
            orientation = 3.0  # Enhance posture stability
            feet_rotation = 0.3
            base_height = 1.5
            base_acc = 0.2
            action_smoothness = -0.001
            torques = -8e-9
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -1.
            stand_still = 5.0  # 静止状态奖励权重
            # 新增 idle_march 奖励权重 —— 当无外部命令且处于 idle 状态时鼓励踏步节奏（防止僵硬站立）
            idle_march = 3.0
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1
            # Removed roll_angle as it's not implemented
            #####新增
            standup = 5.0
            only_feet_contact = 4.0
            min_base_height = 8.0
            
    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
            # Removed roll_angle scale
        clip_observations = 100.
        clip_actions = 100.


class X1DHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64
        in_channels = X1DHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        if X1DHStandCfg.terrain.measure_heights:
            lin_vel_idx = (X1DHStandCfg.env.single_num_privileged_obs + X1DHStandCfg.terrain.num_height) * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index
        else:
            lin_vel_idx = X1DHStandCfg.env.single_num_privileged_obs * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24
        max_iterations = 20000

        save_interval = 100
        experiment_name = 'x1_dh_stand_corrected'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

