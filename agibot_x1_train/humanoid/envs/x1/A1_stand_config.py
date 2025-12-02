from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A1DHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66        # 观测历史栈长度（66帧历史观测，提升状态感知连续性）
        short_frame_stack = 5   #简短的历史步骤9
        c_frame_stack = 3       #all histroy privileged obs num
        num_single_obs = 47     # 单帧观测维度（包含关节、速度、姿态等核心信息）
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12        # 动作维度（12个关节，对应A1机器人6条腿×2个关节/腿）
        num_envs =1024         #4096
        episode_length_s = 40   #24  给机器人足够时间完成 “平躺→起身→静止” 全过程。
        use_ref_actions = False # 关闭参考动作（避免干扰起身动作的自主学习）
        num_commands = 5       # 保持指令维度但实际值为零num_commands = 5       

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85


    class asset(LeggedRobotCfg.asset):
        file = "/home/wtbu/agibot_x1/agibot_x1_train/resources/robots/x1/urdf/A1.urdf"
        xml_file = "/home/wtbu/agibot_x1/agibot_x1_train/resources/robots/x1/mjcf/xyber_x1_flat.xml"

        name = "A1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"

        terminate_after_contacts_on = []  # 1. 关闭基座触地终止（允许起身过程触地）    
         #terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"] # 2. 基座触地时扣分（负奖励），但不终止
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        contact_force_scale = 0.01  # 3. 触地惩罚系数（可调整，建议0.01-0.1）
        #允许机器人在起身过程中基座短暂触地（不终止训练）；
        #通过负奖励惩罚持续触地行为，引导机器人尽快抬升基座。
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        #mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.8
        dynamic_friction = 0.8
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # 地形行数（层级数）
        num_cols = 20  # 地形列（类型）的数量
        max_init_terrain_level = 5  # 课程开始状态
        platform = 3.
        terrain_dict = {"flat": 0.3, 
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2, 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0, 
                        "stairs up": 0., 
                        "stairs down": 0.,
                        "discrete": 0.1, 
                        "wave": 0.0,}
        terrain_proportions = list(terrain_dict.values())
 
        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]   # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01  # 原0.02，减小噪声dof_pos = 0.02
            dof_vel = 1.0   # 原1.5，减小噪声dof_vel = 1.5 
            ang_vel = 0.2   
            lin_vel = 0.1   
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.1]#[0.0,0.0,0.7] 降低初始高度，使机器人贴近地面
        rot = [0.0, -1.0, 0.0, 1.0]# x,y,z,w [quat]
        #关节角度设计为 “自然伸展”，避免腿部蜷缩导致起身时无发力空间。
        default_joint_angles = {  # = target angles [rad] when action = 0.0
       # 左腿：髋关节后伸，膝关节微屈，避免与地面挤压
                 'left_hip_pitch_joint': -0.3,  # 髋关节后伸（负角度，远离地面）
                 'left_hip_roll_joint': 0.0,
                 'left_hip_yaw_joint': 0.0,
                 'left_knee_pitch_joint': 0.5,   # 膝关节微屈（正角度，预留起身发力空间）
                 'left_ankle_pitch_joint': -0.2, # 踝关节后伸（避免足部拖地）
                 'left_ankle_roll_joint': 0.0,
        # 右腿：与左腿对称，确保平躺姿态平衡
                 'right_hip_pitch_joint': -0.3,
                 'right_hip_roll_joint': 0.0,
                 'right_hip_yaw_joint': 0.0,
                 'right_knee_pitch_joint': 0.5,
                 'right_ankle_pitch_joint': -0.2,
                 'right_ankle_roll_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
        'hip_pitch_joint': 30, 
        'hip_roll_joint': 40,
        'hip_yaw_joint': 40,        
        'knee_pitch_joint': 180,  # 原150，提高刚度 'knee_pitch_joint': 150, 
        'ankle_pitch_joint': 45, 
        'ankle_roll_joint': 45}
        damping = {
        'hip_pitch_joint': 4.0, 
        'hip_roll_joint': 3.0,
        'hip_yaw_joint': 4.0,               
        'knee_pitch_joint': 18.0,  # 原15.0，提高阻尼 'knee_pitch_joint': 15.0, 
        'ankle_pitch_joint': 1.0, 
        'ankle_roll_joint': 1.0}

        # action scale: target angle = actionScale * action + defaultAngle
        #  动作缩放：目标角度 = 动作缩放值 × 动作 + 默认角度
        action_scale = 0.6  # 从0.5提高，允许更大的关节角度调整（便于起身动作） action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False  # 关闭摩擦随机化（避免打滑干扰起身）randomize_friction = True
        friction_range = [0.1, 1.5]  # 原[0.2, 1.3]，扩大范围 friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        # push
        push_robots = True  # 启用轻微推力，增强抗干扰能力push_robots = True
        push_interval_s = 4 # every this second, push robot
        update_step = 2000 * 24 # after this count, increase push_duration index
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25] # 训练期间增加推力持续时间
        max_push_vel_xy = 0.08  # 降低推机器人的最大速度，避免过度干扰
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3] # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]     # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]
        
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]
        
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]
        
        add_imu_lag = False
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]
        
        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 关闭难度递增（先确保能完成基础动作）#curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 25.  # 命令被更改之前的时间[s]
        gait = ["stand"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"stand": [100,100]}

        heading_command = False  # 禁用朝向指令（无需旋转）
        stand_com_threshold = 0.005  # 静止判断更严格（速度<0.005m/s视为静止）
        sw_switch = True # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [0.0, 0.0]  # x方向速度强制为0  lin_vel_x = [-0.4, 1.2] # min max [m/s] 
            lin_vel_y = [0.0, 0.0]  # y方向速度强制为0  lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # 角速度强制为0   ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [0.0, 0.0]  # 固定朝向

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61 # 目标站立高度（A1机器人标准站立高度，起身的最终高度目标）
        foot_min_dist = 0.2
        foot_max_dist = 1.0
        # 调整高度阈值以明确阶段划分，阶段划分阈值（与env.py中的状态判断对齐）
        lay_height_threshold = 0.2    # 仰卧平躺：高度<0.2m
        half_up_threshold = 0.4       # 半起身：0.2m≤高度<0.4m
        stand_threshold = 0.55        # 站立：高度≥0.55m且姿态直立
        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos
        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
        target_feet_height = 0.03 
        target_feet_height_max = 0.06
        feet_to_ankle_distance = 0.041
        cycle_time = 0.7
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(-error*sigma)
        tracking_sigma = 5 
        max_contact_force = 700  # forces above this value are penalized
        
#################################################0928     
        class scales:
            # 1. 核心阶段奖励（权重最高，引导起身流程）
            lay_to_half_up = 18.0      # 仰卧→半起身：一次性奖励，引导初始姿态转换
            half_up_to_stand = 25.0    # 半起身→站立：一次性奖励，强化站立目标
            stand_still = 30.0         # 站立后静止：持续奖励（每帧），优先维持站立状态            
            # 2. 足部与步态约束（辅助稳定站立）
            ref_joint_pos = 2.2        # 关节参考位置跟踪：辅助维持合理关节角度
            feet_clearance = 1.0       # 足部离地间隙：避免足部拖地
            feet_contact_number = 2.0  # 足部接触数量：鼓励双脚着地（稳定支撑）
            feet_air_time = 1.2        # 足部悬空时间：抑制不必要的抬腿动作
            foot_slip = -5.0           # 足部滑动：强惩罚防止打滑（影响稳定）
            feet_distance = 0.2        # 双脚间距：维持合理支撑宽度
            knee_distance = 0.2        # 膝盖间距：避免膝盖内扣/外扩过度
            # 3. 接触力约束（保护机械结构）
            feet_contact_forces = -0.01 # 足部接触力：轻微惩罚过大冲击力
            # 4. 速度跟踪（禁用移动相关，专注站立）
            tracking_lin_vel = 0.0     # 线速度跟踪：禁用（无需移动）
            tracking_ang_vel = 0.0     # 角速度跟踪：禁用（避免转动）
            vel_mismatch_exp = 0.5     # 速度不匹配：轻微惩罚垂直/旋转方向速度偏差
            low_speed = 0.0            # 低速奖励：禁用
            track_vel_hard = 0.5       # 速度硬跟踪：弱权重（仅作辅助约束）
            # 5. 姿态与高度稳定性（核心约束）
            orientation = 15.0         # 直立姿态：提高权重至15.0，强化垂直姿态保持
            feet_rotation = 0.3        # 足部旋转：轻微约束足部姿态
            base_height = 12.0         # 基座高度：提高至12.0，强化接近目标高度（0.61m）
            base_acc = 0.2             # 基座加速度：轻微惩罚剧烈晃动
            base_ang_vel = -2.0        # 基座旋转角速度：新增惩罚，抑制身体晃动（负值表示惩罚）
            # 6. 能量与动作平滑性（抑制剧烈动作）
            action_smoothness = -0.005 # 动作平滑性：惩罚力度加大，避免动作突变（从-0.002调整）
            torques = -1e-8            # 关节力矩：轻微惩罚过大扭矩（从-8e-9调整，更严格）
            dof_vel = -5e-8            # 关节速度：惩罚力度加大，避免关节快速运动（从-2e-8调整）
            dof_acc = -2e-7            # 关节加速度：惩罚力度加大，抑制关节冲击（从-1e-7调整）
            # 7. 安全约束（防止失败状态）
            collision = -60.0          # 基座触地（摔倒）：惩罚加重至-60.0，强化避障（从-50.0调整）
            low_height_move = -25.0    # 低姿态移动：惩罚加重，防止未站立时移动（从-20.0调整）
            # 8. 关节限制（保护机械结构）
            dof_vel_limits = -1.5      # 关节速度超限：惩罚力度加大（从-1调整）
            dof_pos_limits = -25.0     # 关节位置超限：惩罚加重，避免起身时关节过度运动（从-20.0调整）
            dof_torque_limits = -0.15  # 关节扭矩超限：轻微加大惩罚（从-0.1调整） 
            default_joint_pos = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class A1DHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        #for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   #long history output dim
        in_channels = A1DHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 5e-6
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        if A1DHStandCfg.terrain.measure_heights:
            lin_vel_idx = (A1DHStandCfg.env.single_num_privileged_obs + A1DHStandCfg.terrain.num_height) * (A1DHStandCfg.env.c_frame_stack - 1) + A1DHStandCfg.env.single_linvel_index
        else:
            lin_vel_idx = A1DHStandCfg.env.single_num_privileged_obs * (A1DHStandCfg.env.c_frame_stack - 1) + A1DHStandCfg.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 12000  # number of policy updates

        # logging
        save_interval = 50  # 每隔这么多迭代次数检查一次潜在的保存项
        experiment_name = 'A1_stand'
        run_name = 'A1'
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
