import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime
import pygame
from threading import Thread, Event
import time
import threading

# --- 核心修改点 1: 增大速度范围 ---
MAX_X_VEL = 1.5       # 最大前进/后退速度
MAX_Y_VEL = 0.8       # 最大左右移动速度
MAX_YAW_VEL = 1.5     # 最大转向速度
VEL_STEP = 0.05       # 速度调整步长
YAW_STEP = 0.05       # 转向调整步长

# --- 核心修改点 2: 延长超时时间 ---
CMD_TIMEOUT = 8.0     # 命令超时时间（从4秒增加到20秒）

# 其他全局常量
STANDSTILL_VEL = 0.0  # 完全静止速度
IN_PLACE_VX = 0.00    # 原地踏步基准速度（急停后使用）

# 全局控制变量（线程安全处理）
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = STANDSTILL_VEL, STANDSTILL_VEL, STANDSTILL_VEL
last_cmd_time = None  # 记录最后一次命令时间
exit_flag = False     # 全局退出标志
cmd_lock = threading.Lock()  # 命令操作锁
init_event = Event()   # 用于同步初始化的事件

def handle_keyboard_input():
    """键盘输入处理线程"""
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, exit_flag, last_cmd_time

    # 等待初始化完成
    init_event.wait()

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("X1 机器人控制 (速度范围增大)")

    # 加载字体（确保中文显示）
    try:
        font = pygame.font.SysFont(["SimHei", "WenQuanYi Micro Hei", "Heiti TC"], 20)
    except:
        font = pygame.font.SysFont(None, 20)  # 备选字体

    instructions = [
        "操作说明 (速度范围已增大):",
        "W: 前进  S: 后退",
        "A: 左移  D: 右移",
        "Q: 左转  E: 右转",
        "空格: 急停  ESC: 退出",
        f"超时时间: {CMD_TIMEOUT}秒"
    ]

    clock = pygame.time.Clock()
    while not exit_flag:
        current_time = time.time()
        # 绘制界面
        screen.fill((240, 240, 240))

        # 绘制操作说明
        for i, text in enumerate(instructions):
            text_surf = font.render(text, True, (0, 0, 0))
            screen.blit(text_surf, (20, 20 + i * 25))

        # 更新状态显示逻辑
        state = "未知状态"
        state_color = (255, 0, 0) # 默认红色
        with cmd_lock:
            time_since_last_cmd = current_time - last_cmd_time
            if time_since_last_cmd > CMD_TIMEOUT:
                state = "完全静止"
                state_color = (0, 128, 0)  # 绿色
            elif abs(x_vel_cmd - IN_PLACE_VX) < 0.001 and abs(y_vel_cmd) < 0.001 and abs(yaw_vel_cmd) < 0.001:
                state = "原地踏步"
                state_color = (255, 165, 0) # 橙色
            else:
                state = "执行命令"
                state_color = (128, 0, 0)  # 红色

            vel_text = f"当前命令: vx={x_vel_cmd:.2f}, vy={y_vel_cmd:.2f}, yaw={yaw_vel_cmd:.2f}"
            time_text = f"距离上次命令: {time_since_last_cmd:.1f}s"

        # 绘制状态信息
        state_surf = font.render(f"当前状态: {state}", True, state_color)
        screen.blit(state_surf, (20, 180))
        screen.blit(font.render(vel_text, True, (0, 0, 0)), (20, 210))
        screen.blit(font.render(time_text, True, (0, 0, 0)), (20, 240))

        pygame.display.flip()

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_flag = True
                return
            if event.type == pygame.KEYDOWN:
                with cmd_lock:
                    last_cmd_time = time.time()  # 更新命令时间

                    if event.key == pygame.K_w:
                        x_vel_cmd = min(x_vel_cmd + VEL_STEP, MAX_X_VEL)
                    elif event.key == pygame.K_s:
                        x_vel_cmd = max(x_vel_cmd - VEL_STEP, -MAX_X_VEL)
                    elif event.key == pygame.K_a:
                        y_vel_cmd = min(y_vel_cmd + VEL_STEP, MAX_Y_VEL)
                    elif event.key == pygame.K_d:
                        y_vel_cmd = max(y_vel_cmd - VEL_STEP, -MAX_Y_VEL)
                    elif event.key == pygame.K_q:
                        yaw_vel_cmd = min(yaw_vel_cmd + YAW_STEP, MAX_YAW_VEL)
                    elif event.key == pygame.K_e:
                        yaw_vel_cmd = max(yaw_vel_cmd - YAW_STEP, -MAX_YAW_VEL)
                    elif event.key == pygame.K_SPACE:
                        # 急停后回到原地踏步
                        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = IN_PLACE_VX, STANDSTILL_VEL, STANDSTILL_VEL
                    elif event.key == pygame.K_ESCAPE:
                        exit_flag = True
                        return

        clock.tick(30)  # 限制帧率
        time.sleep(0.01)

    pygame.quit()

def play(args):
    # 声明使用全局变量
    global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_cmd_time

    # 初始化 RENDER 变量以避免 UnboundLocalError
    RENDER = False
    video = None

    # 初始化全局变量
    exit_flag = False
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = STANDSTILL_VEL, STANDSTILL_VEL, STANDSTILL_VEL
    last_cmd_time = time.time()

    # 通知键盘线程初始化完成
    init_event.set()

    # 启动键盘线程
    keyboard_thread = Thread(target=handle_keyboard_input)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    print("键盘控制线程启动，等待窗口初始化...")
    time.sleep(1)

    try:
        # 加载配置
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 1
        
        # 简化配置，确保与训练时一致
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_gravity = False
        env_cfg.domain_rand.randomize_mass = False
        env_cfg.commands.heading_command = False
        env_cfg.commands.stand_com_threshold = 0.05
        env_cfg.control.decimation = 4

        # 创建环境
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
        print("环境创建完成")

        # 加载策略模型
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print(f"策略模型加载完成: {train_cfg.runner.experiment_name}")

        # 渲染设置
        RENDER = True
        if RENDER:
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 1920
            camera_properties.height = 1080
            h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
            camera_offset = gymapi.Vec3(1.5, -1.5, 0.8)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.2, 0.1, 1), np.deg2rad(130))
            actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
            body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
            env.gym.attach_camera_to_body(h1, env.envs[0], body_handle,
                                         gymapi.Transform(camera_offset, camera_rotation),
                                         gymapi.FOLLOW_POSITION)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"x1_control_{datetime.now().strftime('%H%M%S')}.mp4")
            video = cv2.VideoWriter(video_path, fourcc, 50.0, (1920, 1080))
            print(f"视频将保存至: {video_path}")

        # 主仿真循环
        obs = env.get_observations()
        step = 0
        last_step_time = time.perf_counter()

        # 初始化环境命令为完全静止
        env.commands[:, 0] = STANDSTILL_VEL
        env.commands[:, 1] = STANDSTILL_VEL
        env.commands[:, 2] = STANDSTILL_VEL

        while not exit_flag and step < 30000:
            current_time = time.perf_counter()
            elapsed = current_time - last_step_time

            # 命令超时处理
            with cmd_lock:
                if (time.time() - last_cmd_time) > CMD_TIMEOUT:
                    x_vel_cmd = STANDSTILL_VEL
                    y_vel_cmd = STANDSTILL_VEL
                    yaw_vel_cmd = STANDSTILL_VEL

                env.commands[:, 0] = x_vel_cmd
                env.commands[:, 1] = y_vel_cmd
                env.commands[:, 2] = yaw_vel_cmd

            # --- 核心修改点 3: 打印详细调试信息 ---
            if step % 20 == 0:
                print(f"\n=== Step {step} ===")
                with cmd_lock:
                    print(f"目标命令: vx={x_vel_cmd:.2f}, vy={y_vel_cmd:.2f}, yaw={yaw_vel_cmd:.2f}")
                print(f"观测值范围: [{obs.min():.4f}, {obs.max():.4f}]")
                print(f"观测值前10个: {obs[0, :10].cpu().numpy()}")

            # 生成动作
            actions = policy(obs.detach())
            
            # --- 核心修改点 3: 打印动作信息 ---
            if step % 20 == 0:
                #action_np = actions.cpu().numpy()
                action_np = actions.clone().detach().cpu().numpy()
                print(f"模型输出动作范围: [{action_np.min():.4f}, {action_np.max():.4f}]")
                print(f"动作前10个值: {action_np[0, :10]}")

            # 执行仿真步
            obs, _, _, _, _ = env.step(actions.detach())

            # 渲染
            if RENDER and video is not None:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                video.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))

            # 控制仿真速度
            sleep_time = max(0, env_cfg.sim.dt * env_cfg.control.decimation - elapsed)
            time.sleep(sleep_time)
            last_step_time = current_time

            step += 1

    except Exception as e:
        import traceback
        print(f"\n!!! 运行错误: {str(e)}")
        traceback.print_exc()
    finally:
        # 清理资源
        exit_flag = True
        if RENDER and video is not None:
            video.release()
        cv2.destroyAllWindows()
        keyboard_thread.join()
        print("\n程序已正常退出")

if __name__ == '__main__':
    args = get_args()
    play(args)
