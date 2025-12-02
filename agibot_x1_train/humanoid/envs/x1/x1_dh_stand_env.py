# ============================================================
# x1_dh_stand_env.py (Enhanced wrapper, final)
# - Inherits original implementation in x1_dh_stand_env_orig.py
# - Adds: noise vector, state machine, idle behavior, comprehensive reward set
# ============================================================

import torch
import numpy as np

# Import original implementation (must exist as x1_dh_stand_env_orig.py in same package)
from humanoid.envs.x1.x1_dh_stand_env_orig import X1DHStandEnv as _OrigX1DHStandEnv

class X1DHStandEnv(_OrigX1DHStandEnv):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # call original constructor (keep original heavy logic)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # --- state machine fields ---
        self.state = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # 0 idle, 1 move
        self.idle_timer = torch.zeros(self.num_envs, device=self.device)
        self.last_command_time = torch.zeros(self.num_envs, device=self.device)

        # read config thresholds (with safe defaults)
        self.command_threshold = getattr(self.cfg.env, "command_threshold", 0.01)
        self.inactivity_threshold = getattr(self.cfg.env, "inactivity_threshold", 10.0)
        self.command_timeout = getattr(self.cfg.commands, "command_timeout", 3.0)

    # --------------------------
    # Noise vector builder
    # --------------------------
    def _get_noise_scale_vec(self, cfg):
        """Construct noise scaling vector (dof_pos, dof_vel, lin_vel, ang_vel, quat, gravity)."""
        num_dofs = getattr(self, "num_actions", 12)
        noise_scales = getattr(cfg.noise, "noise_scales", None)
        if noise_scales is None:
            # fallback zero vector of reasonable length
            return torch.zeros((num_dofs*2 + 3 + 3 + 4 + 3,), device=self.device)
        try:
            dof_pos = noise_scales.dof_pos * torch.ones(num_dofs, device=self.device)
            dof_vel = noise_scales.dof_vel * torch.ones(num_dofs, device=self.device)
            lin = noise_scales.lin_vel * torch.ones(3, device=self.device)
            ang = noise_scales.ang_vel * torch.ones(3, device=self.device)
            quat = noise_scales.quat * torch.ones(4, device=self.device)
            grav = noise_scales.gravity * torch.ones(3, device=self.device)
            return torch.cat([dof_pos, dof_vel, lin, ang, quat, grav])
        except Exception:
            return torch.zeros((num_dofs*2 + 3 + 3 + 4 + 3,), device=self.device)

    # --------------------------
    # State machine: update & command processing
    # --------------------------
    def update_state(self, commands):
        """Update per-env state (idle vs move) based on commanded velocities."""
        if commands is None:
            return
        # effective command if norm of first 3 dims > threshold
        try:
            has_cmd = (torch.norm(commands[:, :3], dim=1) > self.command_threshold)
        except Exception:
            # fallback: if commands shape unexpected, treat as no command
            has_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        sim_time = self.common_step_counter * self.dt
        self.last_command_time = torch.where(has_cmd, torch.full_like(self.last_command_time, sim_time), self.last_command_time)
        self.idle_timer = torch.where(has_cmd, torch.zeros_like(self.idle_timer), self.idle_timer + self.dt)
        self.state = torch.where(has_cmd, torch.ones_like(self.state), self.state)
        self.state = torch.where(self.idle_timer > self.inactivity_threshold, torch.zeros_like(self.state), self.state)

    def process_commands(self, commands):
        """Return commands modified for idle envs (force idle default command)."""
        if commands is None:
            return commands
        idle_cmd_list = getattr(self.cfg.env, 'default_idle_command', [0.0, 0.0, 0.0])
        try:
            idle_cmd = torch.tensor(idle_cmd_list, device=self.device, dtype=commands.dtype).unsqueeze(0).expand(commands.shape[0], -1)
        except Exception:
            idle_cmd = torch.zeros((self.num_envs, min(3, commands.shape[1])), device=self.device)
        processed = commands.clone()
        idle_mask = (self.state == 0)
        if processed.shape[1] >= 3:
            processed[idle_mask, 0:3] = idle_cmd[idle_mask, 0:3]
        else:
            processed[idle_mask] = idle_cmd[idle_mask, :processed.shape[1]]
        return processed

    # --------------------------
    # Override step to inject state-machine and idle reward
    # --------------------------
    def step(self, actions):
        # update state machine using last stored self.commands (original env sets/updates commands)
        try:
            self.update_state(self.commands)
            self.commands = self.process_commands(self.commands)
        except Exception:
            # If self.commands not ready yet, ignore
            pass

        # enforce original command timeout behavior: zero commands if last_command_time too old
        try:
            sim_time = self.common_step_counter * self.dt
            timeout_mask = (sim_time - self.last_command_time) > self.command_timeout
            if timeout_mask.any():
                self.commands[timeout_mask, :3] = 0.0
        except Exception:
            pass

        # 核心修复：正确接收父类返回的5个值（obs, privileged_obs, rew, done, info）
        obs, privileged_obs, rew, done, info = super().step(actions)

        # inject idle bonus into rew if applicable
        try:
            idle_weight = float(getattr(self.cfg.rewards.scales, "idle_march", 0.0))
            idle_bonus = (self.state == 0).float() * idle_weight
            # rew might be tensor or dict; handle common case (tensor)
            if isinstance(rew, torch.Tensor):
                rew = rew + idle_bonus
            else:
                # if rew is dict or array, try to add to main reward key or ignore
                pass
        except Exception:
            pass

        # optionally attach timeouts flag into info for logging
        try:
            if getattr(self.cfg.env, 'send_timeouts', False) and isinstance(info, dict):
                sim_time = self.common_step_counter * self.dt
                info["timeouts"] = ((sim_time - self.last_command_time) > self.command_timeout)
        except Exception:
            pass

        # 核心修复：返回5个值，匹配上层reset的解包逻辑
        return obs, privileged_obs, rew, done, info

    # --------------------------
    # Reward functions: implement all names used in cfg.rewards.scales
    # Each function returns a tensor shape (num_envs,)
    # --------------------------

    def _reward_ref_joint_pos(self):
        try:
            diff = self.dof_pos - self.ref_dof_pos
            r = torch.exp(-2.0 * torch.norm(diff, dim=1))
            return r
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_clearance(self):
        try:
            target = getattr(self.cfg.rewards, "target_feet_height", 0.03)
            return -torch.norm(self.feet_height - target, dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_contact_number(self):
        try:
            # self.feet_indices assumed list of foot link indices (two feet)
            left_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 1.0).float()
            right_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 1.0).float()
            return left_contact + right_contact
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_air_time(self):
        try:
            stance_mask = self._get_stance_mask()
            # air_time as non-stance count
            return (1.0 - stance_mask).sum(dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_foot_slip(self):
        try:
            slip = torch.norm(self.foot_velocities[:, :, :2], dim=2).sum(dim=1)
            return -slip
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_distance(self):
        try:
            # encourage reasonable feet spread: measure dist between feet
            left_pos = self.feet_pos[:, 0, :2]
            right_pos = self.feet_pos[:, 1, :2]
            dist = torch.norm(left_pos - right_pos, dim=1)
            target_min = getattr(self.cfg.rewards, "foot_min_dist", 0.2)
            target_max = getattr(self.cfg.rewards, "foot_max_dist", 1.0)
            # reward is high when within [min, max], simple triangular shape
            reward = - (torch.clamp(dist - target_max, min=0.0) + torch.clamp(target_min - dist, min=0.0))
            return reward
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_knee_distance(self):
        try:
            # penalize knees too close/far; use joint positions indexes heuristically
            # fallback: use dof_pos indices if mapping unknown
            left_knee = self.dof_pos[:, 3] if self.num_actions >= 4 else torch.zeros(self.num_envs, device=self.device)
            right_knee = self.dof_pos[:, 9] if self.num_actions > 9 else torch.zeros(self.num_envs, device=self.device)
            return -torch.abs(left_knee - right_knee)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_contact_forces(self):
        try:
            forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=2).sum(dim=1)
            return -forces
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_tracking_lin_vel(self):
        try:
            vel_err = self.commands[:, 0:2] - self.root_states[:, 7:9]
            sigma = getattr(self.cfg.rewards, "tracking_sigma", 5.0)
            return torch.exp(-sigma * torch.norm(vel_err, dim=1))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_tracking_ang_vel(self):
        try:
            yaw_err = self.commands[:, 2] - self.root_states[:, 12]
            sigma = getattr(self.cfg.rewards, "tracking_sigma", 5.0)
            return torch.exp(-sigma * torch.abs(yaw_err))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_vel_mismatch_exp(self):
        try:
            vel_err = self.commands[:, :2] - self.root_states[:, 7:9]
            return torch.exp(-torch.norm(vel_err, dim=1))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_low_speed(self):
        try:
            speed = torch.norm(self.root_states[:, 7:9], dim=1)
            return - (speed < 0.05).float()
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_track_vel_hard(self):
        try:
            vel_err = torch.norm(self.commands[:, :2] - self.root_states[:, 7:9], dim=1)
            return -vel_err
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_default_joint_pos(self):
        try:
            diff = self.dof_pos - self.default_dof_pos
            return torch.exp(-5.0 * torch.norm(diff, dim=1))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_orientation(self):
        try:
            rpy = getattr(self, "base_euler_xyz", None)
            if rpy is None:
                # compute or fallback to zeros
                rpy = torch.zeros((self.num_envs, 3), device=self.device)
            return -torch.norm(rpy[:, :2], dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_rotation(self):
        try:
            # small penalty for large ankle rotations (approx using dof positions)
            left = self.dof_pos[:, 4:6] if self.num_actions >= 6 else torch.zeros((self.num_envs, 2), device=self.device)
            right = self.dof_pos[:, 10:12] if self.num_actions >= 12 else torch.zeros((self.num_envs, 2), device=self.device)
            return - (torch.norm(left, dim=1) + torch.norm(right, dim=1))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_base_height(self):
        try:
            height = self.root_states[:, 2]
            target = getattr(self.cfg.rewards, "base_height_target", 0.61)
            return -torch.abs(height - target)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_base_acc(self):
        try:
            if not hasattr(self, "last_root_vel"):
                self.last_root_vel = self.root_states[:, 7:13].clone()
            acc = (self.root_states[:, 7:13] - self.last_root_vel) / self.dt
            self.last_root_vel = self.root_states[:, 7:13].clone()
            return -torch.norm(acc, dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_action_smoothness(self):
        try:
            if not hasattr(self, "last_actions"):
                self.last_actions = torch.zeros_like(getattr(self, "actions", torch.zeros((self.num_envs, self.num_actions), device=self.device)))
            diff = self.actions - self.last_actions
            self.last_actions = self.actions.clone()
            return torch.exp(-10.0 * torch.norm(diff, dim=1))
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_torques(self):
        try:
            return -torch.sum(torch.abs(self.torques), dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_dof_vel(self):
        try:
            return -torch.norm(self.dof_vel, dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_dof_acc(self):
        try:
            if not hasattr(self, "last_dof_vel"):
                self.last_dof_vel = self.dof_vel.clone()
            acc = (self.dof_vel - self.last_dof_vel) / self.dt
            self.last_dof_vel = self.dof_vel.clone()
            return -torch.norm(acc, dim=1)
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_collision(self):
        try:
            # penalize base-link contacts (index base_link_idx expected)
            contact = (self.contact_forces[:, self.base_link_idx, 2] > 1.0).float()
            return -contact
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_stand_still(self):
        try:
            speed = torch.norm(self.root_states[:, 7:9], dim=1)
            return -(speed > 0.05).float()
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_idle_march(self):
        """Encourage light cyclic motion while idle (prevent rigid standing)."""
        try:
            idle_mask = (self.state == 0).float()
            # prefer small cyclic actions (not zero, not large)
            act_mag = torch.norm(getattr(self, "actions", torch.zeros((self.num_envs, self.num_actions), device=self.device)), dim=1)
            # reward highest for small nonzero action magnitude; gaussian-ish
            reward = torch.exp(-5.0 * (act_mag - 0.1)**2)
            return reward * idle_mask
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_dof_vel_limits(self):
        try:
            limit = getattr(self.cfg.rewards, "soft_dof_vel_limit", 0.9)
            exceed = (self.dof_vel.abs() > limit).float().sum(dim=1)
            return -exceed
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_dof_pos_limits(self):
        try:
            limit = getattr(self.cfg.rewards, "soft_dof_pos_limit", 0.98)
            exceed = (self.dof_pos.abs() > limit).float().sum(dim=1)
            return -exceed
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)

    def _reward_dof_torque_limits(self):
        try:
            limit = getattr(self.cfg.rewards, "soft_torque_limit", 0.9)
            exceed = (self.torques.abs() > limit).float().sum(dim=1)
            return -exceed
        except Exception:
            return torch.zeros(self.num_envs, device=self.device)
            
            
    def _reward_standup(self):
        base_height = self.root_states[:, 2]
        height_reward = torch.clamp(base_height - 0.55, min=0.0, max=0.3)
        upright = 1.0 - torch.abs(self.projected_gravity[:, 0])
        upright_reward = torch.clamp(upright, min=0.0)
        return height_reward + 0.5 * upright_reward
    
    def _reward_only_feet_contact(self):
        contact = torch.norm(self.contact_forces, dim=2) > 1.0
        contact = contact.float()
        feet_mask = torch.zeros(self.num_bodies, device=self.device)
        feet_mask[self.feet_indices] = 1.0
        non_feet_mask = 1.0 - feet_mask
        non_feet_contacts = contact * non_feet_mask
        penalty = torch.sum(non_feet_contacts, dim=1)
        return -penalty * 2.0
    
    def _reward_min_base_height(self):
        base_h = self.root_states[:, 2]
        return torch.clamp(base_h - 0.35, min=0.0)
        
    
# End of file
