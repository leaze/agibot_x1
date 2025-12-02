# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

import math #cmj
import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        # each sub terrain length
        self.env_length = cfg.terrain_length
        # each sub terrain width
        self.env_width = cfg.terrain_width
        # each terrain type proportion
        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # self.platform is size of platform for some terrain type, like pit, gap, slope
        self.platform = cfg.platform
        # max_difficulty is based on num_rows
        # terrain difficulty is from 0 to max
        self.max_difficulty = (cfg.num_rows-1)/cfg.num_rows

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # border_size is whole terrain border
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # whole terrain cols
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # whole terrain rows
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.idx = 0
        
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()  
              
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            # i j select row col position in whole terrain
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    # choice select terrain type, difficulty select row, row increase difficulty increase
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        rought_flat_min_height = - self.cfg.rough_flat_range[0] - difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        rought_flat_max_height = self.cfg.rough_flat_range[0] + difficulty * (self.cfg.rough_flat_range[1] - self.cfg.rough_flat_range[0]) / self.max_difficulty
        slope = self.cfg.slope_range[0] + difficulty * (self.cfg.slope_range[1] - self.cfg.slope_range[0]) / self.max_difficulty
        rought_slope_min_height = - self.cfg.rough_slope_range[0] - difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        rought_slope_max_height = self.cfg.rough_slope_range[0] + difficulty * (self.cfg.rough_slope_range[1] - self.cfg.rough_slope_range[0]) / self.max_difficulty
        stair_width = self.cfg.stair_width_range[0] + difficulty * (self.cfg.stair_width_range[1] - self.cfg.stair_width_range[0]) / self.max_difficulty
        stair_height = self.cfg.stair_height_range[0] + difficulty * (self.cfg.stair_height_range[1] - self.cfg.stair_height_range[0]) / self.max_difficulty
        discrete_obstacles_height = self.cfg.discrete_height_range[0] + difficulty * (self.cfg.discrete_height_range[1] - self.cfg.discrete_height_range[0]) / self.max_difficulty

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        amplitude = 0.2 + 0.333 * difficulty
        if choice < self.proportions[0]:
            idx = 1
            return terrain
        elif choice < self.proportions[1]:
            idx = 2
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=rought_flat_min_height, 
                                                 max_height=rought_flat_max_height, 
                                                 step=0.005, 
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            idx = 4
            if choice < self.proportions[2]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform)
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=rought_slope_min_height, 
                                                 max_height=rought_slope_max_height,
                                                 step=0.005, 
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[5]:
            idx = 6
            if choice < self.proportions[4]:
                idx = 5
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform)
        elif choice < self.proportions[7]:
            idx = 8
            if choice<self.proportions[6]:
                idx = 7
                stair_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, 
                                                 step_width=stair_width, 
                                                 step_height=stair_height, 
                                                 platform_size=self.platform)
        elif choice < self.proportions[8]:
            idx = 9
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, 
                                                     discrete_obstacles_height, 
                                                     rectangle_min_size, 
                                                     rectangle_max_size, 
                                                     num_rectangles, 
                                                     platform_size=self.platform)
        elif choice < self.proportions[9]:
            idx = 10
            terrain_utils.wave_terrain(terrain, 
                                       num_waves=3, 
                                       amplitude=amplitude)
        elif choice < self.proportions[10]:
            idx = 11
            gap_terrain(terrain, 
                        gap_size=gap_size, 
                        platform_size=self.platform)
        #cmj 2025.9.10
        #else:
        elif choice < self.proportions[11]:
            idx = 12
            pit_terrain(terrain, 
                        depth=pit_depth, 
                        platform_size=self.platform)
                        
        elif choice < self.proportions[12]:
            idx = 13
             # 生成碎石地形
            gravel_terrain(terrain, difficulty)
        elif choice < self.proportions[13]:
            idx = 14
             # 生成草地地形
            grass_terrain(terrain, difficulty)
        elif choice < self.proportions[14]:
            idx = 15
             # 生成废墟
            rugged_ruins_terrain(terrain, difficulty)
        elif choice < self.proportions[15]:
            idx = 16
             # 生成沙地
            sandy_terrain(terrain, difficulty)
        elif choice < self.proportions[16]:
            idx = 17
             # 生成雪地
            icy_terrain(terrain, difficulty)   
        elif choice < self.proportions[17]:
            idx = 18
             # 生成狭窄通道
            narrow_passage_terrain(terrain, difficulty)            
        elif choice < self.proportions[18]:
            idx = 19
             # 生成街道
            urban_terrain(terrain, difficulty)  
        elif choice < self.proportions[19]:
            idx = 20
             # 生成台阶
            create_stair_terrain(terrain) 
        
                        
        self.idx = idx
        return terrain
    
    # row col select position in whole terrain
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = self.idx

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

#cmj 2025.9.10
def gravel_terrain(terrain, difficulty):
    
    # 设置石头参数
    stone_density = 0.1 + 0.2 * difficulty  
    max_height = 0.03 + 0.10 * difficulty   

    max_radius_px = max(1, int(0.15 / terrain.horizontal_scale)) 
    max_height_px = max(1, int(max_height / terrain.vertical_scale))
    terrain.height_field_raw[:, :] = 0

    num_stones = int(stone_density * terrain.length * terrain.width)
    
    # 生成随机石头
    for _ in range(num_stones):
        center_x = np.random.randint(max_radius_px, terrain.length - max_radius_px)
        center_y = np.random.randint(max_radius_px, terrain.width - max_radius_px)
        radius = np.random.randint(1, max_radius_px + 1)
        height = np.random.randint(1, max_height_px + 1)
        
        # 应用石头形状
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x_idx = center_x + dx
                y_idx = center_y + dy

                if 0 <= x_idx < terrain.length and 0 <= y_idx < terrain.width:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        attenuation = 1.0 - (distance / radius) ** 2
                        new_height = int(height * attenuation)
                        if new_height > terrain.height_field_raw[x_idx, y_idx]:
                            terrain.height_field_raw[x_idx, y_idx] = new_height

def grass_terrain(terrain, difficulty):
    
    height = 0.02 + 0.05 * difficulty  # 草丛高度随难度增加
    density = 0.3 + 0.5 * difficulty   # 草丛密度随难度增加
    # 将高度转换为像素值
    height_pixels = int(height / terrain.vertical_scale)
    
    # 创建基础平面
    terrain.height_field_raw[:, :] = 0
    
    # 随机生成草丛位置
    for i in range(terrain.length):
        for j in range(terrain.width):            
            if np.random.rand() < density:                
                grass_height = np.random.randint(1, height_pixels + 1)
               
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < terrain.length and 0 <= nj < terrain.width:                         
                            dist_factor = 1.0 - (abs(di) + abs(dj)) / 4.0
                            current_height = int(grass_height * dist_factor)                            
                            if current_height > terrain.height_field_raw[ni, nj]:
                                terrain.height_field_raw[ni, nj] = current_height

def rugged_ruins_terrain(terrain, difficulty):
    base_height = 0
    terrain.height_field_raw[:, :] = base_height
    
    max_height = int((0.1 + 0.3 * difficulty) / terrain.vertical_scale) 
    debris_density = 0.2 + 0.3 * difficulty     
    num_debris = int(debris_density * (terrain.width * terrain.length) / 50)
    
    for _ in range(num_debris):
        x = np.random.randint(0, terrain.length)
        y = np.random.randint(0, terrain.width)
        size = np.random.randint(3, 8)  # 方体尺寸
        
        height = np.random.randint(2, max_height)
        
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                nx = x + dx
                ny = y + dy
                
                if 0 <= nx < terrain.length and 0 <= ny < terrain.width:
                    edge_dist_x = min(dx + size, size - dx)
                    edge_dist_y = min(dy + size, size - dy)
                    edge_dist = min(edge_dist_x, edge_dist_y)
                    
                    if edge_dist <= 2:
                        height_factor = edge_dist / 2.0 
                    else:
                        height_factor = 1.0  

                    terrain.height_field_raw[nx, ny] += int(height * height_factor)

def sandy_terrain(terrain, difficulty):

    dune_height = 0.3 + 0.2 * difficulty  # 沙丘高度
    dune_frequency = 0.08 - 0.03 * difficulty  # 难度高时沙丘更大
    dune_height_px = int(dune_height / terrain.vertical_scale)  # 米转像素
    
    scale = dune_frequency
    octaves = 6
    lacunarity = 2.0
    persistence = 0.5
    
    base_height = 0
    terrain.height_field_raw[:, :] = base_height
    
    x = np.arange(0, terrain.length)  
    y = np.arange(0, terrain.width)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    height_map = np.zeros((terrain.length, terrain.width))
    amplitude = 1.0
    frequency = scale
    
    for octave in range(octaves):
        wave_x = np.sin(xx * frequency)
        wave_y = np.cos(yy * frequency * 0.7)
        wave_base = (wave_x + wave_y) / 2.0
        
        wave_detail = np.sin(xx * frequency * 3.0) * np.cos(yy * frequency * 3.0)

        wave_texture = np.sin(xx * frequency * 8.0) * np.sin(yy * frequency * 8.0)
        wave_val = wave_base * 0.7 + wave_detail * 0.2 + wave_texture * 0.1
        random_val = np.random.uniform(-0.1, 0.1, size=(terrain.length, terrain.width))
        wave_val += random_val
        height_map += amplitude * wave_val
        frequency *= lacunarity
        amplitude *= persistence
    min_val = np.min(height_map)
    max_val = np.max(height_map)
    if max_val > min_val:
        height_map = (height_map - min_val) / (max_val - min_val)

    for i in range(terrain.length):
        for j in range(terrain.width):
            height = int(dune_height_px * height_map[i, j])
            terrain.height_field_raw[i, j] = base_height + max(0, height)
    
def icy_terrain(terrain, difficulty):

    # 1. 基本参数设置
    base_height = 0.0  # 基础高度（米）
    snow_depth = 0.1 + difficulty * 0.3  # 积雪深度（0.1-0.4米）
    ice_thickness = 0.02 + difficulty * 0.05  # 冰层厚度（0.02-0.07米）

    base_height_grid = int(base_height / terrain.vertical_scale)
    snow_depth_grid = int(snow_depth / terrain.vertical_scale)
    ice_thickness_grid = int(ice_thickness / terrain.vertical_scale)
    snow_height = base_height_grid + snow_depth_grid
    terrain.height_field_raw[:, :] = snow_height
    
    for i in range(terrain.length):
        for j in range(terrain.width):
            noise_val = 0.5 * (np.sin(i * 0.1) * np.cos(j * 0.1) + 
                              np.sin(i * 0.03) * np.cos(j * 0.03))
            height_variation = int(noise_val * snow_depth_grid * 0.5)
            terrain.height_field_raw[i, j] += height_variation
    ice_height = snow_depth_grid + ice_thickness_grid
    terrain.height_field_raw[:, :] += ice_thickness_grid
 
 
def narrow_passage_terrain(terrain, difficulty):

    # 基本参数设置
    wall_height = 2.0  # 墙壁高度（米）
    path_height = 0.0  # 通道高度（米）
    

    wall_height_grid = int(wall_height / terrain.vertical_scale)
    path_height_grid = int(path_height / terrain.vertical_scale)
    
    # 初始化整个地形为墙壁高度
    terrain.height_field_raw[:, :] = wall_height_grid
    
    grid_size = 3  # 网格单元数量
    cell_size = min(terrain.length, terrain.width) // grid_size
    
    min_width = 1  # 最小通道宽度（像素）
    max_width = cell_size // 3  # 最大通道宽度（像素）
    path_width = int(min_width + (max_width - min_width) * (1 - difficulty))
    
    # 创建水平和垂直通道
    for i in range(grid_size + 1):
        # 水平通道
        y_pos = i * cell_size
        start_x = 0
        end_x = terrain.length
        start_y = max(0, y_pos - path_width // 2)
        end_y = min(terrain.width, y_pos + path_width // 2)
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = path_height_grid
        
        # 垂直通道
        x_pos = i * cell_size
        start_x = max(0, x_pos - path_width // 2)
        end_x = min(terrain.length, x_pos + path_width // 2)
        start_y = 0
        end_y = terrain.width
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = path_height_grid  

def urban_terrain(terrain, difficulty):
   
    # 1. 参数设置
    road_height = 0.0          # 道路高度
    sidewalk_height = 0.15     # 人行道高度
    building_height = 3.0     # 建筑高度
    lane_mark_height = 0.02    # 车道线高度
    
    street_width = 6.0         # 街道宽度
    sidewalk_width = 4.0       # 人行道宽度
    building_size = 3.0       # 建筑尺寸
    lane_width = 10.0           # 车道宽度
    lane_mark_width = 0.15     # 车道线宽度
    
    # 2. 网格单位转换
    try:
        hs, vs = terrain.horizontal_scale, terrain.vertical_scale
        # 宽度尺寸
        sw, sww, bsz, lw, lmw = street_width, sidewalk_width, building_size, lane_width, lane_mark_width
        street_w = max(1, int(sw/hs))
        sidewalk_w = max(1, int(sww/hs))
        building_s = max(1, int(bsz/hs))
        lane_w = max(1, int(lw/hs))
        lane_mark_w = max(1, int(lmw/hs))
        
        # 高度值
        road_h = int(road_height/vs)
        sidewalk_h = int(sidewalk_height/vs)
        building_h = int(building_height/vs)
        cross_h = int((road_height + 0.1)/vs)
        lane_mark_h = int((road_height + lane_mark_height)/vs)
        
        # 地形参数
        w, l = terrain.width, terrain.length
        cx, cy = l//2, w//2
    except ZeroDivisionError:
        return
    
    # 3. 初始化地形
    terrain.height_field_raw[:, :] = road_h
    
    # 4. 创建十字街道
    sy, ey = cy - street_w//2, cy + street_w//2
    sx, ex = cx - street_w//2, cx + street_w//2
    
    terrain.height_field_raw[:, sy:ey] = road_h  # 水平街道
    terrain.height_field_raw[sx:ex, :] = road_h  # 垂直街道
    
    # 5. 添加车道系统
    if street_w > lane_w*2:
        # 水平车道
        terrain.height_field_raw[:, sy:sy+lane_w] = road_h          # 左车道
        terrain.height_field_raw[:, ey-lane_w:ey] = road_h          # 右车道
        # 车道分隔线
        lm_start = cy - lane_mark_w//2
        lm_end = cy + lane_mark_w//2
        terrain.height_field_raw[:, lm_start:lm_end] = lane_mark_h
        
        # 垂直车道
        terrain.height_field_raw[sx:sx+lane_w, :] = road_h          # 上车道
        terrain.height_field_raw[ex-lane_w:ex, :] = road_h          # 下车道
        # 车道分隔线
        lm_start = cx - lane_mark_w//2
        lm_end = cx + lane_mark_w//2
        terrain.height_field_raw[lm_start:lm_end, :] = lane_mark_h
    
    # 6. 人行道系统
    # 水平街道上下
    terrain.height_field_raw[:, ey:ey+sidewalk_w] = sidewalk_h
    terrain.height_field_raw[:, sy-sidewalk_w:sy] = sidewalk_h
    # 垂直街道左右
    terrain.height_field_raw[ex:ex+sidewalk_w, :] = sidewalk_h
    terrain.height_field_raw[sx-sidewalk_w:sx, :] = sidewalk_h
    
    # 7. 创建建筑物
    quad_size_x, quad_size_y = cx//2, cy//2
    builds = [
        (quad_size_x - building_s//2, quad_size_y - building_s//2),  # 左上
        #(cx + quad_size_x - building_s//2, quad_size_y - building_s//2),  # 右上
        #(quad_size_x - building_s//2, cy + quad_size_y - building_s//2),  # 左下
        (cx + quad_size_x - building_s//2, cy + quad_size_y - building_s//2)  # 右下
    ]
    
    for x, y in builds:
        x, y = max(0, min(x, l - building_s)), max(0, min(y, w - building_s))
        terrain.height_field_raw[x:x+building_s, y:y+building_s] = building_h
    
    # 8. 十字路口标记
    cross_size = max(1, min(street_w//3, 3))
    # 水平标记
    terrain.height_field_raw[cx-cross_size:cx+cross_size, cy-street_w//3:cy+street_w//3] = cross_h
    # 垂直标记
    terrain.height_field_raw[cx-street_w//3:cx+street_w//3, cy-cross_size:cy+cross_size] = cross_h
    
    
def create_stair_terrain(terrain):

    step_height=0.15
    step_width=1
    step_length=0.3
    num_steps=5
    direction='y'
    platform_size=0.8
    # 转换物理尺寸到网格单元
    step_length_cells = int(step_length / terrain.horizontal_scale)
    step_width_cells = int(step_width / terrain.horizontal_scale)
    step_height_units = int(step_height / terrain.vertical_scale)
    platform_cells = int(platform_size / terrain.horizontal_scale)
    
    # 计算中心位置
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    
    # 计算楼梯总尺寸
    total_stair_length = num_steps * step_length_cells
    
    if direction == 'x':
        # X方向楼梯 (从左到右上升)
        stair_start_x = center_x - total_stair_length // 2
        stair_start_y = center_y - step_width_cells // 2
        stair_end_y = center_y + step_width_cells // 2
        
        for i in range(num_steps):
            step_start_x = stair_start_x + i * step_length_cells
            step_end_x = step_start_x + step_length_cells
            current_height = (i + 1) * step_height_units
            
            # 创建台阶
            terrain.height_field_raw[step_start_x:step_end_x, stair_start_y:stair_end_y] = current_height
        
        # 添加顶部平台
        if platform_size > 0:
            platform_start_x = stair_start_x + num_steps * step_length_cells
            platform_end_x = platform_start_x + platform_cells
            platform_height = num_steps * step_height_units
            terrain.height_field_raw[platform_start_x:platform_end_x, stair_start_y:stair_end_y] = platform_height
            
            # 添加底部平台
            bottom_platform_start_x = stair_start_x - platform_cells
            bottom_platform_end_x = stair_start_x
            terrain.height_field_raw[bottom_platform_start_x:bottom_platform_end_x, stair_start_y:stair_end_y] = 0
    
    elif direction == 'y':
        # Y方向楼梯 (从前到后上升)
        stair_start_x = center_x - step_width_cells // 2
        stair_end_x = center_x + step_width_cells // 2
        stair_start_y = center_y - total_stair_length // 2
        
        for i in range(num_steps):
            step_start_y = stair_start_y + i * step_length_cells
            step_end_y = step_start_y + step_length_cells
            current_height = (i + 1) * step_height_units
            
            # 创建台阶
            terrain.height_field_raw[stair_start_x:stair_end_x, step_start_y:step_end_y] = current_height
        
        # 添加顶部平台
        if platform_size > 0:
            platform_start_y = stair_start_y + num_steps * step_length_cells
            platform_end_y = platform_start_y + platform_cells
            platform_height = num_steps * step_height_units
            terrain.height_field_raw[stair_start_x:stair_end_x, platform_start_y:platform_end_y] = platform_height
            
            # 添加底部平台
            bottom_platform_start_y = stair_start_y - platform_cells
            bottom_platform_end_y = stair_start_y
            terrain.height_field_raw[stair_start_x:stair_end_x, bottom_platform_start_y:bottom_platform_end_y] = 0