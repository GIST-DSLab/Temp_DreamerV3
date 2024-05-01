from arcle.envs.arcenv import AbstractARCEnv
from arcle.loaders import Loader
from gymnasium.core import ObsType, ActType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from io import BytesIO
import gym
import copy
import pdb
from PIL import Image
import random
import torch

def rotate_left(state, action):
        temp_state = copy.deepcopy(state)
        rotate_state = []
        for  i in range(5):
            temp = []
            for j in range(5):
                temp.append(temp_state[j][4-i])
            rotate_state.append(temp)
        return rotate_state

# rotate_right function is a clockwise rotation about the given state.
def rotate_right(state, action):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-j][i])
        rotate_state.append(temp)
    return rotate_state

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state, action):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-i][j])
        rotate_state.append(temp)
    return rotate_state

# vertical_flip function is a flip by y-axis about the given state.
def vertical_flip(state, action):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[i][4-j])
        rotate_state.append(temp)
    return rotate_state

def create_img(state):
    cvals  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]     # [Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Light blue, Brown]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    output_folder = './'
    fig, axs = plt.subplots(1, 1, figsize=(3 * 0.7, 3 * 0.7))
    rows, cols = np.array(state).shape
    axs.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    axs.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    axs.tick_params(which='minor', size=0)
    axs.grid(True, which='minor', color='#555555', linewidth=1)
    axs.set_xticks([]); axs.set_yticks([])
    axs.imshow(np.array(state), cmap=cmap, vmin=0, vmax=9)

    # Find the corresponding task_id for the task_name
    # output_folder = os.path.join('', str('0c786b71'))
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # case_file_name = f"{file_name}.png"
    # output_file_path = os.path.join(output_folder, case_file_name)

    plt.tight_layout()
    tmpfile = BytesIO()
    plt.savefig(tmpfile, bbox_inches='tight', format='png', dpi=300)
    plt.close()

    return tmpfile

class SimpleARCEnv(AbstractARCEnv):
    def __init__(self, 
                img_size, 
                data_loader,  
                max_grid_size, 
                colors,
                max_step = 2,
                max_trial = -1,
                render_mode = None, 
                render_size = None):
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
        self._size = img_size
        self._resize = 'pillow'
        self.max_step = max_step
        self.observation_space = None

    def reward(self, state):
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if tuple(state['grid_dim']) == self.answer.shape:
            h,w = self.answer.shape
            if np.all(np.array(state['grid'])[0:h, 0:w] == self.answer):
                return 1
        return 0

    @property
    def observation_space(self):
        spaces = {}
        # CNN
        # spaces["image"] = gym.spaces.Box(0, 255, (self._size[0], self._size[1], 3) , dtype=np.uint8)

        #MLP
        spaces["image"] = gym.spaces.Box(0, 9, (30, 30) , dtype=np.uint8)
        return gym.spaces.Dict(spaces)
    
    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    # TODO 바꾸기
    def reset(self, seed = 777, options = None):
        # super().reset(seed=seed,options=options)

        # Reset Internal States
        self.truncated = False
        self.submit_count = 0
        self.last_action  = None
        self.last_action_op   = None
        self.last_reward = 0
        self.action_steps = 0
        
        # env option
        self.prob_index = None
        self.subprob_index = None
        self.adaptation = True
        self.reset_on_submit = False
        self.options = options

        if options is not None:
            self.prob_index = options.get('prob_index')
            self.subprob_index = options.get('subprob_index')
            _ad = options.get('adaptation')
            self.adaptation = True if _ad is None else bool(_ad)
            _ros = options.get('reset_on_submit')
            self.reset_on_submit = False if _ros is None else _ros
        
        ex_in = np.array(np.random.randint(0, 10, size=(5, 5)).tolist())
        # ex_out = np.array(horizontal_flip(rotate_right(ex_in, None), None))
        ex_out = np.array(rotate_right(ex_in, None), None)
        # ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=self.prob_index)

        
        if self.adaptation:
            self.input_ = ex_in
            self.answer = ex_out

        # else:
        #     self.subprob_index = np.random.randint(0,len(tt_in)) if self.subprob_index is None else self.subprob_index
        #     self.input_ = tt_in[self.subprob_index]
        #     self.answer = tt_out[self.subprob_index]

        self.init_state(self.input_.copy(),options)


        if self.render_mode:
            self.render()

        obs = self.current_state
        # obs['image'] = self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]['image']
        self.info = self.init_info()

        return self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]#, self.info
    
    def create_operations(self):
        ops = [rotate_right]
        # ops = [rotate_left, rotate_right, horizontal_flip, vertical_flip]

        return ops

    def render_ansi(self):
        return 
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')

        print(f'\033[{self.H+3}A\033[K', end='')

        state = self.current_state
        grid = state['grid']
        grid_dim = state['grid_dim']

        for i,dd in enumerate(grid):
            for j,d in enumerate(dd):
                
                if i >= grid_dim[0] or j>= grid_dim[1]:
                    print('\033[47m  ', end='')
                else:
                    print("\033[48;5;"+str(self.ansi256arc[d])+"m  ", end='')

            print('\033[0m')

        print('Dimension : '+ str(grid_dim), end=' ')
        print('Action : ' + str(self.op_names[self.last_action_op] if self.last_action_op is not None else '') , end=' ')

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        state = self.current_state
        # grid = state['grid']
        # image = Image.open(create_img(grid)).convert('RGB')
        # image = image.resize(self._size)
        # image = np.array(image)
        grid = torch.tensor(state['grid'])
        bottom_pad_size = 30 - grid.shape[0]
        right_pad_size = 30 - grid.shape[1] 
        image = torch.nn.functional.pad(grid, (0, right_pad_size, 0, bottom_pad_size), 'constant', 0)#.unsqueeze(-1)

        return (
            {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )

    def step(self, action):
        # 문제였던 부분.
        self.last_action_op = action

        if len(action.shape) >= 1:
            action = np.argmax(action)

        # self.last_action = action

        # do action
        state = self.current_state
        if action == 4:
            self.current_state['grid']= self.operations[action](state, action)
        else:
            self.current_state['grid'] = self.operations[action](state['grid'], action)
        state = self.current_state
        reward = self.reward(state)
        self.last_reward += reward
        self.action_steps+=1
        self.info["steps"] = self.action_steps
        #self.render()

        
        is_terminal = bool(reward)
        state["terminated"][0] = is_terminal
        info = self.info

        if self.info["steps"] == 1:
            self.submit(state)
            is_terminal = True
        
        return self._obs(
            self.last_reward,
            is_last=info['steps'] == self.max_step,
            is_terminal=is_terminal,
        )

    # @property
    def action_space(self):
        space = self.create_action_space(len(self.create_operations()))
        space.discrete = True
        return space
    
    def create_action_space(self, action_count) -> gym.spaces.Dict: 
        return gym.spaces.Discrete(action_count)
    
    def submit(self, state) -> None:
        if state["trials_remain"][0] !=0:
            state["trials_remain"][0] -=1
            self.submit_count +=1
            h,w = state["grid_dim"][0], state["grid_dim"][1]
            if self.answer.shape == (h,w) and np.all(self.answer==np.array(state["grid"])[:h,:w]):
                state["terminated"][0] = 1 # correct
            if self.reset_on_submit:
                self.init_state(self.input_, options=self.options)

        if state["trials_remain"][0] == 0:
            state["terminated"][0] = 1 # end 
        
        self.current_state = state

