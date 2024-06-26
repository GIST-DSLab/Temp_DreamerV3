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
from numpy.typing import NDArray
from typing import Dict,Optional,Union,Callable,List, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from gymnasium import spaces

def rotate_left(state):
        temp_state = copy.deepcopy(state)
        rotate_state = []
        for  i in range(5):
            temp = []
            for j in range(5):
                temp.append(temp_state[j][4-i])
            rotate_state.append(temp)
        return rotate_state

# rotate_right function is a clockwise rotation about the given state.
def rotate_right(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-j][i])
        rotate_state.append(temp)
    return rotate_state

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[4-i][j])
        rotate_state.append(temp)
    return rotate_state

# vertical_flip function is a flip by y-axis about the given state.
def vertical_flip(state):
    temp_state = copy.deepcopy(state)
    rotate_state = []
    for  i in range(5):
        temp = []
        for j in range(5):
            temp.append(temp_state[i][4-j])
        rotate_state.append(temp)
    return rotate_state

def _get_bbox(img: NDArray) -> Tuple[int,int,int,int]:
    '''
    Receives NDArray, returns bounding box of its truthy values.
    '''

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def _init_objsel(state: dict, selection: NDArray) -> Tuple[int,int,int,int]:
    '''
    Initialize object selection states for smooth object-oriented actions
    '''

    objdict = state['grid']
    sel = selection
    # when something newly selected, previous object selection will be wiped
    if np.any(sel): 
        
        xmin, xmax, ymin, ymax = _get_bbox(sel) # bounding box of selection

        # return bounding box of selection
        return xmin, xmax, ymin, ymax

def _pad_assign(dst: NDArray, src: NDArray):
    h, w = src.shape
    dst[:h, :w] = src
    dst[h:,:] = 0
    dst[:, w:] = 0

def _apply_patch(state: dict):
    '''
    Combine 'background' and 'object' at 'object_pos', and put it into 'grid'.
    '''
    objdict = state['object_states']
    p: NDArray = objdict['object']

    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]

    # copy background
    np.copyto(state['grid'], objdict['background'])
    if   x+h>0 and x<gh and y+w>0 and y<gw  :
        # if patch is inside of the grid
        
        # patch paste bounding box
        stx = max(0,x)
        edx = min(gh,x+h)
        sty = max(0,y)
        edy = min(gw,y+w)
        
        # truncate patch
        p = p[ stx-x : edx-x, sty-y : edy-y ]
        np.copyto(state['grid'][stx:edx, sty:edy], p, where=(p>0))

def _apply_sel(state):
    '''
    Place the 'object_sel' into 'selected', at 'object_pos'
    '''
    objdict = state['object_states']
    p: NDArray = objdict['object_sel']

    x, y = objdict['object_pos']
    h, w = objdict['object_dim']
    gh, gw = state['grid_dim']
    p = p[:h, :w]

    # copy background
    state['selected'][:,:] = 0
    if   x+h>0 and x<gh and y+w>0 and y<gw  :
        # if patch is inside of the grid
        
        # patch paste bounding box
        stx = max(0,x)
        edx = min(gh,x+h)
        sty = max(0,y)
        edy = min(gw,y+w)
        
        # truncate patch
        p = p[ stx-x : edx-x, sty-y : edy-y ]
        np.copyto(state['selected'][stx:edx, sty:edy], p)

def gen_rotate(k=1):
    '''
    Generates Rotate90 / Rotate180 / Rotate270 actions counterclockwise.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`object_states`: Dict)
    '''
    assert 0<k<4

    def Rotate(state, action):

        xmin, xmax, ymin, ymax = _init_objsel(state,action['selection'])
        if xmin is None:
            return
        
        objdict = state['object_states']
        h, w = objdict['object_dim']

        if k%2 ==0:
            pass

        elif h%2 == w%2:
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            x,y = objdict['object_pos']
            objdict['object_pos'][:] = ( int(np.floor(cx-cy+y)), int(np.floor(cy-cx+x))) #left-top corner will be diagonally swapped
            objdict['object_dim'][:] = (w,h)
            

        else: # ill-posed rotation. Manually setted
            cx = (xmax + xmin) *0.5
            cy = (ymax + ymin) *0.5
            objdict['rotation_parity'][0] +=k
            objdict['rotation_parity'][0] %=2
            sig = (k+2)%4-2
            mod = 1-objdict['rotation_parity'][0]
            mx = min(  cx+sig*(cy-ymin) , cx+sig*(cy-ymax) )+mod
            my = min(  cy-sig*(cx-xmin) , cy-sig*(cx-xmax) )+mod
            objdict['object_pos'][:] = (int(np.floor(mx)),int(np.floor(my)))
            objdict['object_dim'][:] = (w,h)
            
        
        _pad_assign(objdict['object'], np.rot90(objdict['object'][:h,:w],k=k))
        _pad_assign(objdict['object_sel'], np.rot90(objdict['object_sel'][:h,:w],k=k))
        _apply_patch(state)
        _apply_sel(state)

    Rotate.__name__ = f"Rotate_{90*k}"    
    return Rotate

def gen_flip(axis:str = "H"):
    '''
    Generates Flip[H, V, D0, D1] actions. H=Horizontal, V=Vertical, D0=Major diagonal(transpose), D1=Minor diagonal 

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`object_states`: Dict)
    '''
    
    
    flips = {
        "H": lambda x: np.fliplr(x), 
        "V": lambda x: np.flipud(x), 
        "D0": lambda x: np.rot90(np.fliplr(x)), 
        "D1": lambda x: np.fliplr(np.rot90(x))
        }
    assert axis in flips,  "Invalid Axis"
    
    flipfunc = flips[axis]

    def Flip(state, action):
        sel = action['selection']
        valid,_,_,_ = _init_objsel(state,sel)
        if valid is None:
            return
        objdict = state['object_states']
        h, w = objdict['object_dim']

        _pad_assign(objdict['object'],flipfunc(objdict['object'][:h,:w]))
        _pad_assign(objdict['object_sel'],flipfunc(objdict['object_sel'][:h,:w]))
        _apply_patch(state)
        _apply_sel(state)
    
    Flip.__name__ = f"Flip_{axis}"
    return Flip


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

class BBoxDiagonalARCEnv(AbstractARCEnv):
    def __init__(self, 
                img_size, 
                data_loader,  
                max_grid_size, 
                colors,
                max_step = 2,
                max_trial = -1,
                render_mode = None, 
                render_size = None,
                few_shot=True
                ):
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)
        self._size = img_size
        self._resize = 'pillow'
        self.max_step = max_step
        # self.observation_space = self.create_observation_space() # self.create_state_space()
        self.eval_count = 0
        self.eval_list = None
        self.few_shot=few_shot

        if not os.path.exists('./logdir/BBox-DiagonalARC_Log/eval_diagonal.npy'):
            ex_in_list = np.array([np.array(np.random.randint(0, 10, size=(5, 5)).tolist()) for _ in range(1000)])
            ex_out_list = np.array([np.array(horizontal_flip(rotate_right(target))) for target in ex_in_list])
            full_list = np.stack((ex_in_list, ex_out_list))
            np.save('./logdir/BBox-DiagonalARC_Log/eval_diagonal.npy', full_list)
        
        self.eval_list = np.load('./logdir/BBox-DiagonalARC_Log/eval_diagonal.npy')

    def init_state(self, initial_grid: NDArray, options: Dict) -> None:
        super().init_state(initial_grid, options)
        
        add_dict = {
            "selected": np.zeros((self.H,self.W), dtype=np.int8),
            "clip" : np.zeros((self.H,self.W), dtype= np.int8),
            "clip_dim" : np.zeros((2,), dtype=np.int8),
            "object_states": {
                "active": np.zeros((1,),dtype=np.int8), 
                "object": np.zeros((self.H, self.W), dtype=np.int8),
                "object_sel": np.zeros((self.H, self.W), dtype=np.int8),
                "object_dim": np.zeros((2,), dtype=np.int8),
                "object_pos": np.zeros((2,), dtype=np.int8), 
                "background": np.zeros((self.H, self.W), dtype=np.int8), 
                "rotation_parity": np.zeros((1,),dtype=np.int8),
            }
        }

        self.current_state.update(add_dict)

    def reward(self, state):
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if tuple(state) == self.answer.shape:
            h,w = self.answer.shape
            if np.all(np.array(state)[0:h, 0:w] == self.answer):
                return 1
        return 0

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
        self.adaptation = False
        self.reset_on_submit = False
        self.options = options

        if options is not None:
            self.prob_index = options.get('prob_index')
            self.subprob_index = options.get('subprob_index')
            _ad = options.get('adaptation')
            self.adaptation = True if _ad is None else bool(_ad) # TODO self.adaptation이 뭔지 확인하기
            _ros = options.get('reset_on_submit')
            self.reset_on_submit = False if _ros is None else _ros
        
        ex_in = np.array(np.random.randint(0, 10, size=(5, 5)).tolist())
        ex_out = np.array(horizontal_flip(rotate_right(ex_in)))
        # ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=self.prob_index)

        
        if not self.adaptation:
            self.input_ = ex_in
            self.answer = ex_out

        # # TODO test 시점에서 아래가 어떤변수로 조건문이 통과되는지 확인하기
        else:
            self.input_ = self.eval_list[0][self.eval_count]
            self.answer = self.eval_list[1][self.eval_count]
            self.eval_count = 0 if (self.eval_count+1) % 1000 == 0 else self.eval_count+1

        self.init_state(self.input_.copy(),options)


        if self.render_mode:
            self.render()

        obs = self.current_state
        # obs['image'] = self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]['image']
        self.info = self.init_info()

        return self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]#, self.info
    
    def create_operations(self):
        ops = [gen_rotate(1), gen_rotate(3), gen_flip("H"), gen_flip("V")]

        return ops
    
    # def create_observation_space(self):
    #     spaces = {}
    #     # CNN
    #     # spaces["image"] = gym.spaces.Box(0, 255, (self._size[0], self._size[1], 3) , dtype=np.uint8)

    #     #MLP
    #     # spaces["image"] = gym.spaces.Box(0, 9, (30, 30) , dtype=np.uint8)

    #     # BBox
    #     spaces["grid"] = gym.spaces.Box(0, 9, (30, 30) , dtype=np.uint8)

    #     return gym.spaces.Dict(spaces)
    
    def create_state_space(self):
        old_space = super().create_state_space()

        '''
        active: is object selection mode enabled?
        object: original data of object shapes and colors
        object_sel: original shape of selection area, same-shaped to object_dim
        object_pos: position of object
        background: background separated to object, same-shaped with grid_dim
        rotation_parity: rotation parity to keep rotation center
        '''

        new_space_dict = {
                "selected": spaces.Box(0,1,(self.H,self.W),dtype=np.int8),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
                "clip_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),

                "object_states":spaces.Dict({
                    "active": spaces.MultiBinary(1),
                    "object": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
                    "object_sel":  spaces.Box(0,1,(self.H,self.W),dtype=np.int8),
                    "object_dim": spaces.Box(low=np.array([0,0]), high=np.array([self.H,self.W]), dtype=np.int8),
                    "object_pos": spaces.Box(low=np.array([-128,-128]), high=np.array([127,127]), dtype=np.int8), 

                    "background": spaces.Box(0, self.colors, (self.H,self.W),dtype=np.int8),
                    "rotation_parity": spaces.MultiBinary(1),
                })
        }

        new_space_dict.update(old_space.spaces)
        return spaces.Dict(new_space_dict)

    def render_ansi(self):
        return 
        if self.rendering is None:
            self.rendering = True
            print('\033[2J',end='')

        print(f'\033[{self.H+3}A\033[K', end='')

        state = self.current_state
        grid = state
        grid_dim = state

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
        # grid = state
        # image = Image.open(create_img(grid)).convert('RGB')
        # image = image.resize(self._size)
        # image = np.array(image)
        grid = torch.tensor(state['grid'])
        bottom_pad_size = 30 - grid.shape[0]
        right_pad_size = 30 - grid.shape[1] 
        image = torch.nn.functional.pad(grid, (0, right_pad_size, 0, bottom_pad_size), 'constant', 0)#.unsqueeze(-1)

        return (
            {"grid": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        )

    def step(self, action):
        # 문제였던 부분.
        self.last_action_op = action

        # do action
        state = self.current_state
        self.operations[action['operation']](state, action)
        state = self.current_state
        reward = self.reward(state)
        self.last_reward += reward
        self.action_steps+=1
        self.info["steps"] = self.action_steps
        #self.render() print(self.operations[2](self.operations[1](self.input_, 1),2)) print(self.operations[action](state,1))

        
        is_terminal = bool(reward)
        state["terminated"][0] = is_terminal
        info = self.info

        if self.info["steps"] == 2:
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

