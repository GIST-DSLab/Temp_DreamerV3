from arcle.envs.arcenv import AbstractARCEnv
from arcle.loaders import ARCLoader
from gymnasium import spaces
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
from functools import wraps
from numpy import ma
import json
import pickle


class EntireSelectionLoader(ARCLoader):
    def __init__(self, data_index):
        self.data_index = data_index
        super().__init__()

    def parse(self, **kwargs):
        dat = []

        for i, p in enumerate(self._pathlist):
            with open(p) as fp:

                problem = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                if i+1 in [self.data_index]:
                    for d in problem['train']:
                            inp = np.array(d['input'],dtype=np.int8)
                            oup = np.array(d['output'],dtype=np.int8)
                            ti.append(inp)
                            to.append(oup)

                    for d in problem['test']:
                            inp = np.array(d['input'],dtype=np.int8)
                            oup = np.array(d['output'],dtype=np.int8)
                            ei.append(inp)
                            eo.append(oup)
                    if len(ti) == 0:
                        continue

                    desc = {'id': os.path.basename(fp.name).split('.')[0]}
                    dat.append((ti,to,ei,eo,desc))
                else:
                    continue
        return dat

def chang_color_permute(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    permute_color = np.random.choice([i for i in range(0,10)], 10, replace=False).tolist()
    for i in range(3):
        for j in range(3):
            temp_state[i][j] = permute_color[temp_state[i][j]]
    return temp_state


def rotate_left(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    h, w = np.array(temp_state).shape
    for  i in range(h):
        temp = []
        for j in range(w):
            temp.append(temp_state[j][w-1-i])
        rotate_state.append(temp)
    return rotate_state

# rotate_right function is a clockwise rotation about the given state.
def rotate_right(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    h, w = np.array(temp_state).shape
    rotate_state = []
    for  i in range(h):
        temp = []
        for j in range(w):
            temp.append(temp_state[w-1-j][i])
        rotate_state.append(temp)
    return rotate_state

# vertical_flip function is a flip by y-axis about the given state.
def vertical_flip(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    h, w = np.array(temp_state).shape
    rotate_state = []
    for  i in range(h):
        temp = []
        for j in range(w):
            temp.append(temp_state[h-1-i][j])
        rotate_state.append(temp)
    return rotate_state

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state): 
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    h, w = np.array(temp_state).shape
    for  i in range(h):
        temp = []
        for j in range(w):
            temp.append(temp_state[i][w-1-j])
        rotate_state.append(temp)
    return rotate_state

def move_down(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    new_temp_state = np.zeros((temp_state.shape[0]+1,temp_state.shape[1]))
    new_temp_state[1:,:] = temp_state 
    return new_temp_state[:-1,:]

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

    objdict = state['object_states']
    sel = selection
    # when something newly selected, previous object selection will be wiped
    if np.any(sel): 
        
        xmin, xmax, ymin, ymax = _get_bbox(sel) # bounding box of selection
        
        h = xmax-xmin+1
        w = ymax-ymin+1

        
        # wipe object and set a new object from selected pixels
        objdict['object_dim'][:] = (h, w)
        selected_part = sel[xmin:xmax+1, ymin:ymax+1] >0

        objdict['object'][:, :] = 0 
        np.copyto(objdict['object'][0:h,0:w], state['grid'][xmin:xmax+1, ymin:ymax+1], where=selected_part)
        
        objdict['object_sel'][:,:] = 0
        np.copyto(objdict['object_sel'][0:h,0:w], selected_part, where=selected_part)
        
        # background backup
        np.copyto(objdict['background'], state['grid'])
        np.copyto(objdict['background'], 0, where=(sel>0))

        # position, active, parity initialize
        objdict['object_pos'][:] = (int(xmin), int(ymin)) 
        objdict['active'][0] = 1
        objdict['rotation_parity'][0] = 0

        # copy selection into selected obs
        np.copyto(state['selected'], np.copy(sel).astype(np.int8))

        # return bounding box of selection
        return xmin, xmax, ymin, ymax
    

    # when object selection was active without new selection, continue with prev objsel
    elif objdict['active'][0]: 
        # gives previous bounding pox
        x, y = objdict['object_pos']
        h, w = objdict['object_dim']
        return x, x+h-1, y, y+w-1
    
    # when objsel inactive and no selection, we ignore this action
    else:
        return None, None, None, None

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

    def Rotate(state):
        sel = np.pad(np.ones((state['grid_dim'])), ((0,state['grid'].shape[0]-state['grid_dim'][0]), (0, state['grid'].shape[1]-state['grid_dim'][1])))
        xmin, xmax, ymin, ymax = _init_objsel(state, sel)
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

    def Flip(state):
        sel = np.pad(np.ones((state['grid_dim'])), ((0,state['grid'].shape[0]-state['grid_dim'][0]), (0, state['grid'].shape[1]-state['grid_dim'][1])), 'constant', constant_values=0).astype(dtype=np.int8)
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

def reset_sel(function):
    '''
    Wrapper for Non-O2ARC actions. This wrapper resets `selected` of obs space and resets object-operation states.

    It does this before calling function.
    ```
        state['selected'] = np.zeros((H,W), dtype=np.int8)
        state['object_states']['active'][0] = 0
    ```
    '''
    @wraps(function)
    def wrapper(state, **kwargs):
        sel = np.pad(np.ones((state['input_dim'])), ((0,state['input'].shape[0]-state['input_dim'][0]), (0, state['input'].shape[1]-state['input_dim'][1])), 'constant', constant_values=0).astype(dtype=np.int8)
        state['selected'] = sel
        state['object_states']['active'][0] = 0
        
        return function(state, **kwargs)
    return wrapper

def gen_copy(source="I"):
    '''
    Generates Copy[I,O] actions. Source is input grid when "I", otherwise "O". It is for O2ARCv2Env. If you want to use generic Copy/Paste, please wait further updates.

    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: NDArray)
    '''
    assert source in ["I", "O"], "Invalid Source grid"
    srckey = 'input' if source=="I" else 'grid'
    def Copy(state):
        sel = np.pad(np.ones((state['input_dim'])), ((0,state['input'].shape[0]-state['input_dim'][0]), (0, state['input'].shape[1]-state['input_dim'][1])), 'constant', constant_values=0).astype(dtype=np.int8)
        
        if not np.any(sel>0): #nothing to copy
            return
        
        xmin, ymin = 0, 0
        xmax, ymax = np.array(state[srckey]).shape

        ss_h, ss_w = state[srckey+'_dim']

        if xmax>ss_h or ymax>ss_w: # out of bound
            return
        
        h = xmax-xmin+1
        w = ymax-ymin+1

        state['clip'][:,:] = 0
        state['clip_dim'][:] = (h,w)

        src_grid = np.array(state[srckey])[xmin:xmax+1, ymin:ymax+1]
        np.copyto(state['clip'][:h, :w], src_grid, \
                      where=np.logical_and(src_grid, sel[xmin:xmax+1, ymin:ymax+1] ))
    Copy.__name__ = f"Copy_{source}"
    return Copy

def gen_paste(paste_blank = False):
    def Paste(state):
        '''
        Paste action.

        Action Space Requirements (key: type) : (`selection`: NDArray)

        Class State Requirements (key: type) :  (`grid`: NDArray), (`grid_dim`: NDArray), (`clip`: NDArray), (`clip_dim`: NDArray)
        '''
        # 아래의 selection을 항상 전체 그리드를 대상으로 하게끔 바꾸기
        xmin, ymin = 0, 0

        H, W = state['input'].shape
        h, w = state['clip_dim']

        if xmin >= H or ymin >= W or h==0 or w==0: # out of bound or no selection
            return
        
        patch = state['clip'][:h, :w]
        
        # truncate patch
        edx = min(xmin+h, H)
        edy = min(ymin+w, W)
        patch = patch[:edx-xmin, :edy-ymin]
        
        # paste
        if paste_blank:
            np.copyto(state['grid'][xmin:edx, ymin:edy], patch) # for debug'
        else:
            np.copyto(state['grid'][xmin:edx, ymin:edy], patch, where=(patch>0))
    return Paste


def gen_color(color: SupportsInt) -> Callable:
    '''
    Generates Color0 ~ Color9 functions that color multiple pixels within selection.
    
    Action Space Requirements (key: type) : (`selection`: NDArray)

    Class State Requirements (key: type) : (`grid`: NDArray)
    '''
    def colorf(state) -> None:
        sel = np.pad(np.ones((state['input_dim'])), ((0,state['input'].shape[0]-state['input_dim'][0]), (0, state['input'].shape[1]-state['input_dim'][1])), 'constant', constant_values=0).astype(dtype=np.int8)
        if not np.any(sel):
            return
        state['grid'] = ma.array(state['grid'], mask=sel).filled(fill_value=color)
    
    colorf.__name__ = f"Color{color}"
    return colorf

def gen_move(d=0):
    '''
    Generates Move[U,D,R,L] actions. d=0 means move up, d=1 is down, d=2 is right, d=3 is left.

    Action Space Requirements (key: type) : (`selection`: NDArray)  

    Class State Requirements (key: type) : (`grid`: NDArray), (`grid_dim`: NDArray), (`selected`: NDArray), (`object_states`: Dict)
    '''
    assert 0<= d <4
    dirX = [-1, +1, 0, 0]
    dirY = [0, 0, +1, -1]

    def Move(state):
        sel = np.pad(np.ones((state['grid_dim'])), ((0,state['grid'].shape[0]-state['grid_dim'][0]), (0, state['grid'].shape[1]-state['grid_dim'][1])), 'constant', constant_values=0).astype(dtype=np.int8)
        par,_,_,_ = _init_objsel(state,sel)

        if par is None:
            return

        x, y = state['object_states']['object_pos']
        state['object_states']['object_pos'][:] = (int(x + dirX[d]), int(y + dirY[d]))
        _apply_patch(state)
        _apply_sel(state)
        
    Move.__name__ = f"Move_{'UDRL'[d]}"
    return Move


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

class DiagonalARCEnv(AbstractARCEnv):
    def __init__(self, 
                img_size, 
                data_loader,  
                max_grid_size, 
                colors,
                max_step = 2,
                max_trial = -1,
                render_mode = None, 
                render_size = None,
                config = None,
                ): 
        self._config = config
        self.num_func = config.num_func
        self.submit_flag = config.submit_flag
        super().__init__(data_loader, max_grid_size, colors, config.max_trial, render_mode, render_size)
        self.max_grid_size = max_grid_size
        self._size = img_size
        self._resize = 'pillow'
        self.max_step = max_step
        self.observation_space = None
        self.train_count = 0
        self.eval_count = 0
        self.eval_list = None
        self.few_shot = config.few_shot
        self.log_dir = config.logdir.split('/')[-1]
        self.color_permute = config.color_permute
        self.acc_flag = config.acc_flag
        self.train_set = None
        self.oracle_reward = config.oracle_reward
        self.aug_train_num = config.aug_train_num
        self.aug_eval_num = config.aug_eval_num
        self.aug_func = {179: self.aug_179, 241: self.aug_241, 150: self.aug_150, 380: self.aug_380, 53: self.aug_53, 385: self.aug_385}
        self.task_index = config.task_index
        self.dataset_dir = config.dataset_dir
        self.use_example = config.use_example
        self.RQ4_eval_mode = config.RQ4_eval_mode
        self.train_index_set = set()

        # self.epiosde_index = 0 # 0: 'rotate_left', 1: 'rotate_right', 2: 'horizental_flip', 3: 'vertical_flip'
        # self.count_action_case = {i+' '+j: 0 for i in ['rotate_left','rotate_right', 'horizental_flip','vertical_flip'] for j in ['rotate_left','rotate_right', 'horizental_flip','vertical_flip']}
        # self.current_action_case = ''

        if not self.few_shot:
            if not os.path.exists(f'{self.dataset_dir}/train_{self.task_index}.pkl') and type(self.task_index) == int:
                self.train_list = self.aug_func[config.task_index](mode='train')
                with open(f'{self.dataset_dir}/train_{self.task_index}.pkl', 'wb') as f:
                    pickle.dump(self.train_list, f, pickle.HIGHEST_PROTOCOL)
            else:
                if type(self.task_index) == int:
                    with open(f'{self.dataset_dir}/train_{self.task_index}.pkl', 'rb') as f:
                        self.train_list = pickle.load(f)
                else:
                    with open(f'{self.dataset_dir}/new_train_A-B_1000.pkl', 'rb') as f:
                        self.train_list = pickle.load(f)
        if self.acc_flag:
            if not os.path.exists(f'{self.dataset_dir}/eval_{self.task_index}.pkl') and type(self.task_index) == int:                 
                self.eval_list = self.aug_func[config.task_index](mode='eval', train_set=None if self.few_shot else set(map(str,self.train_list[0])))
                with open(f'{self.dataset_dir}/eval_{self.task_index}.pkl', 'wb') as f:
                    pickle.dump(self.eval_list, f, pickle.HIGHEST_PROTOCOL)
            else:
                if type(self.task_index) == int: 
                    with open(f'{self.dataset_dir}/eval_{self.task_index}.pkl', 'rb') as f:
                        self.eval_list = pickle.load(f)
                else:
                    target_file = None
                    if self.task_index == 'RQ4':
                        if self.RQ4_eval_mode == 'A':
                            target_file = f'{self.dataset_dir}/new_eval_B.pkl' 
                        elif self.RQ4_eval_mode == 'B':
                            target_file = f'{self.dataset_dir}/new_eval_B.pkl' 
                    else:
                        target_file = f'{self.dataset_dir}/new_eval_AB.pkl' 
                    with open(target_file, 'rb') as f:
                        self.eval_list = pickle.load(f)


    def aug_179(self, mode, train_set=None):
        if mode == 'train':
            input_list = [np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for i in range(self.aug_train_num)]
            output_list = [np.array(vertical_flip(rotate_right(target))) for target in input_list]
        
        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(vertical_flip(rotate_right(target))) for target in input_list]
        
        return [input_list, output_list]

    def aug_241(self, mode, train_set=None):
        n_dim = [(3,3), (4,4), (5,5), (6,6)]
        if mode == 'train':
            n_list = [np.random.randint(0, 3, size=1).tolist() for _ in range(self.aug_train_num)]
            input_list = [np.array(np.random.randint(0, 10, size=(n_dim[i[0]])).tolist()) for i in n_list]
            output_list = [np.array(vertical_flip(rotate_right(target))) for target in input_list]

        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=n_dim[-1]).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(vertical_flip(rotate_right(target))) for target in input_list]
        
        return [input_list, output_list]

    def aug_380(self, mode, train_set=None):
        if mode == 'train':
            input_list = [np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for i in range(self.aug_train_num)]
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        return [input_list, output_list]

    def aug_150(self, mode, train_set=None):
        n_dim = [(4,4), (7,7), (6,6), (3,3)]
        if mode == 'train':
            n_list = [np.random.randint(0, 3, size=1).tolist() for _ in range(self.aug_train_num)]
            input_list = [np.array(np.random.randint(0, 10, size=(n_dim[i[0]])).tolist()) for i in n_list]
            output_list = [np.array(horizontal_flip(target)) for target in input_list]

        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=n_dim[-1]).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(horizontal_flip(target)) for target in input_list]
        
        return [input_list, output_list]

    # TODO move down 되는지 확인하기 -> 그냥 SOLAR 형식에 맞춰서 증강하기
    def aug_53(self, mode, train_set=None):
        if mode == 'train':
            input_list = [np.concatenate((np.random.randint(0, 10, size=(2, 3)),np.zeros((1,3)))) for i in range(self.aug_train_num)]
            output_list = [np.array(move_down(target)) for target in input_list]
        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        return [input_list, output_list]

    # TODO: 10x4 grid, 6~10번째 행에 대해서 랜덤하게 생성한 후 copyO -> paste-> FlipV를 적용하면 될 듯
    def aug_385(self, mode, train_set=None):
        if mode == 'train':
            target_input_list = [np.array(np.random.randint(0, 10, size=(5, 4)).tolist()) for i in range(self.aug_train_num)] 
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        return [input_list, output_list]
    
    '''
    TODO: 3x3 grid, 
    1. 먼저 몇 개의 점을 찍을지 랜던하게 설정하기
    2. 각 열의 랜덤한 위치에 랜덤한 색의 점을 찍기(점은 1에서 정한 만큼만 찍을 수 있으면 한 열당 1개의 점만 찍힐 수 있음, 이때 검정색 점은 안 찍히도록 하기)
    3. 각 열에서 찍힌 점 중 가장 높은 위치(1행에 가까울 수록 높은 위치, 상단이 1행)에 찍힌 점을 기준으로 CopyO → Paste → MoveD을 [4-높은 위치 행의 값]번 반복해주기
    '''
    def aug_322(self, mode, train_set=None): 
        if mode == 'train':
            input_list = [np.array(np.random.randint(0, 10, size=(3, 3)).tolist()) for i in range(self.aug_train_num)]
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        if mode == 'eval':
            input_list = []
            for _ in range(self.aug_eval_num):
                while True:
                    temp = np.random.randint(0, 10, size=(3, 3)).tolist()
                    if train_set == None or str(temp) not in train_set:
                        input_list.append(np.array(temp))
                        break
            output_list = [np.array(rotate_left(target)) for target in input_list]
        
        return [input_list, output_list]

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
        if tuple(state['grid_dim']) == self.answer.shape:
            h,w = self.answer.shape
            if np.all(np.array(state['grid'])[0:h, 0:w] == self.answer):
                if not self.last_action_op == len(self.operations)-1:
                    return 1
                else:
                    return 1000
        return 0

    @property
    def observation_space(self):
        # spaces = {}
        # # CNN
        # # spaces["image"] = gym.spaces.Box(0, 255, (self._size[0], self._size[1], 3) , dtype=np.uint8)

        # #MLP
        # spaces["grid"] = gym.spaces.Box(0, 9, (30, 30) , dtype=np.uint8)
        # return gym.spaces.Dict(spaces)
         return super().create_state_space() if not self.use_example else spaces.Dict({
            "trials_remain": spaces.Box(-1, self.max_trial, shape=(1,), dtype=np.int8),
            "terminated": spaces.MultiBinary(1), # int8

            "input": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "input_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "grid_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "example_input1": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_input1_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "example_input2": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_input2_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),
            
            "example_input3": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_input3_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "example_output1": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_output1_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "example_output2": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_output2_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),

            "example_output3": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.int8),
            "example_output3_dim": spaces.Box(low=np.array([1,1]), high=np.array([self.H,self.W]), dtype=np.int8),
        })
    
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
        self.adaptation = False
        self.reset_on_submit = False
        self.options = options
        # self.current_action_case = ''

        if options is not None:
            self.prob_index = options.get('prob_index')
            self.subprob_index = options.get('subprob_index')
            _ad = options.get('adaptation')
            self.adaptation = True if _ad is None else bool(_ad) # TODO self.adaptation이 뭔지 확인하기
            _ros = options.get('reset_on_submit')
            self.reset_on_submit = False if _ros is None else _ros
        
        # ex_in = np.array(np.random.randint(0, 10, size=(5, 5)).tolist())
        # ex_out = np.array(horizontal_flip(rotate_right(ex_in)))

        ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=self.prob_index)

        # TODO [고민하기] 현재 self.adaptation을 통해서 train시점과 eval시점을 env환경이 구별해주는데 validation은 어떻게 구별해주면 좋을지 생각하기.
        if self.adaptation:
            if self.few_shot:
                self.subprob_index = np.random.randint(0,len(ex_in)) if self.subprob_index is None else self.subprob_index
                self.input_ = ex_in[self.subprob_index]
                self.answer = ex_out[self.subprob_index]
            else:
                if self.color_permute:
                    self.input_ = chang_color_permute(ex_in[self.subprob_index])
                    self.answer = chang_color_permute(ex_out[self.subprob_index])
                else:
                    if self.use_example:
                        self.example_input = self.train_list['example'][0][self.train_count]
                        self.example_output = self.train_list['example'][1][self.train_count]
                        self.input_ = self.train_list['grid'][self.train_count] # ex_in
                        self.answer = self.train_list['answer'][self.train_count] 
                    else:
                        self.input_ = self.train_list[0][self.train_count] # ex_in
                        self.answer = self.train_list[1][self.train_count] # ex_out

                    self.train_count = 0 if (self.train_count+1) % self.aug_train_num == 0 else self.train_count+1
                    self.train_index_set.add(self.train_count)
        else:
            if self.acc_flag:
                if self.use_example:
                    self.example_input = self.eval_list['example'][0][self.eval_count*3:(self.eval_count+1)*3]
                    self.example_output = self.eval_list['example'][1][self.eval_count*3:(self.eval_count+1)*3]
                    self.input_ = self.eval_list['grid'][self.eval_count] # ex_in
                    self.answer = self.eval_list['answer'][self.eval_count] # ex_out
                else:
                    self.input_ = self.eval_list[0][self.eval_count]
                    self.answer = self.eval_list[1][self.eval_count]
                self.eval_count = 0 if (self.eval_count+1) % self.aug_eval_num == 0 else self.eval_count+1
            else:
                self.subprob_index = np.random.randint(0,len(tt_in)) if self.subprob_index is None else self.subprob_index
                self.input_ = tt_in[self.subprob_index]
                self.answer = tt_out[self.subprob_index]



        # 아래는 코드가 잘 돌아가는 체크를 위해서 image를 저장하도록 함.
        # input_image = Image.open(create_img(self.input_)).convert('RGB')
        # input_image.save(f'./logdir/DiagonalARC_Log/images/{self.epiosde_index}_input.png', 'png')

        # output_image = Image.open(create_img(self.answer)).convert('RGB')
        # output_image.save(f'./logdir/DiagonalARC_Log/images/{self.epiosde_index}_output.png', 'png')

        self.init_state(self.input_.copy(),options)


        if self.render_mode:
            self.render()

        obs = self.current_state
        # obs['image'] = self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]['image']
        self.info = self.init_info()

        return self._obs(reward=0, is_first=True, is_last=False, is_terminal=False)[0]#, self.info
    
    def create_operations(self):
        ops = []

        # ops += [rotate_left, rotate_right, horizontal_flip, vertical_flip]

        ops += [gen_rotate(1), gen_rotate(3), gen_flip("H"), gen_flip("V")] #왼쪽, 오른쪽, 수평, 수직
        if self.num_func == 11:
            ops += [reset_sel(gen_color(i)) for i in [4,6,8,9]] # 노란색, 분홍색, 하늘색, 갈색
            ops += [gen_move(d=1)] # 아래 이동
            ops += [reset_sel(gen_copy("O")) , reset_sel(gen_paste(paste_blank=True))] # grid 복사, 붙여넣기
        if self.submit_flag:
            ops += [self.submit] # 제출
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

        if self._config.use_image:
            grid = state['grid']
            image = Image.open(create_img(grid)).convert('RGB')
            image = image.resize(self._size)
            image = np.array(image)
        else:
            grid = torch.tensor(state['grid'])
            bottom_pad_size = self.max_grid_size[0] - grid.shape[0]
            right_pad_size = self.max_grid_size[1] - grid.shape[1] 
            image = torch.nn.functional.pad(grid, (0, right_pad_size, 0, bottom_pad_size), 'constant', 0)#.unsqueeze(-1)

        if self.use_example:
            image_example_input = []
            image_example_output = []
            
            # TODO 추후에 example pair의 grid 사이즈가 동일하지 않을 수 있으므로 아래의 코드를 수정해서 확장하기 - evaluation일때만 고치면 됨.
            for example_input ,example_output in zip(self.example_input, self.example_output):
                input_bottom_pad_size = self.max_grid_size[0] - torch.tensor(example_input).shape[0]
                input_right_pad_size = self.max_grid_size[1] - torch.tensor(example_input).shape[1] 

                output_bottom_pad_size = self.max_grid_size[0] - torch.tensor(example_output).shape[0]
                output_right_pad_size = self.max_grid_size[1] - torch.tensor(example_output).shape[1] 

                image_example_input.append(torch.nn.functional.pad(torch.tensor(example_input), (0, input_right_pad_size, 0, input_bottom_pad_size), 'constant', 0))
                image_example_output.append(torch.nn.functional.pad(torch.tensor(example_output), (0, output_right_pad_size, 0, output_bottom_pad_size), 'constant', 0))

        return (
            {"grid": image, "is_terminal": is_terminal, "is_first": is_first},
            reward,
            is_last,
            {},
        ) if not self.use_example else(
            {"grid": image, "is_terminal": is_terminal, "is_first": is_first, 
             'example_input1': image_example_input[0], 'example_output1': image_example_output[0], 'example_input2': image_example_input[1], 'example_output2': image_example_output[1], 'example_input3': image_example_input[2], 'example_output3': image_example_output[2]},
            reward,
            is_last,
            {},
        )

    def step(self, action):
        # 문제였던 부분.
        if len(action.shape) >= 1:
            action = np.argmax(action)

        self.last_action_op = action

        # do action
        state = copy.deepcopy(self.current_state)
        self.operations[action](state)
        self.current_state['grid'] = state['grid']

        # state['grid'] = self.operations[action](state)
        # self.current_state['grid'] = state['grid']

        state = self.current_state
        self.action_steps+=1
        reward = self.reward(state)

        self.last_reward = reward
        # self.last_reward = reward
        
        self.info["steps"] = self.action_steps
        #self.render() print(self.operations[2](self.operations[1](self.input_, 1),2)) print(self.operations[action](state['grid'],1))
        # action_map = {0: 'rotate_left', 1: 'rotate_right', 2: 'horizental_flip', 3: 'vertical_flip'}

        # self.current_action_case += ' ' + action_map[action] if self.current_action_case != '' else action_map[action]

        # grid = state['grid']
        # image = Image.open(create_img(grid)).convert('RGB')
        # image.save(f'./logdir/DiagonalARC_Log/images/{self.epiosde_index}_{self.info["steps"]}_{action_map[action]}_{reward}.png', 'png')

        # is_terminal = bool(reward)
        # state["terminated"][0] = is_terminal
        info = self.info

        if self.info["steps"] == self.max_step:
            # self.submit(state)
            is_terminal = True
        
        return self._obs(
            self.last_reward,
            is_last=(info['steps'] == self.max_step) or state["terminated"][0],
            is_terminal=state["terminated"][0],
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
        
        # self.epiosde_index += 1
        # self.count_action_case[self.current_action_case] += 1

        self.current_state = state

        # if self.epiosde_index == 999:
        #     with open('logdir/DiagonalARC_Log/count.json','w') as f:
        #         json.dump(self.count_action_case, f, indent=4)

