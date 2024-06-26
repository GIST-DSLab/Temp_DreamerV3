# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

### Method 1: Manual

Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
Run training on Diagonal Flip:
```
python3 dreamer.py --config diagonal_arc --logdir ./logdir/[F4_B1]Diagonal --task diagonal_
```
Monitor results:
```
tensorboard --logdir ./logdir
```
### Method 2: Docker

Please refer to the Dockerfile for the instructions, as they are included within.

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Results
#### DMC Proprio
![dmcproprio](imgs/dmcproprio.png)
#### DMC Vision
![dmcvision](imgs/dmcvision.png)
#### Atari 100k
![atari100k](https://github.com/NM512/dreamerv3-torch/assets/70328564/0da6d899-d91d-44b4-a8c4-d5b37413aa11)

#### Crafter
<img src="https://github.com/NM512/dreamerv3-torch/assets/70328564/a0626038-53f6-4300-a622-7ac257f4c290" width="300" height="150" />

## Error Solution

AttributeError: 'NoneType' object has no attribute 'glGetError':

```
pip install pyrender
```

ImportError: ('Unable to load OpenGL library', 'OSMesa: cannot open shared object file: No such file or directory', 'OSMesa', None):
```
sudo apt update
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f
```

error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
```
pip install setuptools==65.5.0 pip==21 
```

TypeError: deprecated() got an unexpected keyword argument 'name'
```
pip install pyOpenSSL --upgrade
```

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
