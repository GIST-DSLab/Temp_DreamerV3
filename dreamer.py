import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

import wandb
import gymnasium

to_np = lambda x: x.detach().cpu().numpy()

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._premetrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._premetrics[name] = float(np.mean(values))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        # TODO policy_output이 이산적으로 나와야하는데 그렇지 않음 - 해당 문제 고치기
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)

        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        # 원본
        # if self._config.eval_state_mean:
        #     latent["stoch"] = latent["mean"]
        # feat = self._wm.dynamics.get_feat(latent)
        # if not training:
        #     actor = self._task_behavior.actor(feat)
        #     action = actor.mode()
        # elif self._should_expl(self._step):
        #     actor = self._expl_behavior.actor(feat)
        #     action = actor.sample()
        # else:
        #     actor = self._task_behavior.actor(feat)
        #     action = actor.sample()

        # bbox때문에 아래와 같이 수정함.
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent) # 모르는 부분: 이거는 뭐하는 거지? 왜 쓰지?
        if not training:
            actor = self._task_behavior.actor(feat)
            if self._config.use_bbox:
                action = {k: actor[k].mode() for k in actor.keys()}
            else:
                action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            if type(actor) == dict:
                action = {k: actor[k].sample() for k in actor.keys()}
            else:
                action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()

        # 위에 부분에서 action의 shape 1, 2500인데 왜 이런지 이해 못함.
        if self._config.use_bbox:
            logprob = {k: actor[k].log_prob(action[k]) for k in actor.keys()}
        else:
            logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}

        if self._config.use_bbox:
            action = {k: action[k].detach() for k in actor.keys()}
        else:
            action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "verysimple":
        from envs.simple_arc import SimpleARCEnv

        # 여기 부분 하드코딩 되어 있음
        env = SimpleARCEnv([64, 64],data_loader=None, max_grid_size=(5,5), colors=10, max_step = 1, render_mode ="ansi", render_size= None)
        env = wrappers.OneHotAction(env)
    elif suite == "diagonal":
        from envs.diagonal_arc import DiagonalARCEnv, EntireSelectionLoader

        env = DiagonalARCEnv([64, 64],data_loader=EntireSelectionLoader(data_index=config.task_index), max_grid_size=(3,3), colors=10, max_step = config.batch_length, render_mode ="ansi", render_size= None, few_shot=config.few_shot, log_dir=config.logdir.split('/')[-1], num_func=config.num_func, color_permute=config.color_permute, submit_flag=config.submit_flag, acc_flag=config.acc_flag)
        # env = DiagonalARCEnv([64, 64],data_loader=None, max_grid_size=(3,3), colors=10, max_step = 2, render_mode ="ansi", render_size= None)
        env = wrappers.OneHotAction(env)
    elif suite == "bbox-diagonal":
        from envs.bbox_diagonal_arc import BBoxDiagonalARCEnv
        from arcle.wrappers import BBoxWrapper

        # 여기 부분 하드코딩 되어 있음
        env = BBoxDiagonalARCEnv([64, 64],data_loader=None, max_grid_size=(5,5), colors=10, max_step = 2, render_mode ="ansi", render_size= None, few_shot=config.few_shot)
        # env = wrappers.NormalizeActions(env)
        env = BBoxWrapper(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    wandb.init(project=config.wandb_project_name)
    wandb.config.update(config)
    wandb.run.name = config.logdir.split('/')[-1]

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)

    # 원본
    # config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ARCLE BBOX 때문에 아래와 같이 수정함 
    if not config.use_bbox:
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    else:
        config.num_actions = {k: a.n if hasattr(a, "n") else acts.shape[0] for a, k in zip(list(acts), ['y1','x1','y2','x2','operation'])}

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        # 아래는 ARC에서 action이 tuple인 경우 때문에 추가함.
        elif type(acts) == gymnasium.spaces.tuple.Tuple:
            acts_config = [a.n for a in acts]
            if config.use_bbox:
                random_actor_list = {k: 
                tools.OneHotDist(
                    torch.zeros(target_acts).repeat(config.envs, 1)
                    )
                    for target_acts, k in zip(acts_config, ['y1','x1','y2','x2','operation']) 
                }
            else:
                random_actor_list = [
                tools.OneHotDist(
                    torch.zeros(target_acts).repeat(config.envs, 1)
                    )
                    for target_acts in acts_config]
            
            # random_actor_list = [
            #     torchd.independent.Independent(
            #         torchd.uniform.Uniform(
            #             torch.tensor(0.0).repeat(config.envs, 1),
            #             torch.tensor(float(target_acts)).repeat(config.envs, 1),
            #         ), 1)
            #     for target_acts in acts_config]
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            # 원본
            # action = random_actor.sample()
            # logprob = random_actor.log_prob(action)
            # return {"action": action, "logprob": logprob}, None

            # arcle bbox로 인해 아래와 같이 바꿈.
            if not config.use_bbox:
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
            else:
                action = {k: random_actor_list[k].sample() for k in random_actor_list.keys()}
                logprob = {k: random_actor_list[k].log_prob(action[k]) for k in random_actor_list.keys()}
                # logprob = [random_actor_list[i].log_prob(action[i]) for i in range(len(random_actor_list))]
            return {"action": action, "logprob": logprob}, None
                
        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
            use_bbox=True if config.use_bbox else False,
            option = {'adaptation': True},
            config=config,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config) # -> next(train_dataset) -> (batch_size, batch_length) -> (batch, time)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    # wandb.watch(agent)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
    
    log_model_loss = -1
    log_eval_return = -1 
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every + config.prefill:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
                option = {'adaptation': False},
                use_bbox=True if config.use_bbox else False,
                config=config,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            # best.pt 저장을 위한 부분
            if agent._metrics != {} and (log_model_loss == -1.0 or agent._premetrics['model_loss'] < log_model_loss) and (log_eval_return == -1 or log_eval_return > logger.eval_return):
                log_model_loss = agent._premetrics['model_loss']
                log_eval_return = logger.eval_return
                items_to_save = {
                    "agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                }
                torch.save(items_to_save, logdir / "best.pt")


        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
            option = {'adaptation': True},
            use_bbox=True if config.use_bbox else False,
            config=config,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
