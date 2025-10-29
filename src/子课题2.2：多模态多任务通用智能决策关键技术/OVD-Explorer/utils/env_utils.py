import os

from gym import Env
from gym.spaces import Box, Discrete, Tuple
import numpy as np


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        # print("lb", lb, ub)
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


def domain_to_env(name):

    from gym.envs.mujoco import HalfCheetahEnv, \
        InvertedPendulumEnv, HumanoidEnv, \
        HopperEnv, AntEnv, Walker2dEnv, HumanoidStandupEnv, \
        InvertedDoublePendulumEnv, StochasticInvertedPendulumEnv, StochasticInvertedDoublePendulumEnv,\
        ReacherEnv, SwimmerEnv, PusherEnv, ThrowerEnv, StrikerEnv, SparseHalfCheetahEnv, \
        SparseHopperEnv, SparseAntEnv, SparseHumanoidEnv, SparseWalker2dEnv, SparseHumanoidStandupEnv, \
        StochasticPusherEnv, StochasticAntEnv, StochasticHopperEnv, StochasticWalker2dEnv, StochasticHalfcheetahEnv, StochasticHumanoidEnv
    from gym.envs.diy import GridChaosEnv
    return {
        'gridchaos': GridChaosEnv,
        'invertedpendulum': InvertedPendulumEnv,
        'inverteddoublependulum': InvertedDoublePendulumEnv,
        'humanoid': HumanoidEnv,
        'halfcheetah': HalfCheetahEnv,
        'sparsehalfcheetah': SparseHalfCheetahEnv,
        'sparsehopper': SparseHopperEnv,
        'sparseant': SparseAntEnv,
        'sparsewalker2d': SparseWalker2dEnv,
        'sparsehumanoid': SparseHumanoidEnv,
        'sparsehumanoidstandup': SparseHumanoidStandupEnv,
        'stochasticant': StochasticAntEnv,
        'stochastichopper': StochasticHopperEnv,
        'stochasticwalker2d': StochasticWalker2dEnv,
        "stochastichalfcheetah": StochasticHalfcheetahEnv,
        "stochastichumanoid": StochasticHumanoidEnv,
        "stochasticpusher": StochasticPusherEnv,
        "stochasticinvertedpendulum": StochasticInvertedPendulumEnv,
        "stochasticinverteddoublependulum": StochasticInvertedDoublePendulumEnv,
        'hopper': HopperEnv,
        'ant': AntEnv,
        'walker2d': Walker2dEnv,
        'humanoidstandup': HumanoidStandupEnv,
        'reacher':ReacherEnv,
        'swimmer':SwimmerEnv,
        'pusher':PusherEnv,
        'thrower':ThrowerEnv,
        'striker':StrikerEnv,
    }[name]


def domain_to_epoch(name):

    return {
        'gridchaos': 1250,
        'invertedpendulum': 250,
        'humanoid': 3000,
        'halfcheetah': 1250,
        'hopper': 1250,
        'ant': 2500,
        'walker2d': 1250,
        'sparsehopper': 1250,
        'stochasticant': 1250,
        'sparseant': 2500,
        'humanoidstandup': 1000,
        'inverteddoublependulum': 300,
        'reacher':250,
        'swimmer':2500,
        'pusher':2500,
        'thrower':2500,
        'striker':2500,
        'MountainCarContinuous': 1000,
        'sparsehalfcheetah': 2500,
        'sparsehumanoid': 3000,
        'sparsewalker2d': 1250,
        'sparsehumanoidstandup': 250,
        'stochastichopper': 1250,
        'stochasticwalker2d': 1000,
        "stochastichalfcheetah": 1250,
        "stochastichumanoid": 1250,
        "stochasticinvertedpendulum": 250,
        "stochasticinverteddoublependulum": 300,
        "stochasticpusher": 1250,

    }[name]


def env_producer(domain, seed):

    if domain in ["MountainCarContinuous"]:
        import gym
        env = gym.make("MountainCarContinuous-v0")
    else:
        env = domain_to_env(domain)()
        env.seed(seed)
        env = NormalizedBoxEnv(env)

    return env
