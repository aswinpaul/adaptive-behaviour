from __future__ import annotations

import numpy as np
import pandas as pd

import pongGymEnv as pongEnv
from pong_statetoobs import state_to_obs

# Dimensions of the experiment-informed generative model.
N_BALL_X, N_BALL_Y, N_PADDLE = 38, 8, 8
N_STATES = N_BALL_X * N_BALL_Y * N_PADDLE      # 2432
N_ACTIONS = 3                                   # Stay, Play-Up, Play-Down
STAY, UP, DOWN = 0, 1, 2

N_EPISODES = 70          # matches the published runners
REWARD_CAP = 100         # episode is cut short above this cumulative reward

_env = None


def get_env():
    """The Pong environment is expensive to construct, so it is made once and reused.

    NOT THREAD-SAFE. This returns a single shared instance, and the environment additionally
    keeps module-level state in ``readState`` (notably the ``games`` list, rebuilt into a
    DataFrame on every rally end). Running seeds concurrently must therefore use separate
    *processes* -- see ``run_baseline.py``. Threads interleave ``step`` calls on one environment
    and corrupt every run involved.
    """
    global _env
    if _env is None:
        _env = pongEnv.CartPoleEnv()
        _env.reset(5000)
    return _env


def flat_state(obs) -> int:
    """Flatten [ball-x, ball-y, paddle-y] into a single index in [0, 2432)."""
    return obs[0] * (N_BALL_Y * N_PADDLE) + obs[1] * N_PADDLE + obs[2]


class Policy:
    """Interface expected by :func:`run_seed`.

    Subclasses override :meth:`act`, and optionally :meth:`learn` and :meth:`start_episode`.
    """

    def start_episode(self) -> None:
        pass

    def act(self, obs) -> int:
        raise NotImplementedError

    def learn(self, obs, action, next_obs, hit: bool, done: bool) -> None:
        pass


def run_seed(policy: Policy, seed: int, n_episodes: int = N_EPISODES) -> pd.DataFrame:
    """Run one seed of ``n_episodes`` episodes and return a published-schema frame."""
    env = get_env()
    lengths = np.zeros((n_episodes, 2))
    tau_total = 0

    for episode in range(n_episodes):
        state = env.reset(seed)
        obs = state_to_obs(state)
        policy.start_episode()

        done = False
        reward = 0.0
        while not done:
            prev_reward = reward
            action = policy.act(obs)
            next_state, reward, done, _ = env.step(int(action))
            next_obs = state_to_obs(next_state)
            hit = reward > prev_reward

            policy.learn(obs, action, next_obs, hit, done)
            obs = next_obs

            if reward > REWARD_CAP:
                done = True
            tau_total += 1

        lengths[episode, 0] = reward
        lengths[episode, 1] = tau_total

    return build_frame(lengths, tau_total, seed, n_episodes)


def build_frame(lengths, tau_total, seed, n_episodes) -> pd.DataFrame:
    """Reproduce the published window-splitting and time-labelling exactly."""
    sep = int(tau_total / 4)
    candidates = np.argwhere(lengths[:, 1] <= sep)
    sep_episode = candidates[-1][0] if len(candidates) else 0
    sep_episode = 1 if sep_episode == 0 else sep_episode

    elapse = (lengths[:, 1] / tau_total) * 20
    half = np.array([0.0] * sep_episode + [1.0] * (n_episodes - sep_episode))

    return pd.DataFrame({
        "hit_count": lengths[:, 0],
        "session_num": np.full(n_episodes, seed, dtype=float),
        "elapse_minute_rounded": elapse.astype(int),
        "half": half,
    })


# --------------------------------------------------------------------------- baselines

class RandomPolicy(Policy):
    """Uniformly random action. The floor any learning agent must clear."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


class TrackingPolicy(Policy):
    """Hand-coded controller: move the paddle towards the ball's vertical position.

    ``sign`` selects the screen convention. Both polarities are evaluated in the notebook and
    the better one is reported as the heuristic, since the direction of increasing y in the
    environment is not documented.
    """

    def __init__(self, sign: int = +1):
        self.sign = sign

    def act(self, obs) -> int:
        ball_y, paddle_y = obs[1], obs[2]
        if paddle_y == ball_y:
            return STAY
        going_up = (paddle_y > ball_y) if self.sign > 0 else (paddle_y < ball_y)
        return UP if going_up else DOWN


class DynaQPolicy(Policy):
    """Tabular Dyna-Q over the flattened state space.

    Rewards mirror the risk signal the counterfactual agent receives: +1 on a hit, -1 when the
    episode ends in a miss, 0 otherwise.
    """

    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, replay=0, seed=0):
        from agents.agent_dynaq import dynaq_agent
        np.random.seed(seed)
        self.agent = dynaq_agent(numS=N_STATES, numA=N_ACTIONS, epsilon=epsilon,
                                 alpha=alpha, gamma=gamma, replay=replay)

    def act(self, obs) -> int:
        return int(self.agent.decision_making(flat_state(obs)))

    def learn(self, obs, action, next_obs, hit, done) -> None:
        reward = 1.0 if hit else (-1.0 if done else 0.0)
        self.agent.learning_with_replay(flat_state(obs), int(action),
                                        flat_state(next_obs), reward)


BASELINES = {
    "Random": lambda seed: RandomPolicy(seed=seed),
    "Heuristic": lambda seed: TrackingPolicy(sign=+1),
    "Heuristic-inv": lambda seed: TrackingPolicy(sign=-1),
    "DynaQ": lambda seed: DynaQPolicy(seed=seed),
    "DynaQ-replay": lambda seed: DynaQPolicy(seed=seed, replay=10),
}
