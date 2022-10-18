from collections import deque
from typing import Callable, List, NewType, Tuple
from unityagents import UnityEnvironment as Env
import numpy as np
from dataclasses import dataclass

from dqn_agent import Agent
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



Scores = NewType("Scores", List[float])
@dataclass
class LearningParameters:
    n_episodes: int
    max_t: int
    eps_start: float
    eps_end: float
    eps_decay: float
    target_score: float

    def __init__(self, n_episodes, max_t, eps_start, eps_end, eps_decay, target_score):
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_score = target_score


def learn_agent_dqn(
    env: Env, agent: Agent, learning_parameters: LearningParameters
) -> Tuple[Agent, Scores]:
    """Deep Q-Learning
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = learning_parameters.eps_start  # initialize epsilon

    for i_episode in range(1, learning_parameters.n_episodes + 1):
        brain_number =0
        brain_name  = env.brain_names[brain_number]
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for _ in range(learning_parameters.max_t):
            state, reward, done = perform_action_in_env(env, agent, eps, state)
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # type: ignore
        eps = max(
            learning_parameters.eps_end, learning_parameters.eps_decay * eps
        )  # decrease epsilon
        log_scores(scores_window, i_episode, learning_parameters.target_score)
        if np.mean(scores_window) >= learning_parameters.target_score:
            torch.save(agent.qnetwork_local.state_dict(), "navigation_checkpoint.pth")
            return agent, Scores(scores)
    raise Exception("Environment not solved")


def log_scores(scores_window, i_episode, target_score):
    if i_episode % 100 == 0:
        print(
            f"\nEpisode {i_episode} \tAverage Score: {np.mean(scores_window):.2f}",
            end=" ",
        )
    if np.mean(scores_window) >= target_score:
        print(
            f"\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
        )


def perform_action_in_env(env, agent: Agent, eps: float, state):
    with torch.cuda.amp.autocast():
        action = agent.act(state, eps)
        env_info = env.step(action)["BananaBrain"]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    with torch.cuda.amp.autocast():
        agent.step(state, action, reward, next_state, done)
    return next_state, reward, done


def plot_scores(scores: Scores) -> None:
    """Plot scores and optional rolling mean
    using specified window."""
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()

def print_env_info(brain, env_info):
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    # number of actions


def watch_agent_perform(env: Env,
                    brain_number: int,
                    train_mode: bool,
                    agent_action_fn:Callable[[int, float, int], int]  ):
    brain_name = env.brain_names[brain_number]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    score = 0  # initialize the score
    print_env_info(brain, env_info)
    while True:
        action = agent_action_fn(state, 0.01, action_size) # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break


def watch_agents_perform(env: Env,
                    num_brains: int,
                    train_mode: bool,
                    agent_action_fn:Callable[[int, float, int], int]  ):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    score = 0  # initialize the score
    print_env_info(brain, env_info)
    while True:
        for brain_number in range(num_brains):
            action = agent_action_fn(state, 1.0, action_size)
            brain_name = env.brain_names[brain_number]# select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[brain_number]  # get the next state
            reward = env_info.rewards[brain_number]  # get the reward
            done = env_info.local_done[brain_number]  # see if episode has finished
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break


if __name__ == "__main__":
    env = Env(file_name="Banana_Linux/Banana.x86_64")
    watch_agent_perform(env, 0, train_mode=False, agent_action_fn=lambda state, eps, action_size: 0)
    env.close()
