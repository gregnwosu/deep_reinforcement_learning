import pytest
import torch
from unityagents import UnityEnvironment as Env
from Navigation import LearningParameters

from dqn_agent import Agent


@pytest.fixture
def brain_number():
    return 0

@pytest.fixture
def brain(brain_name, env):
    return env.brains[brain_name]

@pytest.fixture
def brain_name(brain_number, env):
    return env.brain_names[brain_number]

@pytest.fixture()
def gpu_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

@pytest.fixture()
def saved_agent(agent: Agent):
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('navigation_checkpoint.pth'))
    return agent

@pytest.fixture()
def agent(gpu_device, brain_name, brain, env):
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    return Agent(device=gpu_device, state_size=state_size, action_size=action_size, seed=0)

@pytest.fixture()
def agent_no_graphics(gpu_device, brain_number, env_no_graphics):
    brain_name = env_no_graphics.brain_names[brain_number]
    brain = env_no_graphics.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env_no_graphics.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    return Agent(device=gpu_device, state_size=state_size, action_size=action_size, seed=0)

@pytest.fixture()
def learning_parameters():
    return LearningParameters(n_episodes=3000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_score=16.0)

@pytest.fixture()
def env():
    unity_env = Env(
        file_name="/home/greg/dev/reinfo/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64",
        worker_id=3)
    yield unity_env
    unity_env.close()

@pytest.fixture()
def delayed_graphical_env():
    created_envs = []
    def x():
        unity_env = Env(
            file_name="/home/greg/dev/reinfo/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64",
            worker_id=3)
        created_envs.append(unity_env)
        return unity_env
    yield x
    for env in created_envs:
        env.close()

@pytest.fixture()
def env_no_graphics():
    unity_env = Env(
        file_name="/home/greg/dev/reinfo/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64",
        no_graphics=True,
        worker_id=3)
    yield unity_env
    unity_env.close()
