
from typing import Callable
import numpy as np
import pytest
from unityagents import UnityEnvironment as Env
from Navigation import LearningParameters, learn_agent_dqn, perform_action_in_env, plot_scores, watch_agent_perform
from dqn_agent import Agent
from tests.conftest import agent_no_graphics, brain_name, brain_number, env_no_graphics, saved_agent


def random_action_agent(sn, eps , act_sz):
    return np.random.randint(act_sz)

def get_agent_step_fn(agent: Agent):
    def agent_step_fn(sn, eps, act_sz):
        return agent.act(sn, eps)
    return agent_step_fn

@pytest.mark.skip
def test_watch_random_agent_perform(unity_env: Env):
    # get the default brain

    brain_number = 0
    watch_agent_perform(unity_env,
                        brain_number,
                        train_mode=False,
                        agent_action_fn=random_action_agent)


@pytest.mark.skip
def test_watch_agent(env: Env):
    # get the default brain
    brain_number = 0
    watch_agent_perform(env,
                        brain_number,
                        train_mode=False,
                        agent_action_fn=random_action_agent)

@pytest.mark.skip
def test_learning_agent(
    env: Env, agent: Agent, learning_parameters: LearningParameters, brain_number
):
    smart_agent, scores = learn_agent_dqn(env, agent, learning_parameters)
    plot_scores(scores)

    watch_agent_perform(env,
                        brain_number,
                        train_mode=False,
                        agent_action_fn=get_agent_step_fn(smart_agent))

def test_learning_agent_no_graphics(
    env_no_graphics: Env, agent_no_graphics: Agent, learning_parameters: LearningParameters,
):
    _, scores = learn_agent_dqn(env_no_graphics, agent_no_graphics, learning_parameters)
    env_no_graphics.close()
    plot_scores(scores)



def test_watch_learned_agent(
    env: Env, saved_agent: Agent, learning_parameters: LearningParameters
):

    brain_number = 0
    watch_agent_perform(env,
                        brain_number,
                        train_mode=False,
                        agent_action_fn=get_agent_step_fn(saved_agent))

@pytest.mark.skip
def test_can_call_perform_action_in_env(env, agent, brain_name):
    eps = 1.0
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    perform_action_in_env(env, agent, eps, state)

@pytest.mark.skip
def test_env_no_graphics(env_no_graphics):
    assert env_no_graphics is not None
@pytest.mark.skip
def test_agent_no_graphics(agent_no_graphics):
    assert agent_no_graphics is not None