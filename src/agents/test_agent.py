import json
import os
import pickle

import pytest

from agents.Agent import Agent  # Ensure this matches your project structure

TEST_DATA_DIR = "src/tests/data"


@pytest.fixture
def load_json(request):
    """Fixture to load a JSON file."""
    file_path = os.path.join(TEST_DATA_DIR, request.param)
    with open(file_path, "r") as file:
        return json.load(file)


@pytest.fixture
def load_pickle(request):
    """Fixture to load a pickle file."""
    file_path = os.path.join(TEST_DATA_DIR, request.param)
    with open(file_path, "rb") as file:
        return pickle.load(file)


@pytest.fixture
def configure_agents(request):
    """Fixture to configure agents from a config file."""
    config_path = os.path.join("examples", "SingleAgent", request.param,
                               "config.json")
    return Agent.from_config(config_path)


def are_agents_equal(agent1, agent2):
    """Check if two Agent instances are equal."""
    if not isinstance(agent1, Agent) or not isinstance(agent2, Agent):
        return False

    attributes_to_compare = [
        'name', 'state_roles', 'style', 'begins', 'current_role',
        'long_term_memory', 'short_term_memory', 'current_state', 'first_speak'
    ]

    return all(
        getattr(agent1, attr, None) == getattr(agent2, attr, None)
        for attr in attributes_to_compare)


@pytest.mark.parametrize("configure_agents", ["chatbot"], indirect=True)
@pytest.mark.parametrize("load_json", ["chatbot/roles_to_names.json"],
                         indirect=True)
def test_roles_to_names(configure_agents, load_json):
    _, roles_to_names, _ = configure_agents
    assert load_json == roles_to_names


@pytest.mark.parametrize("configure_agents", ["chatbot"], indirect=True)
@pytest.mark.parametrize("load_json", ["chatbot/names_to_roles.json"],
                         indirect=True)
def test_names_to_roles(configure_agents, load_json):
    _, _, names_to_roles = configure_agents
    assert load_json == names_to_roles


@pytest.mark.parametrize("configure_agents", ["chatbot"], indirect=True)
@pytest.mark.parametrize("load_pickle", ["chatbot/agents.pickle"],
                         indirect=True)
def test_agents(configure_agents, load_pickle):
    agents, _, _ = configure_agents
    for agent_name, expected_agent_info in load_pickle.items():
        assert agent_name in agents
        assert are_agents_equal(expected_agent_info, agents[agent_name])
