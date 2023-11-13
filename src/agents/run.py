import argparse

from agents.agent import Agent
from agents.environment import Environment
from agents.sop import SOP
from utils import setup_logging


def init(config):
    sop = SOP.from_config(config)
    agents, roles_to_names, names_to_roles = Agent.from_config(config)
    environment = Environment.from_config(config)
    environment.agents = agents
    environment.roles_to_names, environment.names_to_roles = roles_to_names, names_to_roles
    sop.roles_to_names, sop.names_to_roles = roles_to_names, names_to_roles
    for name, agent in agents.items():
        agent.environment = environment
    return agents, sop, environment


def run(agents, sop, environment):
    while not sop.finished:
        curr_state, curr_agent = sop.next(environment, agents)

        user_input = ""
        if curr_agent.is_user:
            user_input = input(f"{curr_agent.name}: ")

        action = curr_agent.step(curr_state, user_input)
        memory = action.process()
        environment.update_memory(memory, curr_state)


parser = argparse.ArgumentParser(description='A demo of chatbot')
parser.add_argument('--agent', type=str, help='path to SOP json')
args = parser.parse_args()

setup_logging()
agents, sop, environment = init(args.agent)
run(agents, sop, environment)
