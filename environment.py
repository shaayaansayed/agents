from typing import Any, Dict

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from action import Action
from agent import Agent
from config_manager import ConfigManager
from llm import OpenAILLM
from state import State
from utils import append_memory_buffer, extract_formatted_chat_messages


class Environment:
    """
    Represents the environment where agent activities occur, responsible for
    storing shared memories.
    """

    def __init__(self, agents: Dict[str, Agent], llms: Dict[str, Any],
                 ai_agent_name: str, memory_max_token_limit: int,
                 api_key: str) -> None:
        self.agents = agents
        self.curr_chat_history_idx = 0
        self.llms = llms
        self.memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(api_key=api_key),
            max_token_limit=memory_max_token_limit,
            input_key="content",
            output_key="content",
            ai_prefix=ai_agent_name,
            human_prefix="User")

        for agent in agents.values():
            agent.environment = self

    @classmethod
    def from_config(cls, config: ConfigManager,
                    agents: Dict[str, Agent]) -> 'Environment':
        """
        Create an Environment instance from a configuration object and a list of agents.
        """
        llms = {}
        for state_name in config.get("states", {}).keys():
            if state_name != config.get("end_state"):
                llms[state_name] = OpenAILLM()

        ai_agent_name = next(
            (agent_name for agent_name in config.get("agents", {})
             if not config.is_agent_user(agent_name)), None)

        return cls(agents,
                   llms,
                   ai_agent_name,
                   memory_max_token_limit=config.get_mem_token_limit(),
                   api_key=config.get_openai_api_key())

    def get_memory(self):
        chat_messages = extract_formatted_chat_messages(self.memory)
        return self.memory.moving_summary_buffer, chat_messages

    def record_action(self, action: Action, curr_state: State):
        if action.is_user:
            append_memory_buffer(self.memory, user_message=action.response)
        else:
            append_memory_buffer(self.memory, ai_message=action.response)
