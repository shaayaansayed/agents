from typing import Any, Dict

import torch
from agents.agent import Agent
from agents.llm import init_llm
from agents.memory import Memory
from agents.utils import (
    append_memory_buffer,
    extract_formatted_chat_messages,
    get_embedding,
)
from config_manager import ConfigManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

SUMMARY_INFO_TEMPLATE = "Here is the information you need to know:\n\nHere is the summary of the previous dialogue history:\n{summary}\nHere is the latest conversation record:\n{chat_history},\nHere is the relevant chat history you may need:{relevant_history}"

SUMMARY_SYSTEM_PROMPT = "{current_memory};\nYour task is to summarize the historical dialogue records according to the current scene, and summarize the most important information.;\n"

AGENT_MEMORY = "Here's what you need to know (remember, this is just information, do not repeat what's inside):\n\nHere is the relevant chat history you may need:{relevant_memory}\n\nHere is the previous summary of chat history:{agent_short_term_memory}\n\nHere is the relevant memory:{agent_relevant_memory}\n\nHere is the new chat history:\n\n{new_conversations}"


class Environment:
    """
    Represents the environment where agent activities occur, responsible for
    storing shared memories.
    """

    def __init__(self, agents: Dict[str, Agent], llms: Dict[str, Any],
                 ai_agent_name: str, memory_max_token_limit: int,
                 api_key: str) -> None:
        self.shared_memory = {
            "long_term_memory": [],
            "chat_embeddings": [],
            "short_term_memory": None
        }
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
                llms[state_name] = init_llm()

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

    def update_memory(self, memory: Memory, curr_state: str):
        agent_name = memory.send_name
        if agent_name == "User":
            append_memory_buffer(self.memory, user_message=memory.content)
        else:
            append_memory_buffer(self.memory, ai_message=memory.content)

        self.shared_memory["long_term_memory"].append(memory)
