import json
import os

import torch
from agents.llm import init_llm
from agents.memory import Memory
from agents.utils import get_embedding, get_relevant_history

SUMMARY_INFO_TEMPLATE = "Here is the information you need to know:\n\nHere is the summary of the previous dialogue history:\n{summary}\nHere is the latest conversation record:\n{chat_history},\nHere is the relevant chat history you may need:{relevant_history}"

SUMMARY_SYSTEM_PROMPT = "{current_memory};\nYour task is to summarize the historical dialogue records according to the current scene, and summarize the most important information.;\n"

AGENT_MEMORY = "Here's what you need to know (remember, this is just information, do not repeat what's inside):\n\nHere is the relevant chat history you may need:{relevant_memory}\n\nHere is the previous summary of chat history:{agent_short_term_memory}\n\nHere is the relevant memory:{agent_relevant_memory}\n\nHere is the new chat history:\n\n{new_conversations}"


class Environment:
    """
    The place where the agent activities, responsible for storing some shared memories
    """

    def __init__(self, config) -> None:
        self.shared_memory = {"long_term_memory": [], "short_term_memory": None}
        self.agents = None
        self.curr_chat_history_idx = 0
        self.llms = {}
        self.roles_to_names = None
        self.names_to_roles = None

        for state_name, state_dict in config["states"].items():
            if state_name == "end_state":
                continue

            self.llms[state_name] = init_llm()

    @classmethod
    def from_config(cls, config_path):
        with open(config_path) as f:
            config = json.load(f)
        return cls(config)

    def summary(self, current_state):
        """
        Summarize the situation in the current environment every once in a while
        """
        max_chat_history = int(os.environ["MAX_CHAT_HISTORY"])

        long_term_memory = self.shared_memory["long_term_memory"]
        chat_embeddings = self.shared_memory["chat_embeddings"]

        relevant_history = ""
        if len(self.shared_memory["long_term_memory"]) > 1:
            query = self.shared_memory["long_term_memory"][-1].content
            relevant_history = get_relevant_history(query,
                                                    long_term_memory[:-1],
                                                    chat_embeddings[:-1])
            relevant_history = Memory.get_chat_history(relevant_history)

        chat_history_slice = slice(-max_chat_history + 1, None)
        chat_history = Memory.get_chat_history(
            long_term_memory[chat_history_slice])
        summary = self.shared_memory["short_term_memory"]

        current_memory = SUMMARY_INFO_TEMPLATE.format(
            chat_history=chat_history,
            summary=summary,
            relevant_history=relevant_history)

        system_prompt = SUMMARY_SYSTEM_PROMPT.format(
            current_memory=current_memory)
        response = self.llms[current_state.name].get_response(None,
                                                              system_prompt,
                                                              stream=False)
        return response

    def update_memory(self, memory, curr_state):
        max_chat_history = int(os.environ.get("MAX_CHAT_HISTORY"))
        self.shared_memory["long_term_memory"].append(memory)

        embedding = get_embedding(memory.content)
        chat_embeddings = self.shared_memory.get("chat_embeddings",
                                                 torch.tensor([]))
        self.shared_memory["chat_embeddings"] = torch.cat(
            [chat_embeddings, embedding], dim=0)

        if len(self.shared_memory["long_term_memory"]) % max_chat_history == 0:
            self.shared_memory["short_term_memory"] = self.summary(curr_state)

        self.agents[memory.send_name].update_memory(memory, curr_state)

    def _get_agent_last_conversation_idx(self, agent, long_term_memory):
        return next(
            (i for i, history in reversed(list(enumerate(long_term_memory)))
             if history.send_name == agent.name), -1)

    def _get_agent_new_memory(self, agent, long_term_memory):
        last_conversation_idx = self._get_agent_last_conversation_idx(
            agent, long_term_memory)
        new_conversation = long_term_memory[
            last_conversation_idx +
            1:] if last_conversation_idx != -1 else long_term_memory

        max_chat_history = int(os.environ.get("MAX_CHAT_HISTORY"))
        if len(new_conversation) > 2 * max_chat_history:
            new_conversation = new_conversation[-2 * max_chat_history + 1:]

        return Memory.get_chat_history(new_conversation)

    def observe(self, agent):
        max_chat_history = int(os.environ.get("MAX_CHAT_HISTORY"))

        long_term_memory = self.shared_memory["long_term_memory"][
            self.curr_chat_history_idx:]
        long_term_memory = long_term_memory[-max_chat_history + 1:] if len(
            long_term_memory) > max_chat_history else long_term_memory

        chat_embeddings = self.shared_memory["chat_embeddings"][
            self.curr_chat_history_idx:]
        chat_embeddings = chat_embeddings[-max_chat_history + 1:] if len(
            chat_embeddings) > max_chat_history else chat_embeddings

        relevant_memory = ""
        if len(long_term_memory) > 1:
            query = long_term_memory[-1].content
            relevant_memory = get_relevant_history(query, long_term_memory[:-2],
                                                   chat_embeddings[:-2])
            relevant_memory = Memory.get_chat_history(relevant_memory,
                                                      agent.name)

        agent.relevant_memory = relevant_memory
        new_conversations = self._get_agent_new_memory(agent, long_term_memory)

        current_memory = AGENT_MEMORY.format(
            relevant_memory=relevant_memory,
            agent_short_term_memory=agent.short_term_memory,
            new_conversations=new_conversations,
            agent_relevant_memory=agent.relevant_memory)

        return {"role": "user", "content": current_memory}
