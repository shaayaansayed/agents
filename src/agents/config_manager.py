import json
from typing import Any, Dict, Optional, Tuple


class ConfigManager:
    """Class to parse and provide access to configuration data."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._parse_config()

    def _parse_config(self) -> None:
        """Parse the configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to parse configuration: {e}")

    def __getattr__(self, name):
        """Delegate attribute lookup to self.config."""
        try:
            return getattr(self.config, name)
        except AttributeError:
            raise AttributeError(
                f"'ConfigManager' object has no attribute '{name}'")

    def get_states(self) -> Dict[str, Dict]:
        return self.config['states']

    def get_role_info_for_state(self, state_query: str) -> Optional[str]:
        """Retrieve the agent name for a given state and role."""
        return self.config.get('states').get(state_query)

    def get_agent_for_state_role(self, state_query: str,
                                 role_query: str) -> Optional[str]:
        """Get the agent name from state and role."""
        state_info = self.config['states'].get(state_query)
        if state_info:
            return state_info['roles'].get(role_query, {}).get('agent_name')
        return None

    def get_agent_state_roles(self, agent_name: str) -> Optional[str]:
        """Retrieve the states and roles for a given agent."""
        agent_state_roles = {}
        for state_name, state_info in self.get('states').items():
            for role, role_info in state_info['roles'].items():
                if role_info['agent_name'] == agent_name:
                    agent_state_roles[state_name] = role
        return agent_state_roles

    def get_state_begin_role_and_query(self, state_query) -> Tuple[str, str]:
        state_info = self.config.get('states').get(state_query)
        return state_info.get('begin_role'), state_info.get('begin_query')

    def is_agent_user(self, agent_name):
        return "is_user" in self.config.get("agents").get(agent_name)

    def get_openai_api_key(self):
        return self.config.get("config").get("API_KEY")

    def get_mem_token_limit(self):
        return self.config.get("config").get("CONV_SUMMARY_MEM_TOKEN_LIMIT")

    def get_root_state(self):
        return self.config.get('root')
