import json
import os


class Config:

    def __init__(self, config_path):
        self.config_path = config_path
        self.agents = {}
        self.roles_to_names = {}
        self.names_to_roles = {}

        self._parse_config()

    def _parse_config(self):
        with open(self.config_path, 'r') as file:
            config = json.load(file)
