class State:

    def __init__(self, **kwargs):
        self.next_states = {}
        self.name = kwargs.get("name")
        self.env_desc = kwargs.get("env_desc", None)

        roles = kwargs.get("roles", {})

        self.is_begin = True
        self.begin_role = kwargs.get("begin_role")
        self.begin_query = kwargs.get("begin_query")
        self.current_role = self.begin_role
        self.index = list(roles.keys()).index(
            self.begin_role) if self.begin_role in roles.keys() else 0
        self.env_desc = kwargs.get("env_desc")
        self.components = self.init_components(roles) if roles else {}
        self.chat_nums = 0

    def init_components(self, agent_states_dict: dict):
        agent_states = {}
        for role, role_components in agent_states_dict.items():
            components = {}
            for component, component_args in role_components.items():
                if component == "role_desc":
                    components[component] = component_args["role_desc"]
                else:
                    continue

            agent_states[role] = components

        return agent_states
