class State:

    def __init__(self, name: str, state_desc: str, roles_info: dict[str, dict],
                 begin_role: str, begin_query: str):
        self.name = name
        self.state_desc = state_desc
        self.roles = list(roles_info.keys())

        self.is_begin = True
        self.begin_role = begin_role
        self.begin_query = begin_query

        self.curr_role = self.begin_role
        self.index = self.roles.index(self.begin_role)
        self.init_components(roles_info)

    def update_to_next_role(self):
        self.index += 1
        self.index %= len(self.roles)
        self.curr_role = self.roles[self.index]
        return self.curr_role

    def init_components(self, roles_info: dict) -> None:
        self.components = {
            role: {
                key: value
                for key, value in info.items()
                if key in ['role_desc']
            } for role, info in roles_info.items()
        }
