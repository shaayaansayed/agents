SINGLE_MESSAGE_TEMPLATE = "{name}: {content}"


class Memory:

    def __init__(self, role, name, content) -> None:
        self.send_role = role
        self.send_name = name
        self.content = content

    def get_gpt_message(self, role):
        return {"role": role, "content": self.content}

    @classmethod
    def get_chat_history(self, messages):
        chat_history_parts = []
        for message in messages:
            name, content = message.send_name, message.content
            single_message = SINGLE_MESSAGE_TEMPLATE.format(name=name,
                                                            content=content)
            chat_history_parts.append(single_message)
        return '\n'.join(chat_history_parts)

    def get_query(self):
        name, role, content = self.send_name, self.send_role, self.content
        return SINGLE_MESSAGE_TEMPLATE.format(name=name,
                                              role=role,
                                              content=content)
