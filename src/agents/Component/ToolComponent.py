from abc import abstractmethod

from ..utils import extract


class ToolComponent:

    def __init__(self):
        pass

    @abstractmethod
    def func(self):
        pass


class StaticComponent(ToolComponent):
    "Return static response"

    def __init__(self, output):
        super().__init__()
        self.output = output

    def func(self, agent):
        outputdict = {"response": self.output}
        return outputdict


class ExtractComponent(ToolComponent):
    """
    Extract keywords based on the current scene and store them in the environment
    extract_words(list) : Keywords to be extracted
    system_prompt & last_prompt : Prompt to extract keywords
    """

    def __init__(
        self,
        extract_words,
        system_prompt,
        last_prompt=None,
    ):
        super().__init__()
        self.extract_words = extract_words
        self.system_prompt = system_prompt
        self.default_prompt = (
            "Please strictly adhere to the following format for outputting:\n")
        for extract_word in extract_words:
            self.default_prompt += (
                f"<{extract_word}> the content you need to extract </{extract_word}>"
            )
        self.last_prompt = last_prompt if last_prompt else self.default_prompt

    def func(self, agent):
        response = agent.LLM.get_response(agent.long_term_memory,
                                          self.system_prompt, self.last_prompt)
        for extract_word in self.extract_words:
            key = extract(response, extract_word)
            key = key if key else response
            agent.environment.shared_memory[extract_word] = key

        return {}
