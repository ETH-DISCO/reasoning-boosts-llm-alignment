from abc import ABC, abstractmethod

type RewardFunction = callable[[list[str], dict], float] | RewardFunctionABC


class RewardFunctionABC(ABC):
    __name__ = "reward_function_abc"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, completions: str, **kwargs):
        pass
