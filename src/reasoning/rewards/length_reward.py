from reasoning.rewards.reward_function_abc import RewardFunctionABC


class LengthReward(RewardFunctionABC):
    __name__ = "length"

    def __init__(self, optimal_length):
        super().__init__()
        self._optimal_length = optimal_length

    def __call__(self, completions, answer_only, **kwargs):
        completions = [completion[0]["content"] for completion in completions]
        rewards = [
            -abs(self._optimal_length - len(completion)) * 0.001
            for completion in completions
        ]
        rewards = [
            reward if not is_answer_only else 0
            for reward, is_answer_only in zip(rewards, answer_only)
        ]
        return rewards
