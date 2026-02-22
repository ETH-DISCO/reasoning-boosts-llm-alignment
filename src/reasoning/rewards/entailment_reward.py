from reasoning.rewards.reward_function_abc import RewardFunctionABC
from transformers import pipeline


def _extract_reasoning(completion: str) -> str:
    completion = completion.lower()
    return completion.split("<reasoning>")[-1].split("</reasoning>")[0].strip()


class EntailmentReward(RewardFunctionABC):
    __name__ = "entailment_reward"

    def __init__(
        self,
        model_name_or_path: str,
        pos_options: list[str] = ["A", "B"],
        neg_options: list[str] = ["C", "D"],
    ):
        super().__init__()
        self._clf = pipeline(
            "text-classification",
            model=model_name_or_path,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        self._pos_options = pos_options
        self._neg_options = neg_options

    def _find_result(self, results, label):
        for res in results:
            if res["label"] == label:
                return res
        return results[0]

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        completions = [_extract_reasoning(completion) for completion in completions]
        for completion, answer in zip(completions, answers):
            reward = 0.0
            preds = self._clf(completion)[0]
            pred = self._find_result(
                preds, "LABEL_1" if answer in self._pos_options else "LABEL_0"
            )
            reward += 2.0 * pred["score"]
            rewards.append(reward)
        return rewards


class ProfileUsedReward(RewardFunctionABC):
    __name__ = "profile_used_reward"

    signal_list = [
        "based on the user's preferences",
        "based on the user's past",
        "based on the user's profile",
        "based on the user's history",
        "based on the user's data",
        "from the user's profile",
        "from the user's past",
        "from the user's history",
        "from the user's data",
        "from the user's preferences",
        "using the user's profile",
        "using the user's past",
        "using the user's history",
        "using the user's data",
        "using the user's preferences",
        "the user's profile",
        "the user's past",
        "the user's history",
        "the user's data",
        "the user's preferences",
        "user's profile",
        "user's past",
        "user's history",
        "user's data",
        "user's preferences",
        "according to the user's profile",
        "according to the user's past",
        "according to the user's history",
        "according to the user's data",
        "according to the user's preferences",
    ]

    def __init__(self, profile_dims: list[str]):
        super().__init__()
        self._profile_dims = profile_dims

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        completions = [_extract_reasoning(completion) for completion in completions]
        for completion, answer in zip(completions, answers):
            reward = 0.0
            for dim in self._profile_dims:
                if dim.lower() in completion.lower():
                    reward = 1.0
            for signal in self.signal_list:
                if signal.lower() in completion.lower():
                    reward += 0.5
                    break
            rewards.append(reward)
        return rewards
