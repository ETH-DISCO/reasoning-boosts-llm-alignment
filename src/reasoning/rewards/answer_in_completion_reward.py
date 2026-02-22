from reasoning.rewards.reward_function_abc import RewardFunctionABC


def _extract_answer(completion: str) -> str:
    completion = completion.lower()
    return completion.split("<answer>")[-1].split("</answer>")[0].strip()


class AnswerDistanceReward(RewardFunctionABC):
    __name__ = "answer_distance"
    tags = ["<answer>", "</answer>"]

    def __init__(self, options: list[str], values: list[float], **kwargs):
        super().__init__(**kwargs)
        self._options = [option.lower() for option in options]
        self._values = values

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        tags_in_completion = [
            all(tag in completion for tag in self.tags) for completion in completions
        ]
        # completions = [_extract_answer(completion) for completion in completions]
        for completion, answer, tags in zip(completions, answers, tags_in_completion):
            if "<answer>" or "</answer>" in completion:
                completion = _extract_answer(completion)
                completion = completion.lower().replace("rather", "").strip()
                reward = 0.0
                for i, option in enumerate(self._options):
                    answer_area = min(len(completion), 5)
                    if option.lower() in completion[0:answer_area].lower().strip():
                        completion_pos = i
                        answer_pos = self._options.index(answer.lower())
                        reward = 2.0 - 2.0 * abs(
                            self._values[completion_pos] - self._values[answer_pos]
                        )
                        break
                rewards.append(reward)
            else:
                answers_present = 0
                answer_char = ""
                if "a) yes" in completion.lower():
                    answers_present += 1
                    answer_char = "a"
                if "b) no" in completion.lower():
                    answers_present += 1
                    answer_char = "b"
                if "c) neutral" in completion.lower():
                    answers_present += 1
                    answer_char = "c"
                if answers_present == 1:
                    completion_pos = self._options.index(answer_char)
                    answer_pos = self._options.index(answer.lower())
                    reward = 2.0 - 2.0 * abs(
                        self._values[completion_pos] - self._values[answer_pos]
                    )
                    rewards.append(reward)
                else:
                    rewards.append(0.0)
        return rewards


class RightAnswerInCompletionReward(RewardFunctionABC):
    __name__ = "right_answer_in_completion"
    tags = ["<answer>", "</answer>"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        tags_in_completion = [
            all(tag in completion for tag in self.tags) for completion in completions
        ]
        for completion, answer, tags in zip(completions, answers, tags_in_completion):
            if "<answer>" or "</answer>" in completion:
                completion = _extract_answer(completion)
                completion = completion.lower().replace("rather", "").strip()
                completion = completion.replace("yes", "").replace("no", "").strip()
                if len(completion.strip()) == 0:
                    rewards.append(-0.5)
                    continue
                answer_area = min(len(completion), 5)
                if answer.lower() in completion[0:answer_area]:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                answers_present = 0
                answer_correct = False
                if "a) yes" in completion.lower():
                    answers_present += 1
                    if answer.lower() == "a":
                        answer_correct = True
                if "b) no" in completion.lower():
                    answers_present += 1
                    if answer.lower() == "b":
                        answer_correct = True
                if "c) neutral" in completion.lower():
                    answers_present += 1
                    if answer.lower() == "c":
                        answer_correct = True
                if answers_present == 1 and answer_correct:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
        return rewards


class AnswerInCompletionReward(RewardFunctionABC):
    __name__ = "answer_in_completion"
    tags = ["<answer>", "</answer>"]

    def __init__(self, options: list[str], **kwargs):
        super().__init__(**kwargs)
        self._options = [option.lower() for option in options]

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        tags_in_completion = [
            all(tag in completion for tag in self.tags) for completion in completions
        ]
        for completion, tags in zip(completions, tags_in_completion):
            if not tags:
                rewards.append(0.0)
                continue
            # Extract just the answer part and check against options
            extracted_answer = _extract_answer(completion).replace("rather", "").strip()
            if extracted_answer in self._options:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards
