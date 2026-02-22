import re

from reasoning.rewards.reward_function_abc import RewardFunctionABC
try:
    from fast_langdetect import detect_language
except ImportError:
    def detect_language(text):
        return "en"  # Default fallback for testing


class XMLFormatReward(RewardFunctionABC):
    __name__ = "xml_format"

    def __init__(self, tokens: list[str] = None, reasoning_tag: str = "reasoning"):
        """A reward function that rewards completions that contain any of the tokens in the list.

        Args:
            tokens (list[str]): A list of tokens to reward. If None, will use reasoning_tag to generate default tokens.
            reasoning_tag (str): The name of the reasoning tag to use (defaults to "reasoning").
        """
        super().__init__()
        if tokens is None:
            # Generate default tokens based on reasoning_tag
            tokens = [f"<{reasoning_tag}>", f"</{reasoning_tag}>", "<answer>", "</answer>"]
            self._reasoning_tag = reasoning_tag
        else:
            # Extract reasoning_tag from tokens if provided
            if len(tokens) >= 2 and tokens[0].startswith('<') and tokens[1].startswith('</'):
                # Extract tag name from <tag> format
                tag_name = tokens[0][1:-1]  # Remove < and >
                self._reasoning_tag = tag_name
            else:
                self._reasoning_tag = reasoning_tag
        self._tokens = tokens

    def __call__(self, completions, answer_only, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion, is_answer_only in zip(completions, answer_only):
            reward = 0.0

            if is_answer_only:
                if completion.count("<answer>") == 1:
                    reward += 0.5
                if completion.count("</answer>") == 1:
                    reward += 0.5
                reward -= (len(completion.split("<answer>")[0]) - 1) * 0.001
                reward -= (len(completion.split("</answer>")[-1]) - 1) * 0.001
                rewards.append(reward)
                continue

            for token in self._tokens:
                if completion.count(token) == 1:
                    reward += 0.125
            # Penalize content before the first token:
            reward -= (len(completion.split(self._tokens[0])[0]) - 1) * 0.001
            # Penalize extra content after the last token:
            reward -= (len(completion.split(self._tokens[-1])[-1]) - 1) * 0.001

            # Penalize content after </reasoning_tag> but before <answer>:
            reasoning_end_tag = f"</{self._reasoning_tag}>"
            reasoning_end = completion.find(reasoning_end_tag)
            answer_start = completion.find("<answer>")
            if (
                reasoning_end != -1
                and answer_start != -1
                and reasoning_end < answer_start
            ):
                reward -= (answer_start - reasoning_end - len(reasoning_end_tag)) * 0.001
            rewards.append(reward)
        return rewards


class XMLMultiStepFormatReward(RewardFunctionABC):
    __name__ = "xml_multi_step_format"

    def __init__(self, num_thoughts):
        super().__init__()
        self._num_thoughts = num_thoughts

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            reward = 0.0
            thought_openings = completion.split("<thought>")
            if len(thought_openings) - 1 <= self._num_thoughts:
                reward += 0.1 * (len(thought_openings) - 1)
            else:
                reward += 0.5 - 0.1 * (len(thought_openings) - 1)
            thought_closings = completion.split("</thought>")
            if len(thought_closings) - 1 <= self._num_thoughts:
                reward += len(thought_closings) - 1
            else:
                reward += 0.5 - 0.1 * (len(thought_closings) - 1)
            if "<answer>" in completion:
                reward += 0.5
            if "</answer>" in completion:
                reward += 0.5

            snips = [
                snip.split("</thought>")[-1] for snip in completion.split("<thought>")
            ]
            for snip in snips[:-1]:
                reward -= 1 - len(snip) * 0.001
            rewards.append(reward)
        return rewards


class SoftFormatReward(RewardFunctionABC):
    __name__ = "soft_format"

    def __init__(self, reasoning_tag: str = "reasoning", **kwargs):
        """A reward function that rewards completions matching the soft format pattern.
        
        Args:
            reasoning_tag (str): The name of the reasoning tag to use (defaults to "reasoning").
        """
        super().__init__(**kwargs)
        self._reasoning_tag = reasoning_tag
        self._pattern = rf"<{reasoning_tag}>(.*?)</{reasoning_tag}>\s*<answer>(.*?)</answer>"

    def __call__(self, completions, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(self._pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]


class HardFormatReward(RewardFunctionABC):
    __name__ = "hard_format"

    def __init__(self, min_group_lengths: list[int], max_group_lengths: list[int], reasoning_tag: str = "reasoning", **kwargs):
        """A reward function that rewards completions that match the pattern and have groups of the specified lengths.

        Args:
            min_group_lengths (list[int]): A list of minimum lengths for each group in the pattern.
            max_group_lengths (list[int]): A list of maximum lengths for each group in the pattern.
            reasoning_tag (str): The name of the reasoning tag to use (defaults to "reasoning").
        """
        super().__init__(**kwargs)
        self._min_group_lengths = min_group_lengths
        self._max_group_lengths = max_group_lengths
        self._reasoning_tag = reasoning_tag
        self._pattern = rf"<{reasoning_tag}>(.*?)</{reasoning_tag}>\s*<answer>(.*?)</answer>"

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            matched = re.search(self._pattern, completion, re.DOTALL)
            if matched:
                length_constraints = [
                    len(group) >= min_length
                    for group, min_length in zip(
                        matched.groups(), self._min_group_lengths
                    )
                ]
                length_constraints += [
                    len(group) <= max_length if max_length > 0 else True
                    for group, max_length in zip(
                        matched.groups(), self._max_group_lengths
                    )
                ]
                if all(length_constraints):
                    rewards.append(1.0)
                else:
                    rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards


class MultiStepFormatReward(RewardFunctionABC):
    __name__ = "multistep_format"

    def __init__(self, reasoning_tag: str = "thought", **kwargs):
        """A reward function that rewards completions that match the multi-step pattern.
        
        Args:
            reasoning_tag (str): The name of the reasoning tag to use (defaults to "thought" for multi-step).
        """
        super().__init__(**kwargs)
        self._reasoning_tag = reasoning_tag
        self._pattern = rf"((<{reasoning_tag}>(.*?)</{reasoning_tag}>\s*)+)\s*<answer>(.*?)</answer>"

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            matched = re.search(self._pattern, completion, re.DOTALL)
            if matched:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards


class LanguageReward(RewardFunctionABC):
    def __init__(self, language: str):
        super().__init__()
        self._language = language

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            if detect_language(completion) == self._language:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards


class ModularStepFormatReward(RewardFunctionABC):
    __name__ = "modular_step_format"

    def __init__(self, step_names: list[str], min_step_lengths: list[int] = None, max_step_lengths: list[int] = None, **kwargs):
        """A reward function that rewards completions with modular reasoning steps.
        
        Args:
            step_names (list[str]): Names of the reasoning steps (e.g., ["step1", "step2", "step3"]).
            min_step_lengths (list[int]): Minimum character lengths for each step. Defaults to [10] * num_steps.
            max_step_lengths (list[int]): Maximum character lengths for each step (-1 for no limit). Defaults to [-1] * num_steps.
        """
        super().__init__(**kwargs)
        self._step_names = step_names
        self._num_steps = len(step_names)
        
        if min_step_lengths is None:
            min_step_lengths = [10] * self._num_steps
        if max_step_lengths is None:
            max_step_lengths = [-1] * self._num_steps
            
        self._min_step_lengths = min_step_lengths
        self._max_step_lengths = max_step_lengths
        
        # Create dynamic regex pattern based on step names
        step_patterns = [f"<{step}>(.*?)</{step}>" for step in step_names]
        self._pattern = r"\s*".join(step_patterns) + r"\s*<answer>(.*?)</answer>"

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            matched = re.search(self._pattern, completion, re.DOTALL)
            if matched:
                # Check length constraints for each step
                groups = matched.groups()
                length_constraints = []
                
                # Check minimum lengths for all steps
                for i, (group, min_length) in enumerate(zip(groups[:self._num_steps], self._min_step_lengths)):
                    length_constraints.append(len(group.strip()) >= min_length)
                
                # Check maximum lengths for all steps (if specified)
                for i, (group, max_length) in enumerate(zip(groups[:self._num_steps], self._max_step_lengths)):
                    if max_length > 0:
                        length_constraints.append(len(group.strip()) <= max_length)
                
                # Answer should have at least 1 character
                length_constraints.append(len(groups[self._num_steps].strip()) >= 1)
                
                if all(length_constraints):
                    rewards.append(1.0)
                else:
                    rewards.append(0.3)  # Partial reward for correct format but wrong lengths
            else:
                rewards.append(0.0)
        return rewards


# Keep the old class name for backward compatibility
class TwoStageFormatReward(RewardFunctionABC):
    __name__ = "two_stage_format"

    def __init__(self, min_think_lengths: list[int] = [10, 10], **kwargs):
        """A reward function that rewards completions with two-stage thinking format.
        
        Format: <think>...</think> <confidence>...</confidence> <think>...</think> <answer>...</answer>
        
        Args:
            min_think_lengths (list[int]): Minimum character lengths for each think section.
        """
        super().__init__(**kwargs)
        self._min_think_lengths = min_think_lengths
        self._pattern = r"<think>(.*?)</think>\s*<confidence>(.*?)</confidence>\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    def __call__(self, completions, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        for completion in completions:
            matched = re.search(self._pattern, completion, re.DOTALL)
            if matched:
                groups = matched.groups()
                think1, confidence, think2, answer = groups
                
                # Check length constraints for think sections
                # Check confidence format: must be 1-3 digit number between 0-100
                confidence_valid = (
                    re.match(r'^\d{1,3}$', confidence.strip()) and 
                    0 <= int(confidence.strip()) <= 100
                ) if confidence.strip() else False
                
                length_constraints = [
                    len(think1.strip()) >= self._min_think_lengths[0],
                    len(think2.strip()) >= self._min_think_lengths[1] if len(self._min_think_lengths) > 1 else len(think2.strip()) >= 10,
                    len(answer.strip()) >= 1,
                    confidence_valid  # Enforce numerical confidence format
                ]
                
                if all(length_constraints):
                    rewards.append(1.0)
                else:
                    rewards.append(0.3)  # Partial reward for correct format but insufficient content
            else:
                rewards.append(0.0)
        return rewards


class ThreeStepFormatReward(ModularStepFormatReward):
    __name__ = "three_step_format"

    def __init__(self, min_step_lengths: list[int] = [10, 10, 10], max_step_lengths: list[int] = [-1, -1, -1], **kwargs):
        """A reward function that rewards completions with three reasoning steps.
        
        Args:
            min_step_lengths (list[int]): Minimum character lengths for each step.
            max_step_lengths (list[int]): Maximum character lengths for each step (-1 for no limit).
        """
        super().__init__(
            step_names=["step1", "step2", "step3"],
            min_step_lengths=min_step_lengths,
            max_step_lengths=max_step_lengths,
            **kwargs
        )
