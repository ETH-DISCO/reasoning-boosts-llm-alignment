import re
from reasoning.rewards.reward_function_abc import RewardFunctionABC


def _extract_answer(completion: str) -> str:
    """Extract answer from completion."""
    completion = completion.lower()
    return completion.split("<answer>")[-1].split("</answer>")[0].strip()


def _extract_confidence(completion: str) -> float:
    """Extract confidence score from completion. Enforces 0-100 range with max 3 chars."""
    if "<confidence>" not in completion or "</confidence>" not in completion:
        return None
    
    confidence_raw = completion.split("<confidence>")[1].split("</confidence>")[0].strip()
    
    # Enforce max 3 characters
    if len(confidence_raw) > 3:
        return None
    
    # Must be purely numeric (no text, symbols, etc.)
    if not re.match(r'^\d{1,3}$', confidence_raw):
        return None
    
    try:
        confidence_score = float(confidence_raw)
        # Must be in 0-100 range
        if confidence_score < 0 or confidence_score > 100:
            return None
        # Return as 0-1 scale
        return confidence_score / 100
    except ValueError:
        return None


class HighConfidenceAccuracyReward(RewardFunctionABC):
    """Rewards high confidence correct answers, penalizes high confidence wrong answers (excludes neutral cases)."""
    __name__ = "high_confidence_accuracy"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 confidence_threshold: float = 70,
                 neutral_answer: str = "C",
                 correct_reward: float = 1.0,
                 wrong_penalty: float = -0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.neutral_answer = neutral_answer.upper().strip()
        self.correct_reward = correct_reward
        self.wrong_penalty = wrong_penalty

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
                
            confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
            predicted_clean = predicted_answer.upper().strip()
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            if confidence_percentage >= self.confidence_threshold:
                # Skip neutral cases - let high_confidence_neutral_penalty handle them
                if predicted_clean == self.neutral_answer:
                    if is_correct:
                        rewards.append(self.correct_reward)  # Still reward correct neutrals
                    else:
                        rewards.append(0.0)  # Let neutral penalty handle wrong neutrals
                elif is_correct:
                    rewards.append(self.correct_reward)
                else:
                    rewards.append(self.wrong_penalty)
            else:
                rewards.append(0.0)  # Not high confidence, no reward from this component
        
        return rewards


class LowConfidenceHumilityReward(RewardFunctionABC):
    """Rewards cautious behavior when confidence is low (excludes neutral bonus cases)."""
    __name__ = "low_confidence_humility"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 confidence_threshold: float = 70,
                 neutral_threshold: float = 50,
                 neutral_answer: str = "C",
                 correct_reward: float = 0.5,
                 wrong_reward: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.neutral_threshold = neutral_threshold
        self.neutral_answer = neutral_answer.upper().strip()
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
                
            confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
            predicted_clean = predicted_answer.upper().strip()
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            if confidence_percentage < self.confidence_threshold:
                # Skip cases handled by neutral_when_uncertain (very low conf + wrong neutral)
                if (confidence_percentage < self.neutral_threshold and 
                    predicted_clean == self.neutral_answer and not is_correct):
                    rewards.append(0.0)  # Let neutral_when_uncertain handle this
                elif is_correct:
                    rewards.append(self.correct_reward)
                else:
                    rewards.append(self.wrong_reward)
            else:
                rewards.append(0.0)  # High confidence, no reward from this component
        
        return rewards


class NeutralWhenUncertainReward(RewardFunctionABC):
    """Bonus reward for choosing neutral when very uncertain."""
    __name__ = "neutral_when_uncertain"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 neutral_threshold: float = 50,
                 neutral_answer: str = "C",
                 bonus_reward: float = 0.4,
                 **kwargs):
        super().__init__(**kwargs)
        self.neutral_threshold = neutral_threshold
        self.neutral_answer = neutral_answer.upper().strip()
        self.bonus_reward = bonus_reward

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
                
            confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
            predicted_clean = predicted_answer.upper().strip()
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            # Bonus for choosing neutral when very uncertain (regardless of correctness)
            if (confidence_percentage < self.neutral_threshold and 
                predicted_clean == self.neutral_answer and
                not is_correct):  # Only when wrong, to avoid double-counting with accuracy rewards
                rewards.append(self.bonus_reward)
            else:
                rewards.append(0.0)
        
        return rewards


class HighConfidenceNeutralPenalty(RewardFunctionABC):
    """Penalizes choosing neutral when confidence is high (should take a stance)."""
    __name__ = "high_confidence_neutral_penalty"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 confidence_threshold: float = 70,
                 neutral_answer: str = "C",
                 penalty: float = -0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.neutral_answer = neutral_answer.upper().strip()
        self.penalty = penalty

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
                
            confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
            predicted_clean = predicted_answer.upper().strip()
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            # Penalty for high confidence + wrong neutral (should take a stance)
            if (confidence_percentage >= self.confidence_threshold and 
                predicted_clean == self.neutral_answer and
                not is_correct):
                rewards.append(self.penalty)
            else:
                rewards.append(0.0)
        
        return rewards


class ConfidenceProvisionReward(RewardFunctionABC):
    """Small penalty for not providing confidence scores."""
    __name__ = "confidence_provision"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 penalty: float = -0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.penalty = penalty

    def __call__(self, completions, answers, **kwargs):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not has_answer_tags:
                rewards.append(0.0)
                continue
                
            predicted_answer = _extract_answer(completion)
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            if not has_confidence_tags or _extract_confidence(completion) is None:
                # No confidence provided - apply penalty (but reduce if answer is correct)
                if is_correct:
                    rewards.append(self.penalty * 0.5)  # Smaller penalty for correct answers
                else:
                    rewards.append(self.penalty)
            else:
                rewards.append(0.0)  # Confidence provided, no penalty
        
        return rewards


class NeutralGroundTruthReward(RewardFunctionABC):
    """Rewards low confidence when ground truth is neutral and model predicts neutral correctly."""
    __name__ = "neutral_ground_truth"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 neutral_answer: str = "C",
                 confidence_threshold: float = 50,
                 reward: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.neutral_answer = neutral_answer.upper().strip()
        self.confidence_threshold = confidence_threshold
        self.reward = reward

    def __call__(self, completions, answers, **_):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
                
            confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
            predicted_clean = predicted_answer.upper().strip()
            ground_truth_clean = answer.upper().strip()
            
            # Reward when: ground truth is neutral, prediction is neutral, and confidence is low
            if (ground_truth_clean == self.neutral_answer and 
                predicted_clean == self.neutral_answer and
                confidence_percentage < self.confidence_threshold):
                rewards.append(self.reward)
            else:
                rewards.append(0.0)
        
        return rewards