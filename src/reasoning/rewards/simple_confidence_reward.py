import re
from reasoning.rewards.reward_function_abc import RewardFunctionABC


def _extract_answer(completion: str) -> str:
    """Extract answer from completion."""
    answer_text = completion.split("<answer>")[-1].split("</answer>")[0].strip()
    # Extract just the letter (A, B, C) from formats like "A) Yes" or "A"
    if answer_text and answer_text[0].upper() in ['A', 'B', 'C']:
        return answer_text[0].upper()
    return answer_text


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


class SimpleConfidenceReward(RewardFunctionABC):
    """Simple confidence-based reward: confidence for correct, -confidence for wrong, 0 for neutral."""
    __name__ = "simple_confidence"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 neutral_answer: str = "C",
                 no_confidence_penalty: float = -0.1,
                 allow_neutral: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.neutral_answer = neutral_answer.upper().strip()
        self.no_confidence_penalty = no_confidence_penalty
        self.allow_neutral = allow_neutral

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
            predicted_clean = predicted_answer.upper().strip()
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            # Handle neutral answers based on allow_neutral setting
            if predicted_clean == self.neutral_answer:
                if self.allow_neutral:
                    rewards.append(0.0)  # Neutral gets 0 reward (abstention allowed)
                    continue
                else:
                    # Treat neutral as wrong when abstention is not allowed
                    if not has_confidence_tags:
                        rewards.append(self.no_confidence_penalty)
                        continue
                    
                    confidence_score = _extract_confidence(completion)
                    if confidence_score is None:
                        rewards.append(self.no_confidence_penalty)
                        continue
                    
                    # Penalize neutral when abstention not allowed
                    rewards.append(-confidence_score)
                    continue
            
            if not has_confidence_tags:
                # No confidence provided - small penalty
                rewards.append(self.no_confidence_penalty)
                continue
            
            confidence_score = _extract_confidence(completion)
            if confidence_score is None:
                rewards.append(self.no_confidence_penalty)
                continue
            
            # Simple reward structure: confidence if correct, -confidence if wrong
            if is_correct:
                rewards.append(confidence_score)
            else:
                rewards.append(-confidence_score)
        
        return rewards