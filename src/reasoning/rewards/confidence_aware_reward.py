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


class ConfidenceAwareAnswerReward(RewardFunctionABC):
    __name__ = "confidence_aware_answer"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 confidence_threshold: float = 70,
                 neutral_threshold: float = 50,
                 neutral_answer: str = "C",
                 high_confidence_correct_reward: float = 1.0,
                 high_confidence_wrong_penalty: float = -0.5,
                 high_confidence_neutral_penalty: float = -0.3,
                 low_confidence_correct_reward: float = 0.5,
                 low_confidence_wrong_reward: float = 0.2,
                 low_confidence_neutral_reward: float = 0.4,
                 no_confidence_penalty: float = 0.1,
                 **kwargs):
        """
        Confidence-aware answer reward function.
        
        Args:
            confidence_threshold: Threshold below which confidence is considered "low" (0-100 scale)
            neutral_threshold: Below this confidence, neutral predictions get bonus reward (0-100 scale)
            neutral_answer: The answer option that represents "neutral" (e.g. "C")
            high_confidence_correct_reward: Reward for correct answer with high confidence
            high_confidence_wrong_penalty: Penalty for wrong answer with high confidence
            high_confidence_neutral_penalty: Penalty for neutral answer with high confidence (should take a stance)
            low_confidence_correct_reward: Reward for correct answer with low confidence
            low_confidence_wrong_reward: Reward for wrong answer with low confidence (no penalty)
            low_confidence_neutral_reward: Bonus reward for neutral when very uncertain
            no_confidence_penalty: Small penalty when confidence is not provided
        """
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.neutral_threshold = neutral_threshold
        self.neutral_answer = neutral_answer.upper().strip()
        self.high_confidence_correct_reward = high_confidence_correct_reward
        self.high_confidence_wrong_penalty = high_confidence_wrong_penalty
        self.high_confidence_neutral_penalty = high_confidence_neutral_penalty
        self.low_confidence_correct_reward = low_confidence_correct_reward
        self.low_confidence_wrong_reward = low_confidence_wrong_reward
        self.low_confidence_neutral_reward = low_confidence_neutral_reward
        self.no_confidence_penalty = no_confidence_penalty

    def __call__(self, completions, answers, **_):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            # Check if required tags are present
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not has_answer_tags:
                rewards.append(0.0)
                continue
            
            # Extract answer and confidence
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion) if has_confidence_tags else None
            
            # Check if answer is correct
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            
            if confidence_score is None:
                # No confidence provided - small penalty
                if is_correct:
                    rewards.append(self.high_confidence_correct_reward - self.no_confidence_penalty)
                else:
                    rewards.append(self.no_confidence_penalty)
            else:
                # Convert to 0-100 scale for comparison with threshold
                confidence_percentage = confidence_score * 100 if confidence_score <= 1 else confidence_score
                
                if confidence_percentage >= self.confidence_threshold:
                    # High confidence
                    predicted_clean = predicted_answer.upper().strip()
                    if predicted_clean == self.neutral_answer and not is_correct:
                        # High confidence + wrong neutral = penalty (should take a stance)
                        rewards.append(self.high_confidence_neutral_penalty)
                    elif is_correct:
                        rewards.append(self.high_confidence_correct_reward)
                    else:
                        rewards.append(self.high_confidence_wrong_penalty)
                else:
                    # Low confidence
                    if is_correct:
                        rewards.append(self.low_confidence_correct_reward)
                    else:
                        # Check if it's a neutral prediction with very low confidence
                        predicted_clean = predicted_answer.upper().strip()
                        if (predicted_clean == self.neutral_answer and 
                            confidence_percentage < self.neutral_threshold):
                            # Reward neutral predictions when very uncertain (good epistemic behavior)
                            rewards.append(self.low_confidence_neutral_reward)
                        else:
                            # Low confidence wrong answer gets partial credit (no penalty)
                            rewards.append(self.low_confidence_wrong_reward)
        
        return rewards


class BrierCalibrationReward(RewardFunctionABC):
    __name__ = "brier_calibration"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 calibration_weight: float = 0.5,
                 sharpness_weight: float = 0.0,
                 neutral_answer: str = "C",
                 neutral_confidence_threshold: float = 50,
                 **kwargs):
        """
        Rewards well-calibrated confidence scores using Brier score with sharpness regularization.
        
        The Brier score measures the accuracy of probabilistic predictions:
        Brier = (prediction - outcome)²
        
        Sharpness regularization encourages confident predictions:
        total_reward = calibration_reward + sharpness_weight * abs(confidence - 0.5)
        
        Args:
            calibration_weight: Weight for calibration component vs accuracy component
            sharpness_weight: Weight for sharpness regularization (encourages confident predictions)
        """
        super().__init__(**kwargs)
        self.calibration_weight = calibration_weight
        self.sharpness_weight = sharpness_weight
        self.neutral_answer = neutral_answer.upper().strip()
        self.neutral_confidence_threshold = neutral_confidence_threshold

    def __call__(self, completions, answers, **_):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            # Check if required tags are present
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            # Extract answer and confidence
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
            
            # Check if answer is correct
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            ground_truth_clean = answer.upper().strip()
            predicted_clean = predicted_answer.upper().strip()
            
            # Base accuracy reward
            accuracy_reward = 1.0 if is_correct else 0.0
            
            # Special case: For correct neutral predictions, treat low confidence as well-calibrated
            if (is_correct and 
                ground_truth_clean == self.neutral_answer and
                predicted_clean == self.neutral_answer and
                confidence_score * 100 < self.neutral_confidence_threshold):
                # For correct neutrals with low confidence, treat as perfectly calibrated
                calibration_reward = 0.0  # Perfect Brier score (no error)
            else:
                # Brier score: measures calibration quality
                # Brier = (confidence - outcome)² where outcome = 1 if correct, 0 if wrong
                # Lower Brier score = better calibration, so we use -Brier as reward
                outcome = 1.0 if is_correct else 0.0
                brier_score = (confidence_score - outcome) ** 2
                calibration_reward = -brier_score  # Negative Brier score as reward
            
            # Sharpness regularization: encourages confident predictions
            sharpness_reward = self.sharpness_weight * abs(confidence_score - 0.5)
            
            # Combine accuracy, calibration, and sharpness
            total_reward = ((1 - self.calibration_weight) * accuracy_reward + 
                          self.calibration_weight * calibration_reward + 
                          sharpness_reward)
            rewards.append(total_reward)
        
        return rewards


class LogLikelihoodCalibrationReward(RewardFunctionABC):
    __name__ = "loglikelihood_calibration"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 calibration_weight: float = 0.5,
                 neutral_answer: str = "C",
                 neutral_confidence_threshold: float = 50,
                 **kwargs):
        """
        Rewards well-calibrated confidence scores using log-likelihood.
        
        Log-likelihood = correct_label * log(confidence) + (1 - correct_label) * log(1 - confidence)
        Higher log-likelihood = better calibration.
        
        Args:
            calibration_weight: Weight for calibration component vs accuracy component
            neutral_answer: The neutral answer option (default "C")
            neutral_confidence_threshold: Below this threshold, low confidence on correct neutrals is well-calibrated
        """
        super().__init__(**kwargs)
        self.calibration_weight = calibration_weight
        self.neutral_answer = neutral_answer.upper().strip()
        self.neutral_confidence_threshold = neutral_confidence_threshold

    def __call__(self, completions, answers, **_):
        import math
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            # Check if required tags are present
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            # Extract answer and confidence
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
            
            # Check if answer is correct
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            ground_truth_clean = answer.upper().strip()
            predicted_clean = predicted_answer.upper().strip()
            
            # Base accuracy reward
            accuracy_reward = 1.0 if is_correct else 0.0
            
            # Special case: For correct neutral predictions, treat low confidence as well-calibrated
            if (is_correct and 
                ground_truth_clean == self.neutral_answer and
                predicted_clean == self.neutral_answer and
                confidence_score * 100 < self.neutral_confidence_threshold):
                # For correct neutrals with low confidence, treat as optimally calibrated
                # Give perfect log-likelihood reward (0.0, equivalent to log(1.0))
                calibration_reward = 0.0
            else:
                # Log-likelihood calibration: correct_label * log(confidence) + (1 - correct_label) * log(1 - confidence)
                # Higher log-likelihood = better calibration
                outcome = 1.0 if is_correct else 0.0
                
                # Avoid log(0) by clamping confidence to [1e-8, 1-1e-8] range
                clamped_confidence = max(1e-8, min(1 - 1e-8, confidence_score))
                
                calibration_reward = (outcome * math.log(clamped_confidence) + 
                                    (1 - outcome) * math.log(1 - clamped_confidence))
            
            # Combine accuracy and calibration
            total_reward = (1 - self.calibration_weight) * accuracy_reward + self.calibration_weight * calibration_reward
            rewards.append(total_reward)
        
        return rewards


class FocalConfidenceReward(RewardFunctionABC):
    __name__ = "focal_confidence"
    tags = ["<answer>", "</answer>", "<confidence>", "</confidence>"]

    def __init__(self, 
                 epsilon: float = 0.1,
                 neutral_answer: str = "C",
                 neutral_confidence_threshold: float = 50,
                 **kwargs):
        """
        Focal loss style confidence reward that weights confident predictions more heavily.
        
        Uses: weight = abs(confidence - 0.5) + epsilon
        Then: weighted_reward = weight * brier_score
        
        Args:
            epsilon: Small value to avoid zero weight (default 0.1)
            neutral_answer: The neutral answer option (default "C")
            neutral_confidence_threshold: Below this threshold, low confidence on correct neutrals is well-calibrated
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.neutral_answer = neutral_answer.upper().strip()
        self.neutral_confidence_threshold = neutral_confidence_threshold

    def __call__(self, completions, answers, **_):
        rewards = []
        completions = [completion[0]["content"] for completion in completions]
        
        for completion, answer in zip(completions, answers):
            # Check if required tags are present
            has_answer_tags = "<answer>" in completion and "</answer>" in completion
            has_confidence_tags = "<confidence>" in completion and "</confidence>" in completion
            
            if not (has_answer_tags and has_confidence_tags):
                rewards.append(0.0)
                continue
            
            # Extract answer and confidence
            predicted_answer = _extract_answer(completion)
            confidence_score = _extract_confidence(completion)
            
            if confidence_score is None:
                rewards.append(-1.0)  # Strong penalty for malformed confidence
                continue
            
            # Check if answer is correct
            is_correct = predicted_answer.lower().strip() == answer.lower().strip()
            ground_truth_clean = answer.upper().strip()
            predicted_clean = predicted_answer.upper().strip()
            
            # Special case: For correct neutral predictions, treat low confidence as well-calibrated
            if (is_correct and 
                ground_truth_clean == self.neutral_answer and
                predicted_clean == self.neutral_answer and
                confidence_score * 100 < self.neutral_confidence_threshold):
                # For correct neutrals with low confidence, give maximum positive reward
                # Treat as perfect calibration (zero Brier score) with high weight
                weighted_reward = 0.0  # Perfect calibration reward
            else:
                # Focal loss style weighting: higher weight for confident predictions
                # abs(confidence - 0.5) gives 0 weight at 50% confidence, max weight at 0% or 100%
                weight = abs(confidence_score - 0.5) + self.epsilon
                
                # Brier score
                outcome = 1.0 if is_correct else 0.0
                brier_score = (confidence_score - outcome) ** 2
                
                # Weighted reward (negative Brier score, so lower is better)
                weighted_reward = -weight * brier_score
            
            rewards.append(weighted_reward)
        
        return rewards
