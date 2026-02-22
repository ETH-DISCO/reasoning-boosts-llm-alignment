from reasoning.rewards.answer_in_completion_reward import (
    AnswerDistanceReward,
    AnswerInCompletionReward,
    RightAnswerInCompletionReward,
)
from reasoning.rewards.confidence_aware_reward import (
    ConfidenceAwareAnswerReward,
    BrierCalibrationReward,
    LogLikelihoodCalibrationReward,
    FocalConfidenceReward,
)
from reasoning.rewards.confidence_components import (
    HighConfidenceAccuracyReward,
    LowConfidenceHumilityReward,
    NeutralWhenUncertainReward,
    HighConfidenceNeutralPenalty,
    ConfidenceProvisionReward,
    NeutralGroundTruthReward,
)
from reasoning.rewards.simple_confidence_reward import SimpleConfidenceReward
from reasoning.rewards.format_reward import (
    HardFormatReward,
    SoftFormatReward,
    XMLFormatReward,
    LanguageReward,
    MultiStepFormatReward,
    XMLMultiStepFormatReward,
    ThreeStepFormatReward,
    ModularStepFormatReward,
    TwoStageFormatReward
)
from reasoning.rewards.length_reward import LengthReward
from reasoning.rewards.reward_function_abc import RewardFunctionABC
from reasoning.rewards.entailment_reward import EntailmentReward, ProfileUsedReward

REWARD_FUNC_DICT = {
    "answer_in_completion": AnswerInCompletionReward,
    "right_answer_in_completion": RightAnswerInCompletionReward,
    "confidence_aware_answer": ConfidenceAwareAnswerReward,
    "brier_calibration": BrierCalibrationReward,
    "loglikelihood_calibration": LogLikelihoodCalibrationReward,
    "focal_confidence": FocalConfidenceReward,
    "high_confidence_accuracy": HighConfidenceAccuracyReward,
    "low_confidence_humility": LowConfidenceHumilityReward,
    "neutral_when_uncertain": NeutralWhenUncertainReward,
    "high_confidence_neutral_penalty": HighConfidenceNeutralPenalty,
    "confidence_provision": ConfidenceProvisionReward,
    "neutral_ground_truth": NeutralGroundTruthReward,
    "simple_confidence": SimpleConfidenceReward,
    "length": LengthReward,
    "hard_format": HardFormatReward,
    "soft_format": SoftFormatReward,
    "xml_format": XMLFormatReward,
    "answer_distance": AnswerDistanceReward,
    "entailment": EntailmentReward,
    "profile_used_reward": ProfileUsedReward,
    "language": LanguageReward,
    "multistep_format": MultiStepFormatReward,
    "multistep_xml_format": XMLMultiStepFormatReward,
    "three_step_format": ThreeStepFormatReward,
    "modular_step_format": ModularStepFormatReward,
    "two_stage_format": TwoStageFormatReward
}

__all__ = [
    "RewardFunctionABC",
    "AnswerDistanceReward",
    "AnswerInCompletionReward",
    "RightAnswerInCompletionReward",
    "LengthReward",
    "HardFormatReward",
    "SoftFormatReward",
    "XMLFormatReward",
    "EntailmentReward",
    "REWARD_FUNC_DICT",
]
