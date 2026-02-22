import logging
import os
import random

from datasets import load_from_disk
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from reasoning.common import format_sft_response, test_model
from reasoning.rewards import REWARD_FUNC_DICT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


def _validate_reward_functions(reward_funcs):
    """Validate that incompatible reward functions are not used together.

    Args:
        reward_funcs: List of reward function configurations to validate.

    Raises:
        ValueError: If incompatible format reward functions are used together.
    """
    format_rewards = []
    for rf in reward_funcs:
        if rf.name in ["hard_format", "soft_format", "xml_format", "three_step_format", "modular_step_format", "multistep_format", "multistep_xml_format", "two_stage_format"]:
            format_rewards.append(rf.name)
    
    # Check for conflicts
    modular_used = any(name in format_rewards for name in ["three_step_format", "modular_step_format"])
    traditional_used = any(name in format_rewards for name in ["hard_format", "soft_format", "xml_format"])
    other_multistep_used = any(name in format_rewards for name in ["multistep_format", "multistep_xml_format"])
    two_stage_used = "two_stage_format" in format_rewards
    
    if modular_used and traditional_used:
        raise ValueError("Cannot use modular step format rewards with traditional format rewards (hard_format, soft_format, xml_format)")
    
    if modular_used and other_multistep_used:
        raise ValueError("Cannot use modular step format rewards with other multistep format rewards (multistep_format, multistep_xml_format)")
    
    if two_stage_used and (traditional_used or modular_used or other_multistep_used):
        raise ValueError("Cannot use two_stage_format reward with other format rewards")
    
    # Ensure only one modular format reward is used
    modular_count = sum(1 for name in format_rewards if name in ["three_step_format", "modular_step_format"])
    if modular_count > 1:
        raise ValueError("Cannot use multiple modular step format rewards together")


def main():
    """Main training function for GRPO-based reasoning methods.

    Loads configuration from params.yaml and performs optional SFT followed by
    GRPO training with configurable reward functions. Supports various reasoning
    formats and evaluates the model on a test set.

    Configuration structure (params.yaml):
        reasoning:
            random_seed: Random seed for reproducibility
            model: Model configuration (name, args, peft_args)
            dataset: Dataset paths, splits, and column mappings
            trainer: GRPO training args and reward functions
            sft: Optional SFT phase configuration

    Raises:
        ValueError: If incompatible reward functions are configured.
    """
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["reasoning"]
    set_seed(cfg.random_seed)

    # Check if SFT step should be performed
    sft_enabled = "sft_column" in cfg.dataset

    # Validate reward function compatibility
    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "reward_funcs"):
        _validate_reward_functions(cfg.trainer.reward_funcs)
    else:
        logger.error("Missing trainer.reward_funcs in configuration")
        raise ValueError("Configuration must include trainer.reward_funcs")

    reward_funcs = []
    for reward_func in cfg.trainer.reward_funcs:
        logger.info(f"Loading reward function: {reward_func.name}")
        reward_func_cls = REWARD_FUNC_DICT[reward_func.name]
        if "args" in reward_func:
            reward_funcs.append(reward_func_cls(**reward_func.args))
        else:
            reward_funcs.append(reward_func_cls())

    logger.info(f"Loading model: {cfg.model.name}")
    logger.debug(f"Model args: {cfg.model.model_args}")

    if cfg.model.use_unsloth:
        logger.info("Using unsloth for model loading")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.name, **cfg.model.model_args
        )

        if "is_lora" not in cfg.model or not cfg.model.is_lora:
            logger.info("Applying LoRA adapters to model")
            logger.debug(f"LoRA args: {cfg.model.peft_args}")
            model = FastLanguageModel.get_peft_model(model, **cfg.model.peft_args)

    else:
        logger.info("Using transformers for model loading")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, **OmegaConf.to_container(cfg.model.model_args)
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        if "is_lora" not in cfg.model or not cfg.model.is_lora:
            lora_cfg = LoraConfig(
                **OmegaConf.to_container(cfg.model.peft_args, resolve=True)
            )
            model = get_peft_model(model, lora_cfg)

    # Load main dataset for GRPO training and testing
    logger.info(f"Loading main dataset: {cfg.dataset.path}")
    train_set = load_from_disk(
        cfg.dataset.path,
    )[cfg.dataset.get("train_split", "train")]
    test_set = load_from_disk(
        cfg.dataset.path,
    )[cfg.dataset.get("test_split", "test")]
    
    # Load separate SFT dataset if specified
    sft_dataset = None
    if sft_enabled and hasattr(cfg.dataset, 'sft_dataset_path') and cfg.dataset.sft_dataset_path:
        logger.info(f"Loading separate SFT dataset: {cfg.dataset.sft_dataset_path}")
        sft_dataset = load_from_disk(cfg.dataset.sft_dataset_path)
        # For SFT datasets that don't have splits, use the entire dataset
        if hasattr(sft_dataset, 'column_names'):
            # It's a single dataset without splits
            sft_dataset = sft_dataset
        else:
            # It has splits, use train split
            sft_dataset = sft_dataset[cfg.dataset.get("sft_train_split", "train")]
    
    # Apply subset sampling if configured
    if hasattr(cfg.dataset, 'subset_fraction') and cfg.dataset.subset_fraction is not None:
        if not (0.0 < cfg.dataset.subset_fraction <= 1.0):
            raise ValueError(f"subset_fraction must be between 0.0 and 1.0, got {cfg.dataset.subset_fraction}")
        
        original_size = len(train_set)
        subset_size = int(original_size * cfg.dataset.subset_fraction)
        
        logger.info(f"Using random subset of training data: {subset_size}/{original_size} samples ({cfg.dataset.subset_fraction:.1%})")
        
        # Create indices and shuffle with seed for reproducibility
        import random
        indices = list(range(original_size))
        random.seed(cfg.random_seed)
        random.shuffle(indices)
        
        # Select subset
        subset_indices = indices[:subset_size]
        train_set = train_set.select(subset_indices)
        
        logger.info(f"Training set reduced from {original_size} to {len(train_set)} samples")
    if "reasoning_column" in cfg.dataset:
        reasoning_lambda = (  # noqa: E731
            lambda x: {
                "prompt": [
                    {"role": "system", "content": cfg.dataset.system_prompt},
                    {"role": "user", "content": x[cfg.dataset.question_column]},
                    {"role": "assistant", "content": x[cfg.dataset.reasoning_column]},
                ],
                "answers": x[cfg.dataset.answer_column],
                "answer_only": True,
            }
            if x[cfg.dataset.reasoning_column] is not None
            else {
                "prompt": [
                    {"role": "system", "content": cfg.dataset.system_prompt},
                    {"role": "user", "content": x[cfg.dataset.question_column]},
                ],
                "answers": x[cfg.dataset.answer_column],
                "answer_only": False,
            }
        )
        train_set = train_set.map(reasoning_lambda)
        test_set = test_set.map(reasoning_lambda)

    else:
        convo_lambda = (  # noqa: E731
            lambda x: {
                "prompt": [
                    {"role": "system", "content": cfg.dataset.system_prompt},
                    {"role": "user", "content": x[cfg.dataset.question_column]},
                ],
                "answers": x[cfg.dataset.answer_column],
                "answer_only": False,
            }
        )
        train_set = train_set.map(convo_lambda)
        test_set = test_set.map(convo_lambda)

    # Optional SFT step
    if sft_enabled:
        logger.info("Starting SFT phase")
        sft_config = SFTConfig(
            **OmegaConf.to_container(cfg.sft.training_args, resolve=True)
        )

        if "DVC_EXP_NAME" in os.environ and "SLURM_JOB_ID" in os.environ:
            sft_config.run_name = f"sft_{cfg.dataset.train_split}_{os.environ['DVC_EXP_NAME']}_{os.environ['SLURM_JOB_ID']}"
        else:
            sft_config.run_name = f"sft_{cfg.dataset.train_split}"

        # Use separate SFT dataset if provided, otherwise use main dataset
        if sft_dataset is not None:
            logger.info("Using separate SFT dataset for supervised fine-tuning")
            sft_train_set = sft_dataset.map(
                lambda x: {
                    "text": tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": cfg.dataset.system_prompt},
                            {"role": "user", "content": x[cfg.dataset.question_column]},
                            {
                                "role": "assistant",
                                "content": format_sft_response(x, cfg),
                            },
                        ],
                        tokenize=False,
                    )
                }
            ).select_columns(["text"])
        else:
            logger.info("Using main dataset for supervised fine-tuning")
            # Prepare SFT dataset using the sft_column, filtering out None values
            sft_train_set = train_set.filter(
                lambda x: x[cfg.dataset.sft_column] != ""
                and x[cfg.dataset.sft_column] is not None
            )
            sft_train_set = sft_train_set.map(
                lambda x: {
                    "text": tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": cfg.dataset.system_prompt},
                            {"role": "user", "content": x[cfg.dataset.question_column]},
                            {
                                "role": "assistant",
                                "content": format_sft_response(x, cfg),
                            },
                        ],
                        tokenize=False,
                    )
                }
            ).select_columns(["text"])

        sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_config,
            train_dataset=sft_train_set,
        )

        sft_trainer.train()
        model = sft_trainer.model
        logger.info("SFT phase completed")

        # Save SFT checkpoint
        model.save_pretrained("sft_checkpoint")
        tokenizer.save_pretrained("sft_checkpoint")

    # Check if we should skip GRPO training and only do testing
    sft_only = cfg.trainer.get("sft_only", False)

    if sft_only:
        if not sft_enabled:
            logger.warning("SFT-only mode enabled but no sft_column found in dataset config. Running test evaluation on base model.")
        else:
            logger.info("SFT-only mode enabled. Skipping GRPO training and running test evaluation.")
        # Test the model after SFT training (or base model if SFT wasn't run)
        test_model(model, tokenizer, test_set, cfg)
        return

    logger.info("Initializing GRPO trainer")
    logger.debug(f"Training args: {cfg.trainer.training_args}")
    training_args = GRPOConfig(
        **OmegaConf.to_container(cfg.trainer.training_args, resolve=True)
    )
    # training_args = GRPOConfig(**cfg.trainer.training_args)

    if "DVC_EXP_NAME" in os.environ and "SLURM_JOB_ID" in os.environ:
        training_args.run_name = f"{cfg.dataset.train_split}_{os.environ['DVC_EXP_NAME']}_{os.environ['SLURM_JOB_ID']}"
    else:
        training_args.run_name = f"{cfg.dataset.train_split}"

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
    )
    logger.info("Starting training")
    trainer.train()
    logger.info("Training finished")
    model.save_pretrained("grpo_saved_lora")
    tokenizer.save_pretrained("grpo_saved_lora")

    # Test the model after training
    test_model(trainer.model, tokenizer, test_set, cfg)


if __name__ == "__main__":
    main()
