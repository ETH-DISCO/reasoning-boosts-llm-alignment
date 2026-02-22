import argparse
import logging
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from reasoning.common import extract_answer_probabilities, format_sft_response, test_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_wandb(method_name: str, cfg: Dict[str, Any], overrides: Dict[str, Any] = None):
    """Initialize wandb logging for baseline experiments.

    Args:
        method_name: Name of the baseline method (e.g., 'sft', 'dpo', 'in_context_learning').
        cfg: Configuration dictionary with experiment settings.
        overrides: Optional configuration overrides to include in wandb config.
    """
    project_name = cfg.get("wandb_project", "political-agents-baselines")

    # Create run name
    dataset_name = cfg.dataset.path.split("/")[-1] if cfg.dataset.path else "unknown"
    train_split = cfg.dataset.get("train_split", "train")
    test_split = cfg.dataset.get("test_split", "test")
    run_name = f"{method_name}_{dataset_name}_{train_split}_{test_split}"

    # Add configuration details to run name for ICL
    if method_name == "in_context_learning" and "baseline" in cfg:
        num_repeats = cfg.baseline.get("num_repeats", 8)
        run_name += f"_rep{num_repeats}"

    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "method": method_name,
            "dataset_path": cfg.dataset.path,
            "train_split": train_split,
            "test_split": test_split,
            "model_name": cfg.model.name if "model" in cfg else None,
            "random_seed": cfg.get("random_seed", 42),
            **cfg.get("baseline", {}),
            **(overrides or {}),
        },
        tags=[method_name, dataset_name],
    )

    logger.info(f"Initialized wandb logging for {run_name} in project {project_name}")


def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load and return the configuration from the config file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary. Extracts 'reasoning' section if present,
        otherwise returns the full config.
    """
    cfg = OmegaConf.load(config_path)
    # Handle both structured configs (with 'reasoning' key) and flat configs
    if "reasoning" in cfg:
        return cfg["reasoning"]
    else:
        return cfg


def load_datasets(cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Load train and test datasets from disk based on configuration.

    Args:
        cfg: Configuration dictionary with dataset.path, dataset.train_split,
             and dataset.test_split fields.

    Returns:
        Tuple of (train_set, test_set) as HuggingFace Dataset objects.
    """
    logger.info(f"Loading dataset: {cfg.dataset.path}")

    train_set = load_from_disk(cfg.dataset.path)[
        cfg.dataset.get("train_split", "train")
    ]
    test_set = load_from_disk(cfg.dataset.path)[cfg.dataset.get("test_split", "test")]

    logger.info(
        f"Loaded {len(train_set)} training samples and {len(test_set)} test samples"
    )
    return train_set, test_set


def filter_duplicate_questions(train_set: Dataset, cfg: Dict[str, Any]) -> Dataset:
    """Filter out duplicate questions from training set based on 'question' column.

    Args:
        train_set: Training dataset to filter.
        cfg: Configuration dictionary (unused but kept for API consistency).

    Returns:
        Filtered dataset with unique questions only.
    """
    if "question" not in train_set.column_names:
        logger.warning("No 'question' column found for duplicate filtering, skipping")
        return train_set

    original_size = len(train_set)
    seen_questions = set()
    unique_indices = []

    for i, sample in enumerate(train_set):
        question = sample["question"]
        if question not in seen_questions:
            seen_questions.add(question)
            unique_indices.append(i)

    if len(unique_indices) < original_size:
        train_set = train_set.select(unique_indices)
        logger.info(
            f"Filtered duplicate questions: {original_size} -> {len(train_set)} samples ({original_size - len(train_set)} duplicates removed)"
        )
    else:
        logger.info(
            f"No duplicate questions found in training set ({original_size} samples)"
        )

    return train_set


def apply_subset_sampling(train_set: Dataset, cfg: Dict[str, Any]) -> Dataset:
    """Apply subset sampling to the training set if configured.

    Args:
        train_set: Training dataset to sample from.
        cfg: Configuration with optional dataset.subset_fraction field (0.0-1.0).

    Returns:
        Sampled dataset if subset_fraction is configured, otherwise original dataset.

    Raises:
        ValueError: If subset_fraction is not between 0.0 and 1.0.
    """
    if not (
        hasattr(cfg.dataset, "subset_fraction")
        and cfg.dataset.subset_fraction is not None
    ):
        return train_set

    if not (0.0 < cfg.dataset.subset_fraction <= 1.0):
        raise ValueError(
            f"subset_fraction must be between 0.0 and 1.0, got {cfg.dataset.subset_fraction}"
        )

    original_size = len(train_set)
    subset_size = int(original_size * cfg.dataset.subset_fraction)

    logger.info(
        f"Using random subset of training data: {subset_size}/{original_size} samples ({cfg.dataset.subset_fraction:.1%})"
    )

    # Create indices and shuffle with seed for reproducibility
    indices = list(range(original_size))
    random.seed(cfg.random_seed)
    random.shuffle(indices)

    # Select subset
    subset_indices = indices[:subset_size]
    train_set = train_set.select(subset_indices)

    logger.info(
        f"Training set reduced from {original_size} to {len(train_set)} samples"
    )
    return train_set


def format_dataset_with_prompts(
    train_set: Dataset, test_set: Dataset, cfg: Dict[str, Any]
) -> Tuple[Dataset, Dataset]:
    """Format datasets with conversation prompts based on configuration.

    Creates chat-style prompts with system message, user question, and optional
    assistant reasoning. Adds 'prompt', 'answers', and 'answer_only' fields.

    Args:
        train_set: Training dataset to format.
        test_set: Test dataset to format.
        cfg: Configuration with dataset.system_prompt, dataset.question_column,
             and optional dataset.reasoning_column.

    Returns:
        Tuple of (formatted_train_set, formatted_test_set).
    """
    if "reasoning_column" in cfg.dataset:
        # Use reasoning-based format
        reasoning_lambda = (
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
        # Use conversation-based format
        convo_lambda = lambda x: {
            "prompt": [
                {"role": "system", "content": cfg.dataset.system_prompt},
                {"role": "user", "content": x[cfg.dataset.question_column]},
            ],
            "answers": x[cfg.dataset.answer_column],
            "answer_only": False,
        }
        train_set = train_set.map(convo_lambda)
        test_set = test_set.map(convo_lambda)

    logger.info("Dataset formatted with conversation prompts")
    return train_set, test_set


def prepare_data(
    config_path: str = "params.yaml", overrides: Dict[str, Any] = None
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """
    Main data preparation function that loads and prepares datasets for baseline experiments.

    Args:
        config_path: Path to configuration file
        overrides: Dictionary of configuration overrides

    Returns:
        Tuple of (train_set, test_set, config)
    """
    # Load configuration
    cfg = load_config(config_path)

    # Apply overrides if provided
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        logger.info(f"Applied configuration overrides: {overrides}")

    # Set random seed
    set_seed(cfg.random_seed)

    # Load datasets
    train_set, test_set = load_datasets(cfg)

    # Filter duplicate questions from training set
    train_set = filter_duplicate_questions(train_set, cfg)
    test_set = filter_duplicate_questions(test_set, cfg)

    # Apply subset sampling if configured
    train_set = apply_subset_sampling(train_set, cfg)

    # Format datasets with prompts
    train_set, test_set = format_dataset_with_prompts(train_set, test_set, cfg)

    logger.info("Data preparation completed")
    return train_set, test_set, cfg


def run_baseline_method(
    method_name: str,
    train_set: Dataset,
    test_set: Dataset,
    cfg: Dict[str, Any],
    overrides: Dict[str, Any] = None,
):
    """Run a baseline method with wandb logging.

    Available methods:
    - 'in_context_learning': Few-shot prompting with training examples
    - 'sft': Supervised fine-tuning on reasoning traces
    - 'dpo': Direct Preference Optimization using chosen/rejected pairs
    - 'sft_dpo': SFT followed by DPO

    Args:
        method_name: Name of the baseline method to run.
        train_set: Prepared training dataset.
        test_set: Prepared test dataset.
        cfg: Configuration dictionary.
        overrides: Optional configuration overrides for wandb logging.

    Raises:
        ValueError: If method_name is not recognized.
    """
    logger.info(f"Running baseline method: {method_name}")

    # Initialize wandb logging
    init_wandb(method_name, cfg, overrides)

    try:
        if method_name == "in_context_learning":
            run_in_context_learning_baseline(train_set, test_set, cfg)
        elif method_name == "dpo":
            run_dpo_baseline(train_set, test_set, cfg)
        elif method_name == "sft":
            run_sft_baseline(train_set, test_set, cfg)
        elif method_name == "sft_dpo":
            run_sft_dpo_baseline(train_set, test_set, cfg)
        else:
            raise ValueError(f"Unknown baseline method: {method_name}")
    finally:
        # Finish wandb run
        wandb.finish()


def load_model_and_tokenizer(cfg: Dict[str, Any]):
    """Load model and tokenizer for baseline methods.

    Supports both unsloth and standard transformers loading with optional
    quantization (4-bit or 8-bit) and LoRA configuration.

    Args:
        cfg: Configuration with model.name, model.model_args, model.peft_args,
             and optional model.use_unsloth and model.is_lora fields.

    Returns:
        Tuple of (model, tokenizer). Model may have LoRA adapters applied.
    """
    model_name = cfg.model.name
    logger.info(f"Loading model and tokenizer: {model_name}")

    if cfg.model.get("use_unsloth", False):
        from unsloth import FastLanguageModel

        # For unsloth, use the original args since it handles them internally
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.name, **cfg.model.model_args
        )

        if "is_lora" not in cfg.model or not cfg.model.is_lora:
            model = FastLanguageModel.get_peft_model(model, **cfg.model.peft_args)
    else:
        # Prepare model args for standard transformers
        model_args = OmegaConf.to_container(cfg.model.model_args)

        # Handle quantization config properly
        quantization_config = None
        if model_args.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # Remove deprecated arguments
            model_args.pop("load_in_4bit", None)
        elif model_args.get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            # Remove deprecated arguments
            model_args.pop("load_in_8bit", None)

        # Remove arguments that don't belong to AutoModelForCausalLM.from_pretrained
        model_args.pop("max_seq_length", None)
        model_args.pop("fast_inference", None)
        model_args.pop("max_lora_rank", None)
        model_args.pop("gpu_memory_utilization", None)

        # Add quantization config if needed
        if quantization_config:
            model_args["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "is_lora" not in cfg.model or not cfg.model.is_lora:
            lora_cfg = LoraConfig(
                **OmegaConf.to_container(cfg.model.peft_args, resolve=True)
            )
            model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


MAX_CONTEXT_EXAMPLES = 80


def create_in_context_examples(
    train_set: Dataset,
    test_sample: Dict[str, Any],
) -> List[str]:
    """Create in-context examples from training set, filtered by topic.

    Args:
        train_set: Training dataset to sample examples from.
        test_sample: Test sample used for topic filtering.

    Returns:
        List of formatted "Question: ...\nAnswer: ..." strings.
    """
    examples = []

    if "topic" not in train_set.column_names:
        raise ValueError("Dataset must have a 'topic' column for topic filtering")

    test_topic = test_sample.get("topic")
    if not test_topic:
        raise ValueError("Test sample must have a 'topic' field for topic filtering")

    filtered_train = train_set.filter(lambda x: x.get("topic") == test_topic)
    logger.info(
        f"Filtered training set to {len(filtered_train)} examples with topic '{test_topic}'"
    )

    # Limit number of examples
    if len(filtered_train) > MAX_CONTEXT_EXAMPLES:
        indices = random.sample(range(len(filtered_train)), MAX_CONTEXT_EXAMPLES)
        filtered_train = filtered_train.select(indices)
        logger.info(f"Sampled {MAX_CONTEXT_EXAMPLES} examples from training set")

    # Create question-answer pairs
    for sample in filtered_train:
        question = sample["prompt"][-1]["content"]  # Get user question from prompt
        answer = sample["answers"]
        examples.append(f"Question: {question}\nAnswer: {answer}")

    return examples


def run_in_context_learning_baseline(
    train_set: Dataset, test_set: Dataset, cfg: Dict[str, Any]
):
    """Run in-context learning baseline.

    Creates few-shot prompts by prepending training examples to test questions,
    then generates and evaluates answers.

    Configuration options (in cfg.baseline):
    - num_repeats: Number of generations per test sample (default 8)

    Args:
        train_set: Training dataset for in-context examples.
        test_set: Test dataset to evaluate.
        cfg: Configuration dictionary.

    Outputs:
        - in_context_learning_results_topic_{bool}.json: Detailed results
        - Logs metrics and results table to wandb
    """
    logger.info("Running in-context learning baseline")

    # Configuration options
    num_repeats = cfg.get("baseline", {}).get("num_repeats", 8)

    logger.info(f"Number of repeats: {num_repeats}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    results = []
    correct_answers = 0
    unparseable_answers = 0
    total_answers = 0

    for i, test_sample in enumerate(test_set):
        # Create in-context examples
        context_examples = create_in_context_examples(train_set, test_sample)

        # Build the full prompt
        context_text = "\n\n".join(context_examples)
        test_question = test_sample["prompt"][-1]["content"]  # Get user question

        # Create the full prompt with context and test question
        full_prompt = f"{context_text}\n\nQuestion: {test_question}\nAnswer:"

        # Run multiple repeats
        for repeat in range(num_repeats):
            try:
                # Generate response
                inputs = tokenizer(
                    full_prompt, return_tensors="pt", truncation=True, max_length=4000
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=10,  # Short answer expected
                        temperature=0.1 if num_repeats == 1 else 0.9,
                        do_sample=num_repeats > 1,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Decode response
                response = tokenizer.decode(
                    outputs.sequences[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()

                # Extract answer probabilities
                answer_probs = extract_answer_probabilities(
                    model, tokenizer, full_prompt + " "
                )

                # Parse the predicted answer
                predicted_answer = (
                    response.split()[0] if response else ""
                )  # Take first word
                ground_truth = test_sample["answers"]

                # Check if answer is correct
                is_correct = (
                    predicted_answer.upper().strip() == ground_truth.upper().strip()
                )
                parsed_successfully = predicted_answer.upper() in ["A", "B", "C"]

                if is_correct:
                    correct_answers += 1
                if not parsed_successfully:
                    unparseable_answers += 1

                total_answers += 1

                # Store result
                result = {
                    "sample_id": i,
                    "repeat_id": repeat,
                    "question": test_question,
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth,
                    "full_response": response,
                    "full_prompt": full_prompt,
                    "context_examples_count": len(context_examples),
                    "parsed_successfully": parsed_successfully,
                    "is_correct": is_correct,
                    "answer_prob_A": answer_probs["A"],
                    "answer_prob_B": answer_probs["B"],
                    "answer_prob_C": answer_probs["C"],
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing sample {i}, repeat {repeat}: {e}")
                unparseable_answers += 1
                total_answers += 1
                continue

        if i % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_set)} test samples")

    # Calculate metrics
    accuracy = correct_answers / total_answers if total_answers > 0 else 0
    parsing_success_rate = (
        (total_answers - unparseable_answers) / total_answers
        if total_answers > 0
        else 0
    )

    logger.info(
        f"In-context learning results: {correct_answers}/{total_answers} correct ({accuracy * 100:.1f}%), {unparseable_answers} unparseable"
    )
    logger.info(f"Parsing success rate: {parsing_success_rate * 100:.1f}%")

    # Log metrics to wandb
    metrics = {
        "accuracy": accuracy,
        "parsing_success_rate": parsing_success_rate,
        "correct_answers": correct_answers,
        "total_answers": total_answers,
        "unparseable_answers": unparseable_answers,
        "num_repeats": num_repeats,
    }

    wandb.log(metrics)

    # Save results locally
    output_file = "in_context_learning_results.json"
    with open(output_file, "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)

    logger.info(f"Saved results to {output_file}")

    # Log results table to wandb
    results_table = wandb.Table(
        columns=[
            "sample_id",
            "repeat_id",
            "question",
            "predicted_answer",
            "ground_truth",
            "is_correct",
            "parsed_successfully",
            "prob_A",
            "prob_B",
            "prob_C",
        ]
    )
    for result in results[:100]:  # Log first 100 results to avoid huge tables
        results_table.add_data(
            result["sample_id"],
            result["repeat_id"],
            result["question"][:100] + "..."
            if len(result["question"]) > 100
            else result["question"],
            result["predicted_answer"],
            result["ground_truth"],
            result["is_correct"],
            result["parsed_successfully"],
            result.get("answer_prob_A", 0),
            result.get("answer_prob_B", 0),
            result.get("answer_prob_C", 0),
        )

    wandb.log({"results_sample": results_table})


def run_dpo_baseline(train_set: Dataset, test_set: Dataset, cfg: Dict[str, Any]):
    """Run DPO (Direct Preference Optimization) baseline.

    Trains a model using DPO with chosen (sft_column) and rejected (dpo_comment)
    response pairs, then evaluates on test set.

    Args:
        train_set: Training dataset with sft_column and dpo_comment fields.
        test_set: Test dataset to evaluate.
        cfg: Configuration with dataset.sft_column, dataset.dpo_comment.

    Raises:
        ValueError: If required columns (sft_column, dpo_comment) are missing.

    Outputs:
        - dpo_baseline_checkpoint/: Saved model
        - dpo_baseline_test_results.json: Test evaluation results
        - Logs metrics to wandb
    """
    logger.info("Running DPO baseline")

    # Check if required columns exist
    if "dpo_comment" not in train_set.column_names:
        raise ValueError(
            "DPO baseline requires 'dpo_comment' column for rejected generations"
        )

    if cfg.dataset.get("sft_column") is None:
        raise ValueError(
            "DPO baseline requires 'sft_column' to be configured for chosen generations"
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Prepare DPO dataset - filter out samples without both chosen and rejected responses
    dpo_train_set = train_set.filter(
        lambda x: (
            x[cfg.dataset.sft_column] != ""
            and x[cfg.dataset.sft_column] is not None
            and x["dpo_comment"] != ""
            and x["dpo_comment"] is not None
        )
    )

    logger.info(
        f"Filtered DPO training set: {len(dpo_train_set)} samples (from {len(train_set)} original)"
    )

    # Format DPO dataset with chosen/rejected pairs
    def format_dpo_sample(sample):
        prompt = [
            {"role": "system", "content": cfg.dataset.system_prompt},
            {"role": "user", "content": sample[cfg.dataset.question_column]},
        ]

        # Use the same formatting as SFT but create chosen/rejected pairs
        chosen_response = format_sft_response(sample, cfg)

        # Create rejected response using dpo_comment
        rejected_sample = sample.copy()
        rejected_sample[cfg.dataset.sft_column] = sample["dpo_comment"]
        rejected_response = format_sft_response(rejected_sample, cfg)

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    dpo_dataset = dpo_train_set.map(format_dpo_sample)

    # Configure DPO training
    dpo_config = DPOConfig(
        output_dir="dpo_baseline_output",
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=200,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",  # Enable wandb logging
        remove_unused_columns=False,
        beta=0.1,  # DPO beta parameter
    )

    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training")
    dpo_trainer.train()
    logger.info("DPO training completed")

    # Save DPO model
    model.save_pretrained("dpo_baseline_checkpoint")
    tokenizer.save_pretrained("dpo_baseline_checkpoint")
    logger.info("Saved DPO model checkpoint")

    # Test the DPO model
    test_results = test_model(dpo_trainer.model, tokenizer, test_set, cfg)

    # Calculate and log final metrics
    total_responses = len(test_results)
    correct_responses = sum(1 for r in test_results if r["is_correct"])
    unparseable_responses = sum(
        1 for r in test_results if not r.get("parsed_successfully", True)
    )
    accuracy = correct_responses / total_responses if total_responses > 0 else 0
    parsing_success_rate = (
        (total_responses - unparseable_responses) / total_responses
        if total_responses > 0
        else 0
    )

    logger.info(
        f"DPO baseline final accuracy: {correct_responses}/{total_responses} ({accuracy * 100:.1f}%)"
    )
    logger.info(f"Parsing success rate: {parsing_success_rate * 100:.1f}%")

    # Log metrics to wandb
    metrics = {
        "accuracy": accuracy,
        "parsing_success_rate": parsing_success_rate,
        "correct_answers": correct_responses,
        "total_answers": total_responses,
        "unparseable_answers": unparseable_responses,
    }

    wandb.log(metrics)

    # Log results table to wandb
    results_table = wandb.Table(
        columns=[
            "sample_id",
            "repeat_id",
            "question",
            "predicted_answer",
            "ground_truth",
            "is_correct",
            "parsed_successfully",
            "prob_A",
            "prob_B",
            "prob_C",
        ]
    )
    for result in test_results[:100]:  # Log first 100 results to avoid huge tables
        results_table.add_data(
            result.get("sample_id", 0),
            result.get("repeat_id", 0),
            result.get("question", "")[:100] + "..."
            if len(result.get("question", "")) > 100
            else result.get("question", ""),
            result.get("predicted_answer", ""),
            result.get("ground_truth", ""),
            result.get("is_correct", False),
            result.get("parsed_successfully", True),
            result.get("answer_prob_A", 0),
            result.get("answer_prob_B", 0),
            result.get("answer_prob_C", 0),
        )

    wandb.log({"results_sample": results_table})

    return test_results


def run_sft_baseline(train_set: Dataset, test_set: Dataset, cfg: Dict[str, Any]):
    """Run SFT (Supervised Fine-Tuning) baseline.

    Fine-tunes a model on reasoning traces using supervised learning,
    then evaluates on test set.

    Args:
        train_set: Training dataset with sft_column field.
        test_set: Test dataset to evaluate.
        cfg: Configuration with dataset.sft_column and optional sft.* parameters.

    Raises:
        ValueError: If sft_column is not configured.

    Outputs:
        - sft_baseline_checkpoint/: Saved model
        - sft_baseline_test_results.json: Test evaluation results
        - Logs metrics to wandb
    """
    logger.info("Running SFT baseline")

    # Check if required configuration exists
    if cfg.dataset.get("sft_column") is None:
        raise ValueError("SFT baseline requires 'sft_column' to be configured")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Prepare SFT dataset - filter out samples without SFT data
    sft_train_set = train_set.filter(
        lambda x: (
            x[cfg.dataset.sft_column] != "" and x[cfg.dataset.sft_column] is not None
        )
    )

    logger.info(
        f"Filtered SFT training set: {len(sft_train_set)} samples (from {len(train_set)} original)"
    )

    # Format SFT dataset
    def format_sft_sample(sample):
        prompt = [
            {"role": "system", "content": cfg.dataset.system_prompt},
            {"role": "user", "content": sample[cfg.dataset.question_column]},
        ]

        # Format the response using the same logic as DPO
        response = format_sft_response(sample, cfg)

        return {"messages": [*prompt, {"role": "assistant", "content": response}]}

    # Convert to text format for SFTTrainer
    def format_for_sft_trainer(sample):
        messages = sample["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    sft_dataset = (
        sft_train_set.map(format_sft_sample)
        .map(format_for_sft_trainer)
        .select_columns(["text"])
    )

    # Configure SFT training
    sft_config = SFTConfig(
        output_dir="sft_baseline_output",
        learning_rate=cfg.get("sft", {}).get("learning_rate", 5e-5),
        per_device_train_batch_size=cfg.get("sft", {}).get(
            "per_device_train_batch_size", 4
        ),
        gradient_accumulation_steps=cfg.get("sft", {}).get(
            "gradient_accumulation_steps", 2
        ),
        num_train_epochs=cfg.get("sft", {}).get("num_train_epochs", 1),
        max_steps=cfg.get("sft", {}).get("max_steps", 100),
        logging_steps=cfg.get("sft", {}).get("logging_steps", 10),
        save_steps=cfg.get("sft", {}).get("save_steps", 50),
        report_to="wandb",
        remove_unused_columns=False,
        max_seq_length=cfg.get("sft", {}).get("max_seq_length", 2600),
        packing=cfg.get("sft", {}).get("packing", True),
    )

    # Initialize SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=sft_dataset,
    )

    logger.info("Starting SFT training")
    sft_trainer.train()
    logger.info("SFT training completed")

    # Save SFT model
    model.save_pretrained("sft_baseline_checkpoint")
    tokenizer.save_pretrained("sft_baseline_checkpoint")
    logger.info("Saved SFT model checkpoint")

    # Test the SFT model
    test_results = test_model(sft_trainer.model, tokenizer, test_set, cfg)

    # Calculate and log final metrics
    total_responses = len(test_results)
    correct_responses = sum(1 for r in test_results if r["is_correct"])
    unparseable_responses = sum(
        1 for r in test_results if not r.get("parsed_successfully", True)
    )
    accuracy = correct_responses / total_responses if total_responses > 0 else 0
    parsing_success_rate = (
        (total_responses - unparseable_responses) / total_responses
        if total_responses > 0
        else 0
    )

    logger.info(
        f"SFT baseline final accuracy: {correct_responses}/{total_responses} ({accuracy * 100:.1f}%)"
    )
    logger.info(f"Parsing success rate: {parsing_success_rate * 100:.1f}%")

    # Log metrics to wandb
    metrics = {
        "accuracy": accuracy,
        "parsing_success_rate": parsing_success_rate,
        "correct_answers": correct_responses,
        "total_answers": total_responses,
        "unparseable_answers": unparseable_responses,
    }

    wandb.log(metrics)

    # Save results locally
    output_file = "sft_baseline_test_results.json"
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Saved {len(test_results)} test results to {output_file}")

    # Log results table to wandb
    results_table = wandb.Table(
        columns=[
            "sample_id",
            "repeat_id",
            "question",
            "predicted_answer",
            "ground_truth",
            "is_correct",
            "parsed_successfully",
            "prob_A",
            "prob_B",
            "prob_C",
        ]
    )
    for result in test_results:
        results_table.add_data(
            result.get("sample_id", 0),
            result.get("repeat_id", 0),
            result.get("question", "")[:100] + "..."
            if len(result.get("question", "")) > 100
            else result.get("question", ""),
            result.get("predicted_answer", ""),
            result.get("ground_truth", ""),
            result.get("is_correct", False),
            result.get("parsed_successfully", True),
            result.get("answer_prob_A", 0),
            result.get("answer_prob_B", 0),
            result.get("answer_prob_C", 0),
        )

    wandb.log({"results_sample": results_table})

    return test_results


def run_sft_dpo_baseline(train_set: Dataset, test_set: Dataset, cfg: Dict[str, Any]):
    """Run SFT followed by DPO baseline.

    Two-phase training: First performs supervised fine-tuning, then applies
    Direct Preference Optimization on top of the SFT model.

    Args:
        train_set: Training dataset with sft_column and dpo_comment fields.
        test_set: Test dataset to evaluate.
        cfg: Configuration with dataset.sft_column, dataset.dpo_comment,
             and optional sft.* and dpo.* parameters.

    Raises:
        ValueError: If required columns (sft_column, dpo_comment) are missing.

    Outputs:
        - sft_dpo_baseline_sft_checkpoint/: Intermediate SFT model
        - sft_dpo_baseline_final_checkpoint/: Final SFT+DPO model
        - sft_dpo_baseline_test_results.json: Test evaluation results
        - Logs metrics to wandb
    """
    logger.info("Running SFT + DPO baseline")

    # Check if required columns exist for both SFT and DPO
    if cfg.dataset.get("sft_column") is None:
        raise ValueError("SFT + DPO baseline requires 'sft_column' to be configured")
    if "dpo_comment" not in train_set.column_names:
        raise ValueError(
            "SFT + DPO baseline requires 'dpo_comment' column for DPO rejected generations"
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # First phase: SFT training
    logger.info("Phase 1: SFT training")

    # Prepare SFT dataset - filter out samples without SFT data
    sft_train_set = train_set.filter(
        lambda x: (
            x[cfg.dataset.sft_column] != "" and x[cfg.dataset.sft_column] is not None
        )
    )

    logger.info(
        f"Filtered SFT training set: {len(sft_train_set)} samples (from {len(train_set)} original)"
    )

    # Format SFT dataset
    def format_sft_sample(sample):
        prompt = [
            {"role": "system", "content": cfg.dataset.system_prompt},
            {"role": "user", "content": sample[cfg.dataset.question_column]},
        ]

        response = format_sft_response(sample, cfg)

        return {"messages": [*prompt, {"role": "assistant", "content": response}]}

    def format_for_sft_trainer(sample):
        messages = sample["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    sft_dataset = (
        sft_train_set.map(format_sft_sample)
        .map(format_for_sft_trainer)
        .select_columns(["text"])
    )

    # Configure SFT training
    sft_config = SFTConfig(
        output_dir="sft_dpo_baseline_sft_output",
        learning_rate=cfg.get("sft", {}).get("learning_rate", 5e-5),
        per_device_train_batch_size=cfg.get("sft", {}).get(
            "per_device_train_batch_size", 4
        ),
        gradient_accumulation_steps=cfg.get("sft", {}).get(
            "gradient_accumulation_steps", 2
        ),
        num_train_epochs=cfg.get("sft", {}).get("num_train_epochs", 1),
        max_steps=cfg.get("sft", {}).get("max_steps", 100),
        logging_steps=cfg.get("sft", {}).get("logging_steps", 10),
        save_steps=cfg.get("sft", {}).get("save_steps", 50),
        report_to="wandb",
        remove_unused_columns=False,
        max_seq_length=cfg.get("sft", {}).get("max_seq_length", 2600),
        packing=cfg.get("sft", {}).get("packing", True),
    )

    # Initialize SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=sft_dataset,
    )

    logger.info("Starting SFT training")
    sft_trainer.train()
    logger.info("SFT training completed")

    # Save intermediate SFT model
    model.save_pretrained("sft_dpo_baseline_sft_checkpoint")
    tokenizer.save_pretrained("sft_dpo_baseline_sft_checkpoint")
    logger.info("Saved intermediate SFT model checkpoint")

    # Second phase: DPO training on the SFT model
    logger.info("Phase 2: DPO training on SFT model")

    # Prepare DPO dataset - filter out samples without both chosen and rejected responses
    dpo_train_set = train_set.filter(
        lambda x: (
            x[cfg.dataset.sft_column] != ""
            and x[cfg.dataset.sft_column] is not None
            and x["dpo_comment"] != ""
            and x["dpo_comment"] is not None
        )
    )

    logger.info(
        f"Filtered DPO training set: {len(dpo_train_set)} samples (from {len(train_set)} original)"
    )

    # Format DPO dataset with chosen/rejected pairs
    def format_dpo_sample(sample):
        prompt = [
            {"role": "system", "content": cfg.dataset.system_prompt},
            {"role": "user", "content": sample[cfg.dataset.question_column]},
        ]

        # Use the same formatting as SFT but create chosen/rejected pairs
        chosen_response = format_sft_response(sample, cfg)

        # Create rejected response using dpo_comment
        rejected_sample = sample.copy()
        rejected_sample[cfg.dataset.sft_column] = sample["dpo_comment"]
        rejected_response = format_sft_response(rejected_sample, cfg)

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    dpo_dataset = dpo_train_set.map(format_dpo_sample)

    # Configure DPO training
    dpo_config = DPOConfig(
        output_dir="sft_dpo_baseline_dpo_output",
        learning_rate=cfg.get("dpo", {}).get("learning_rate", 5e-6),
        per_device_train_batch_size=cfg.get("dpo", {}).get(
            "per_device_train_batch_size", 2
        ),
        gradient_accumulation_steps=cfg.get("dpo", {}).get(
            "gradient_accumulation_steps", 4
        ),
        num_train_epochs=cfg.get("dpo", {}).get("num_train_epochs", 1),
        max_steps=cfg.get("dpo", {}).get("max_steps", 200),
        logging_steps=cfg.get("dpo", {}).get("logging_steps", 10),
        save_steps=cfg.get("dpo", {}).get("save_steps", 100),
        report_to="wandb",
        remove_unused_columns=False,
        beta=cfg.get("dpo", {}).get("beta", 0.1),
    )

    # Initialize DPO trainer with the SFT-trained model
    dpo_trainer = DPOTrainer(
        model=sft_trainer.model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training")
    dpo_trainer.train()
    logger.info("DPO training completed")

    # Save final SFT+DPO model
    dpo_trainer.model.save_pretrained("sft_dpo_baseline_final_checkpoint")
    tokenizer.save_pretrained("sft_dpo_baseline_final_checkpoint")
    logger.info("Saved final SFT+DPO model checkpoint")

    # Test the SFT+DPO model
    test_results = test_model(dpo_trainer.model, tokenizer, test_set, cfg)

    # Calculate and log final metrics
    total_responses = len(test_results)
    correct_responses = sum(1 for r in test_results if r["is_correct"])
    unparseable_responses = sum(
        1 for r in test_results if not r.get("parsed_successfully", True)
    )
    accuracy = correct_responses / total_responses if total_responses > 0 else 0
    parsing_success_rate = (
        (total_responses - unparseable_responses) / total_responses
        if total_responses > 0
        else 0
    )

    logger.info(
        f"SFT+DPO baseline final accuracy: {correct_responses}/{total_responses} ({accuracy * 100:.1f}%)"
    )
    logger.info(f"Parsing success rate: {parsing_success_rate * 100:.1f}%")

    # Log metrics to wandb
    metrics = {
        "accuracy": accuracy,
        "parsing_success_rate": parsing_success_rate,
        "correct_answers": correct_responses,
        "total_answers": total_responses,
        "unparseable_answers": unparseable_responses,
    }

    wandb.log(metrics)

    # Save results locally
    output_file = "sft_dpo_baseline_test_results.json"
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Saved {len(test_results)} test results to {output_file}")

    # Log results table to wandb
    results_table = wandb.Table(
        columns=[
            "sample_id",
            "repeat_id",
            "question",
            "predicted_answer",
            "ground_truth",
            "is_correct",
            "parsed_successfully",
            "prob_A",
            "prob_B",
            "prob_C",
        ]
    )
    for result in test_results[:100]:  # Log first 100 results to avoid huge tables
        results_table.add_data(
            result.get("sample_id", 0),
            result.get("repeat_id", 0),
            result.get("question", "")[:100] + "..."
            if len(result.get("question", "")) > 100
            else result.get("question", ""),
            result.get("predicted_answer", ""),
            result.get("ground_truth", ""),
            result.get("is_correct", False),
            result.get("parsed_successfully", True),
            result.get("answer_prob_A", 0),
            result.get("answer_prob_B", 0),
            result.get("answer_prob_C", 0),
        )

    wandb.log({"results_sample": results_table})

    return test_results


def main():
    """Main function for running baseline experiments.

    Command-line interface for running various baseline methods (SFT, DPO,
    In-Context Learning, etc.) with configurable dataset splits.

    Usage:
        reasoning-baseline --method sft --config params.yaml
        reasoning-baseline --method dpo --train-split train --test-split test

    Available methods:
        - in_context_learning: Few-shot prompting
        - sft: Supervised fine-tuning
        - dpo: Direct Preference Optimization
        - sft_dpo: SFT followed by DPO
    """
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for political reasoning"
    )
    parser.add_argument(
        "--config", type=str, default="params.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Baseline method to run (overrides config)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help="Training split to use (overrides config)",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=None,
        help="Test split to use (overrides config)",
    )
    args = parser.parse_args()

    logger.info("Starting baseline experiments")

    # Build overrides dictionary from command-line arguments
    overrides = {}
    if args.train_split:
        overrides["dataset"] = overrides.get("dataset", {})
        overrides["dataset"]["train_split"] = args.train_split
    if args.test_split:
        overrides["dataset"] = overrides.get("dataset", {})
        overrides["dataset"]["test_split"] = args.test_split

    # Prepare data
    train_set, test_set, cfg = prepare_data(args.config, overrides)

    # Determine method to run
    if args.method:
        method = args.method
    elif "baseline" in cfg and "method" in cfg.baseline:
        method = cfg.baseline.method
    else:
        method = "dpo"  # Default fallback

    logger.info(f"Running baseline method: {method}")

    try:
        run_baseline_method(method, train_set, test_set, cfg, overrides)
    except Exception as e:
        logger.error(f"Error running baseline method {method}: {e}")
        raise


if __name__ == "__main__":
    main()
