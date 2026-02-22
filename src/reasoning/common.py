"""Common utilities shared across reasoning training and baseline methods."""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import wandb

logger = logging.getLogger(__name__)


def format_sft_response(sample, cfg):
    """Format SFT response based on dataset configuration.

    Supports multiple reasoning formats:
    - Two-stage: <think>...</think><confidence>...</confidence><think>...</think><answer>...</answer>
    - Multi-step: <step1>...</step1><step2>...</step2>...<answer>...</answer>
    - Single reasoning: <reasoning>...</reasoning><answer>...</answer>

    Args:
        sample: Dataset sample containing question, reasoning, and answer.
        cfg: Configuration object with dataset format specifications.

    Returns:
        Formatted string with reasoning and answer tags.
    """
    answer = sample[cfg.dataset.answer_column]

    # Check if two-stage reasoning is configured
    if hasattr(cfg.dataset, 'two_stage_reasoning') and cfg.dataset.two_stage_reasoning:
        # Two-stage format: <think>...</think><confidence>...</confidence><think>...</think><answer>...</answer>
        think1_column = cfg.dataset.two_stage_reasoning.get('think1_column', 'think1')
        confidence_column = cfg.dataset.two_stage_reasoning.get('confidence_column', 'confidence')
        think2_column = cfg.dataset.two_stage_reasoning.get('think2_column', 'think2')

        if all(col in sample and sample[col] for col in [think1_column, confidence_column, think2_column]):
            think1 = sample[think1_column]
            confidence = sample[confidence_column]
            think2 = sample[think2_column]
            return f"<think>{think1}</think><confidence>{confidence}</confidence><think>{think2}</think><answer>{answer}</answer>"

    # Check if multi-step reasoning is configured
    if hasattr(cfg.dataset, 'multistep_reasoning_columns') and cfg.dataset.multistep_reasoning_columns:
        # Multi-step format
        steps = []
        for step_name, column_key in cfg.dataset.multistep_reasoning_columns.items():
            if column_key in sample and sample[column_key]:
                step_content = sample[column_key]
                steps.append(f"<{step_name}>{step_content}</{step_name}>")

        if steps:
            steps_text = "".join(steps)
            return f"{steps_text}<answer>{answer}</answer>"

    # Fall back to traditional single reasoning format
    # Get configurable reasoning tag name (defaults to "reasoning" for backward compatibility)
    reasoning_tag = getattr(cfg.dataset, 'reasoning_tag_name', 'reasoning')
    reasoning = sample[cfg.dataset.sft_column]
    return f"<{reasoning_tag}>{reasoning}</{reasoning_tag}><answer>{answer}</answer>"


def extract_answer_probabilities(model, tokenizer, full_prompt: str) -> Dict[str, float]:
    """Extract answer probabilities for options A, B, C from model logits.

    Tokenizes the prompt, gets model logits at the last position, extracts
    logits for answer tokens A/B/C, and applies softmax to get probabilities.

    Args:
        model: Language model to query.
        tokenizer: Tokenizer for the model.
        full_prompt: Complete prompt ending at answer position.

    Returns:
        Dictionary with keys 'A', 'B', 'C' and probability values.
        Returns uniform distribution on error.
    """
    try:
        # Get token IDs for answer options
        answer_tokens = {
            "A": tokenizer.encode("A", add_special_tokens=False)[0],
            "B": tokenizer.encode("B", add_special_tokens=False)[0],
            "C": tokenizer.encode("C", add_special_tokens=False)[0],
        }

        # Tokenize the full prompt
        inputs = tokenizer(
            full_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        # Get logits at the last position
        with torch.no_grad():
            logits_output = model(**inputs)
            next_token_logits = logits_output.logits[0, -1, :]

            # Extract logits for answer tokens
            answer_logits = {
                "A": next_token_logits[answer_tokens["A"]].item(),
                "B": next_token_logits[answer_tokens["B"]].item(),
                "C": next_token_logits[answer_tokens["C"]].item(),
            }

            # Convert to probabilities using softmax
            answer_logits_tensor = torch.tensor(
                [answer_logits["A"], answer_logits["B"], answer_logits["C"]]
            )
            answer_probs_tensor = F.softmax(answer_logits_tensor, dim=0)

            answer_probs = {
                "A": answer_probs_tensor[0].item(),
                "B": answer_probs_tensor[1].item(),
                "C": answer_probs_tensor[2].item(),
            }

    except Exception as e:
        logger.warning(f"Could not extract answer probabilities: {e}")
        answer_probs = {"A": 0.333, "B": 0.333, "C": 0.333}

    return answer_probs


def test_model(model, tokenizer, test_set, cfg):
    """Test the model on the test set and save results to disk (and optionally log to wandb).

    Generates multiple responses per test sample, extracts answer probabilities,
    parses structured outputs, and calculates accuracy metrics.

    Args:
        model: The trained model to evaluate.
        tokenizer: Tokenizer for the model.
        test_set: Dataset containing test samples with 'prompt' and 'answers' fields.
        cfg: Configuration object with wandb, dataset, and evaluation settings.

    Returns:
        List of result dictionaries with predictions and metrics for each test sample.

    Outputs:
        - test_results.json: Detailed results for each test sample and repeat
        - test_results.csv: CSV format of results
        - test_summary.json: Aggregate metrics (accuracy, parsing rate, etc.)
        - wandb logs: Results table and metrics (if wandb is enabled)
    """
    logger.info("Starting model evaluation on test set")

    # Get evaluation config with defaults
    eval_cfg = cfg.get("evaluation", {})
    num_repeats = eval_cfg.get("num_repeats", 8)
    temperature = eval_cfg.get("temperature", 0.9)
    max_new_tokens = eval_cfg.get("max_new_tokens", 512)

    results = []
    correct_answers = 0
    unparseable_answers = 0
    total_answers = 0

    for i, sample in enumerate(test_set):
        sample_results = []

        # Run each prompt multiple times
        for repeat in range(num_repeats):
            try:
                # Use the same prompt format as in training
                prompt = sample["prompt"]

                # Generate response
                inputs = tokenizer.apply_chat_template(
                    prompt, tokenize=True, return_tensors="pt", return_token_type_ids=False
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    # Generate response
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Decode response
                response = tokenizer.decode(
                    outputs.sequences[0][inputs.shape[1] :], skip_special_tokens=True
                )

                # Extract logits/probabilities for answer options A, B, C
                answer_probs = None
                try:
                    # Get token IDs for answer options A, B, C only
                    answer_tokens = {
                        'A': tokenizer.encode('A', add_special_tokens=False)[0],
                        'B': tokenizer.encode('B', add_special_tokens=False)[0],
                        'C': tokenizer.encode('C', add_special_tokens=False)[0]
                    }

                    # Find the position where <answer> tag appears in the generated text
                    if "<answer>" in response:
                        # Create a prompt that ends just before the answer
                        answer_start = response.find("<answer>") + len("<answer>")
                        pre_answer_text = response[:answer_start]

                        # Tokenize the full prompt + generated text up to answer
                        full_prompt = tokenizer.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=True
                        ) + pre_answer_text

                        full_inputs = tokenizer(
                            full_prompt, return_tensors="pt", add_special_tokens=False
                        ).to(model.device)

                        # Get logits at the answer position
                        with torch.no_grad():
                            logits_output = model(**full_inputs)
                            next_token_logits = logits_output.logits[0, -1, :]

                            # Extract logits for answer tokens A, B, C only
                            answer_logits = {
                                'A': next_token_logits[answer_tokens['A']].item(),
                                'B': next_token_logits[answer_tokens['B']].item(),
                                'C': next_token_logits[answer_tokens['C']].item()
                            }

                            # Convert to probabilities using softmax over just these 3 options
                            answer_logits_tensor = torch.tensor([
                                answer_logits['A'], answer_logits['B'],
                                answer_logits['C']
                            ])
                            answer_probs_tensor = F.softmax(answer_logits_tensor, dim=0)

                            answer_probs = {
                                'A': answer_probs_tensor[0].item(),
                                'B': answer_probs_tensor[1].item(),
                                'C': answer_probs_tensor[2].item()
                            }

                except Exception as e:
                    logger.warning(f"Could not extract answer probabilities: {e}")
                    answer_probs = {'A': 0.333, 'B': 0.333, 'C': 0.333}

                # Extract answer from response if it follows the <answer> format
                parsed_successfully = True
                confidence_score = None

                if "<answer>" in response and "</answer>" in response:
                    predicted_answer = (
                        response.split("<answer>")[1].split("</answer>")[0].strip()
                    )

                    # Also extract confidence score if it's a two-stage format
                    if "<confidence>" in response and "</confidence>" in response:
                        confidence_raw = response.split("<confidence>")[1].split("</confidence>")[0].strip()
                        # Try to extract numeric confidence score
                        try:
                            confidence_match = re.search(r'(\d+(?:\.\d+)?)', confidence_raw)
                            if confidence_match:
                                confidence_score = float(confidence_match.group(1))
                                # Normalize to 0-1 range if it's a percentage (0-100)
                                if confidence_score > 1:
                                    confidence_score = confidence_score / 100
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Could not parse confidence score: {e}")
                            confidence_score = None
                else:
                    predicted_answer = response.strip()
                    # If no structured answer tags, consider it unparseable
                    if not predicted_answer or len(predicted_answer) > 200:
                        parsed_successfully = False
                        unparseable_answers += 1

                # Get ground truth
                ground_truth = sample["answers"]

                # Check if answer is correct (case-insensitive comparison)
                is_correct = False
                if parsed_successfully and predicted_answer.lower().strip() == ground_truth.lower().strip():
                    is_correct = True
                    correct_answers += 1

                total_answers += 1

                # Store result
                result = {
                    "question": sample["prompt"][-1]["content"] if sample["prompt"] else "",
                    "predicted_answer": predicted_answer,
                    "ground_truth": ground_truth,
                    "full_response": response,
                    "sample_id": i,
                    "repeat_id": repeat,
                    "parsed_successfully": parsed_successfully,
                    "is_correct": is_correct,
                    "confidence_score": confidence_score,
                    "answer_prob_A": answer_probs['A'] if answer_probs else None,
                    "answer_prob_B": answer_probs['B'] if answer_probs else None,
                    "answer_prob_C": answer_probs['C'] if answer_probs else None,
                }
                sample_results.append(result)

            except Exception as e:
                logger.error(f"Error processing sample {i}, repeat {repeat}: {e}")
                unparseable_answers += 1
                total_answers += 1
                continue

        results.extend(sample_results)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(test_set)} test samples ({(i+1)*num_repeats} total responses)")

    logger.info(f"Evaluation complete: {correct_answers}/{total_answers} correct ({correct_answers/total_answers*100:.1f}%), {unparseable_answers} unparseable")

    # Save results to disk as JSON
    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} test results to {output_file}")

    # Save results to CSV for easier analysis
    csv_file = "test_results.csv"
    if results:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample_id", "repeat_id", "question", "predicted_answer",
                "ground_truth", "full_response", "parsed_successfully", "is_correct",
                "confidence_score", "answer_prob_A", "answer_prob_B", "answer_prob_C"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        logger.info(f"Saved {len(results)} test results to {csv_file}")

    # Check if wandb is enabled via config or environment
    use_wandb = cfg.get("use_wandb", True)
    if os.environ.get("DISABLE_WANDB", "").lower() in ["true", "1", "yes"]:
        use_wandb = False

    # Log results to wandb if enabled
    if use_wandb and wandb.run is not None:
        # Create a wandb table for detailed results
        table = wandb.Table(
            columns=[
                "sample_id",
                "repeat_id",
                "question",
                "predicted_answer",
                "ground_truth",
                "full_response",
                "parsed_successfully",
                "is_correct",
                "confidence_score",
                "answer_prob_A",
                "answer_prob_B",
                "answer_prob_C",
            ]
        )
        for result in results:
            table.add_data(
                result["sample_id"],
                result["repeat_id"],
                result["question"],
                result["predicted_answer"],
                result["ground_truth"],
                result["full_response"],
                result["parsed_successfully"],
                result["is_correct"],
                result["confidence_score"],
                result["answer_prob_A"],
                result["answer_prob_B"],
                result["answer_prob_C"],
            )

        # Calculate accuracy
        accuracy = correct_answers / total_answers if total_answers > 0 else 0
        parsing_success_rate = (total_answers - unparseable_answers) / total_answers if total_answers > 0 else 0

        # Calculate average probabilities for each answer option (A, B, C only)
        valid_results = [r for r in results if r["answer_prob_A"] is not None]
        if valid_results:
            avg_prob_A = sum(r["answer_prob_A"] for r in valid_results) / len(valid_results)
            avg_prob_B = sum(r["answer_prob_B"] for r in valid_results) / len(valid_results)
            avg_prob_C = sum(r["answer_prob_C"] for r in valid_results) / len(valid_results)
        else:
            avg_prob_A = avg_prob_B = avg_prob_C = 0.333

        # Calculate confidence score metrics
        confidence_results = [r for r in results if r["confidence_score"] is not None]
        avg_confidence = sum(r["confidence_score"] for r in confidence_results) / len(confidence_results) if confidence_results else None
        confidence_availability_rate = len(confidence_results) / len(results) if results else 0

        wandb.log(
            {
                "test_results": table,
                "test_samples_processed": len(results),
                "test_samples_total": len(test_set),
                "total_responses": total_answers,
                "correct_answers": correct_answers,
                "unparseable_answers": unparseable_answers,
                "accuracy": accuracy,
                "parsing_success_rate": parsing_success_rate,
                "avg_answer_prob_A": avg_prob_A,
                "avg_answer_prob_B": avg_prob_B,
                "avg_answer_prob_C": avg_prob_C,
                "avg_confidence": avg_confidence,
                "confidence_availability_rate": confidence_availability_rate,
                "probability_extraction_success_rate": len(valid_results) / len(results) if results else 0,
            }
        )

        logger.info(f"Logged {len(results)} test results to wandb")
    elif use_wandb:
        logger.info("wandb run not active, skipping result logging")
    else:
        logger.info("wandb logging disabled via configuration")

    # Calculate and save summary statistics
    accuracy = correct_answers / total_answers if total_answers > 0 else 0
    parsing_success_rate = (total_answers - unparseable_answers) / total_answers if total_answers > 0 else 0

    # Calculate average probabilities for each answer option
    valid_results = [r for r in results if r["answer_prob_A"] is not None]
    if valid_results:
        avg_prob_A = sum(r["answer_prob_A"] for r in valid_results) / len(valid_results)
        avg_prob_B = sum(r["answer_prob_B"] for r in valid_results) / len(valid_results)
        avg_prob_C = sum(r["answer_prob_C"] for r in valid_results) / len(valid_results)
    else:
        avg_prob_A = avg_prob_B = avg_prob_C = 0.333

    # Calculate confidence score metrics
    confidence_results = [r for r in results if r["confidence_score"] is not None]
    avg_confidence = sum(r["confidence_score"] for r in confidence_results) / len(confidence_results) if confidence_results else None
    confidence_availability_rate = len(confidence_results) / len(results) if results else 0

    # Save summary statistics
    summary = {
        "test_samples_processed": len(results),
        "test_samples_total": len(test_set),
        "total_responses": total_answers,
        "correct_answers": correct_answers,
        "unparseable_answers": unparseable_answers,
        "accuracy": accuracy,
        "parsing_success_rate": parsing_success_rate,
        "avg_answer_prob_A": avg_prob_A,
        "avg_answer_prob_B": avg_prob_B,
        "avg_answer_prob_C": avg_prob_C,
        "avg_confidence": avg_confidence,
        "confidence_availability_rate": confidence_availability_rate,
        "probability_extraction_success_rate": len(valid_results) / len(results) if results else 0,
    }

    summary_file = "test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved test summary to {summary_file}")

    return results
