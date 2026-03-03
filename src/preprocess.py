"""Dataset loading and preprocessing for EA-CoT experiment."""

from datasets import load_dataset
from typing import List, Dict
import os


def load_gsm8k_data(
    cache_dir: str = ".cache", num_tuning: int = 50, num_eval: int = 150
) -> Dict[str, List[Dict]]:
    """
    Load GSM8K dataset and split into tuning and evaluation sets.

    Args:
        cache_dir: Directory to cache the dataset
        num_tuning: Number of examples for threshold tuning
        num_eval: Number of examples for final evaluation

    Returns:
        Dictionary with 'tuning' and 'eval' keys containing lists of examples
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Load GSM8K test split
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)

    # Take first (num_tuning + num_eval) examples
    total_needed = num_tuning + num_eval
    dataset = dataset.select(range(min(total_needed, len(dataset))))

    # Convert to list of dicts
    examples = []
    for item in dataset:
        # Extract numeric answer from the format "#### 123"
        answer_str = item["answer"].split("####")[-1].strip()
        try:
            answer = float(answer_str.replace(",", ""))
        except ValueError:
            # If parsing fails, keep as string
            answer = answer_str

        examples.append(
            {
                "question": item["question"],
                "answer": answer,
                "answer_str": item["answer"],  # Keep original for reference
            }
        )

    # Split into tuning and eval
    tuning_examples = examples[:num_tuning]
    eval_examples = examples[num_tuning : num_tuning + num_eval]

    return {"tuning": tuning_examples, "eval": eval_examples}


def extract_numeric_answer(text: str) -> float:
    """
    Extract numeric answer from model output.
    Handles various formats like "The answer is 42" or "42" or "#### 42"
    """
    import re

    # Try to find patterns like "#### NUMBER" (GSM8K format)
    match = re.search(r"####\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))

    # Try to find "answer is NUMBER" patterns
    match = re.search(r"answer\s+is\s+([0-9,]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))

    # Try to find any number in the text (prefer the last one)
    numbers = re.findall(r"([0-9,]+(?:\.[0-9]+)?)", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    # If no number found, raise error
    raise ValueError(f"Could not extract numeric answer from: {text[:100]}")
