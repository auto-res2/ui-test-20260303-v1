"""Inference module implementing all prompting methods."""

import os
import json
import wandb
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.preprocess import load_gsm8k_data, extract_numeric_answer


def get_llm_client(provider: str, api_key: Optional[str] = None):
    """Initialize LLM client based on provider."""
    if provider == "openai":
        import openai

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        return openai.OpenAI(api_key=api_key)
    elif provider == "anthropic":
        import anthropic

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(
    client,
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Tuple[str, int]:
    """
    Call LLM and return response text and token count.

    Returns:
        Tuple of (response_text, token_count)
    """
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        return text, tokens
    elif provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens
    else:
        raise ValueError(f"Unknown provider: {provider}")


def method_direct(
    client,
    provider: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Direct answering without reasoning."""
    prompt = f"""Answer this math question with just the final numeric answer.

Question: {question}

Answer (just the number):"""

    response, tokens = call_llm(
        client, provider, model, prompt, temperature, max_tokens
    )

    try:
        answer = extract_numeric_answer(response)
    except ValueError:
        answer = None

    return {
        "method": "direct",
        "response": response,
        "answer": answer,
        "total_tokens": tokens,
        "triggered_cot": False,
    }


def method_fixed_cot(
    client,
    provider: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Fixed CoT - always use reasoning."""
    prompt = f"""Solve this math question step by step, then provide the final numeric answer.

Question: {question}

Let's solve this step by step:"""

    response, tokens = call_llm(
        client, provider, model, prompt, temperature, max_tokens
    )

    try:
        answer = extract_numeric_answer(response)
    except ValueError:
        answer = None

    return {
        "method": "fixed_cot",
        "response": response,
        "answer": answer,
        "total_tokens": tokens,
        "triggered_cot": True,
    }


def method_ca_cot(
    client,
    provider: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Confidence-Adaptive CoT - trigger reasoning if confidence is low."""
    # Stage 1: Get draft answer + confidence
    prompt1 = f"""Answer this math question and rate your confidence.

Question: {question}

Provide:
1. Your answer (just the number)
2. Your confidence (0.0 to 1.0)

Format your response as:
Answer: <number>
Confidence: <0.0-1.0>"""

    response1, tokens1 = call_llm(client, provider, model, prompt1, temperature, 100)

    # Parse confidence
    confidence = 0.5  # default
    try:
        lines = response1.strip().split("\n")
        for line in lines:
            if "confidence:" in line.lower():
                conf_str = line.split(":")[-1].strip()
                confidence = float(conf_str)
                break
    except:
        pass

    # Decide whether to trigger CoT
    triggered = confidence < confidence_threshold

    if not triggered:
        # Use draft answer
        try:
            answer = extract_numeric_answer(response1)
        except ValueError:
            answer = None

        return {
            "method": "ca_cot",
            "response": response1,
            "answer": answer,
            "confidence": confidence,
            "total_tokens": tokens1,
            "triggered_cot": False,
        }
    else:
        # Trigger full CoT
        prompt2 = f"""Solve this math question step by step, then provide the final numeric answer.

Question: {question}

Let's solve this step by step:"""

        response2, tokens2 = call_llm(
            client, provider, model, prompt2, temperature, max_tokens
        )

        try:
            answer = extract_numeric_answer(response2)
        except ValueError:
            answer = None

        return {
            "method": "ca_cot",
            "response": response2,
            "answer": answer,
            "confidence": confidence,
            "total_tokens": tokens1 + tokens2,
            "triggered_cot": True,
        }


def method_ea_cot(
    client,
    provider: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    confidence_threshold: float,
    evidence_threshold: float,
    num_key_facts: int = 3,
) -> Dict[str, Any]:
    """Evidence-Adaptive CoT - trigger reasoning if confidence is low OR evidence is weak."""
    # Stage 1: Get draft answer + key facts + confidence
    prompt1 = f"""Answer this math question and provide supporting evidence.

Question: {question}

Provide:
1. Your answer (just the number)
2. {num_key_facts} key facts that must be true for your answer to be correct (brief, one per line)
3. Your confidence (0.0 to 1.0)

Format your response as:
Answer: <number>
Key Facts:
- Fact 1
- Fact 2
- Fact 3
Confidence: <0.0-1.0>"""

    response1, tokens1 = call_llm(client, provider, model, prompt1, temperature, 200)

    # Parse confidence and key facts
    confidence = 0.5  # default
    key_facts = []
    try:
        lines = response1.strip().split("\n")
        in_facts = False
        for line in lines:
            line_lower = line.lower()
            if "confidence:" in line_lower:
                conf_str = line.split(":")[-1].strip()
                confidence = float(conf_str)
                in_facts = False
            elif "key facts:" in line_lower:
                in_facts = True
            elif in_facts and line.strip().startswith("-"):
                fact = line.strip()[1:].strip()
                if fact:
                    key_facts.append(fact)
    except:
        pass

    # Stage 2: Verify key facts
    if len(key_facts) > 0:
        facts_text = "\n".join([f"{i + 1}. {fact}" for i, fact in enumerate(key_facts)])
        prompt2 = f"""Given only the question below, mark each fact as SUPPORTED, UNSUPPORTED, or UNCLEAR based on whether it can be directly verified from the question alone (not the solution).

Question: {question}

Facts to verify:
{facts_text}

For each fact, respond with just: SUPPORTED, UNSUPPORTED, or UNCLEAR
Format: "1. SUPPORTED" etc."""

        response2, tokens2 = call_llm(
            client, provider, model, prompt2, temperature, 100
        )

        # Count supported facts
        supported_count = response2.upper().count("SUPPORTED")
        evidence_score = supported_count / len(key_facts) if len(key_facts) > 0 else 0.0
    else:
        tokens2 = 0
        evidence_score = 0.0
        response2 = "No facts extracted"

    # Decide whether to trigger CoT
    triggered = (confidence < confidence_threshold) or (
        evidence_score < evidence_threshold
    )

    if not triggered:
        # Use draft answer
        try:
            answer = extract_numeric_answer(response1)
        except ValueError:
            answer = None

        return {
            "method": "ea_cot",
            "response": response1,
            "answer": answer,
            "confidence": confidence,
            "evidence_score": evidence_score,
            "key_facts": key_facts,
            "total_tokens": tokens1 + tokens2,
            "triggered_cot": False,
        }
    else:
        # Trigger full CoT
        prompt3 = f"""Solve this math question step by step, then provide the final numeric answer.

Question: {question}

Let's solve this step by step:"""

        response3, tokens3 = call_llm(
            client, provider, model, prompt3, temperature, max_tokens
        )

        try:
            answer = extract_numeric_answer(response3)
        except ValueError:
            answer = None

        return {
            "method": "ea_cot",
            "response": response3,
            "answer": answer,
            "confidence": confidence,
            "evidence_score": evidence_score,
            "key_facts": key_facts,
            "total_tokens": tokens1 + tokens2 + tokens3,
            "triggered_cot": True,
        }


# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: cfg.method/model/dataset accessed incorrectly (should be under cfg.run)
# [CAUSE]: Run config is loaded under cfg.run namespace by Hydra config groups
# [FIX]: Update all config accessors to use cfg.run.* paths
#
# [OLD CODE]:
# cfg.model.temperature, cfg.model.max_tokens, cfg.method.num_key_facts
#
# [NEW CODE]:
def tune_thresholds(
    client,
    provider: str,
    model: str,
    tuning_data: List[Dict],
    method_type: str,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Tune thresholds on tuning set using grid search."""
    print(f"\nTuning thresholds for {method_type}...")

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: CoT never triggered (cot_trigger_rate=0) because confidence threshold too low
    # [CAUSE]: Models like GPT-4o-mini report very high confidence (>0.8), so threshold 0.8 never triggers CoT
    # [FIX]: Expand grid search to include higher thresholds (0.85, 0.9, 0.95) to allow triggering CoT when confidence is high but not perfect
    #
    # [OLD CODE]:
    # for conf_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    #
    # [NEW CODE]:
    if method_type == "ca_cot":
        # Grid search for confidence threshold
        best_acc = 0.0
        best_threshold = 0.5

        for conf_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            correct = 0
            for item in tuning_data[:20]:  # Use subset for faster tuning
                result = method_ca_cot(
                    client,
                    provider,
                    model,
                    item["question"],
                    cfg.run.model.temperature,
                    cfg.run.model.max_tokens,
                    conf_thresh,
                )
                if (
                    result["answer"] is not None
                    and abs(result["answer"] - item["answer"]) < 0.01
                ):
                    correct += 1

            acc = correct / 20
            print(f"  conf_threshold={conf_thresh:.2f}: acc={acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                best_threshold = conf_thresh

        print(f"Best confidence_threshold: {best_threshold} (acc={best_acc:.3f})")
        return {"confidence_threshold": best_threshold}

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: EA-CoT may also have low trigger rate due to insufficient confidence threshold range
    # [CAUSE]: Similar to CA-CoT, models report high confidence so threshold 0.7 may be too low
    # [FIX]: Expand confidence threshold grid to include higher values (0.8, 0.85, 0.9)
    #
    # [OLD CODE]:
    # for conf_thresh in [0.4, 0.5, 0.6, 0.7]:
    #
    # [NEW CODE]:
    elif method_type == "ea_cot":
        # Grid search for both thresholds
        best_acc = 0.0
        best_conf_thresh = 0.5
        best_ev_thresh = 0.5

        for conf_thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
            for ev_thresh in [0.3, 0.5, 0.7]:
                correct = 0
                for item in tuning_data[:20]:  # Use subset for faster tuning
                    result = method_ea_cot(
                        client,
                        provider,
                        model,
                        item["question"],
                        cfg.run.model.temperature,
                        cfg.run.model.max_tokens,
                        conf_thresh,
                        ev_thresh,
                        cfg.run.method.num_key_facts,
                    )
                    if (
                        result["answer"] is not None
                        and abs(result["answer"] - item["answer"]) < 0.01
                    ):
                        correct += 1

                acc = correct / 20
                print(f"  conf={conf_thresh:.2f}, ev={ev_thresh:.2f}: acc={acc:.3f}")

                if acc > best_acc:
                    best_acc = acc
                    best_conf_thresh = conf_thresh
                    best_ev_thresh = ev_thresh

        print(
            f"Best thresholds: conf={best_conf_thresh:.2f}, ev={best_ev_thresh:.2f} (acc={best_acc:.3f})"
        )
        return {
            "confidence_threshold": best_conf_thresh,
            "evidence_threshold": best_ev_thresh,
        }

    else:
        return {}


def run_inference(cfg: DictConfig) -> None:
    """Main inference runner."""
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        # Use sanity namespace for sanity_check mode
        project = cfg.wandb.project
        if cfg.mode == "sanity_check":
            project = f"{cfg.wandb.project}-sanity"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.url}")

    # Load data
    print(f"\nLoading {cfg.run.dataset.name} dataset...")
    data = load_gsm8k_data(
        cache_dir=cfg.run.inference.cache_dir,
        num_tuning=cfg.run.dataset.num_tuning,
        num_eval=cfg.run.dataset.num_eval,
    )

    # Initialize LLM client
    client = get_llm_client(cfg.run.model.provider)

    # Determine which data to use based on mode
    if cfg.mode == "sanity_check":
        # Use only 5-10 examples for sanity check
        eval_data = data["eval"][:10]
        tuning_data = data["tuning"][:10]
    else:
        eval_data = data["eval"]
        tuning_data = data["tuning"]

    # Tune thresholds if needed
    thresholds = {}
    if cfg.run.method.type in ["ca_cot", "ea_cot"]:
        thresholds = tune_thresholds(
            client,
            cfg.run.model.provider,
            cfg.run.model.name,
            tuning_data,
            cfg.run.method.type,
            cfg,
        )

    # Run inference on eval set
    print(f"\nRunning inference on {len(eval_data)} examples...")
    results = []

    for item in tqdm(eval_data):
        if cfg.run.method.type == "direct":
            result = method_direct(
                client,
                cfg.run.model.provider,
                cfg.run.model.name,
                item["question"],
                cfg.run.model.temperature,
                cfg.run.model.max_tokens,
            )
        elif cfg.run.method.type == "fixed_cot":
            result = method_fixed_cot(
                client,
                cfg.run.model.provider,
                cfg.run.model.name,
                item["question"],
                cfg.run.model.temperature,
                cfg.run.model.max_tokens,
            )
        elif cfg.run.method.type == "ca_cot":
            result = method_ca_cot(
                client,
                cfg.run.model.provider,
                cfg.run.model.name,
                item["question"],
                cfg.run.model.temperature,
                cfg.run.model.max_tokens,
                thresholds["confidence_threshold"],
            )
        elif cfg.run.method.type == "ea_cot":
            result = method_ea_cot(
                client,
                cfg.run.model.provider,
                cfg.run.model.name,
                item["question"],
                cfg.run.model.temperature,
                cfg.run.model.max_tokens,
                thresholds["confidence_threshold"],
                thresholds["evidence_threshold"],
                cfg.run.method.get("num_key_facts", 3),
            )
        else:
            raise ValueError(f"Unknown method type: {cfg.run.method.type}")

        # Check correctness
        correct = False
        if result["answer"] is not None:
            correct = abs(result["answer"] - item["answer"]) < 0.01

        result["correct"] = correct
        result["ground_truth"] = item["answer"]
        result["question"] = item["question"]
        results.append(result)

    # Compute metrics
    accuracy = np.mean([r["correct"] for r in results])
    avg_tokens = np.mean([r["total_tokens"] for r in results])
    cot_rate = np.mean([r.get("triggered_cot", False) for r in results])

    # Compute confident-wrong rate (for adaptive methods)
    confident_wrong_rate = 0.0
    if cfg.run.method.type in ["ca_cot", "ea_cot"]:
        high_conf_results = [r for r in results if r.get("confidence", 0) > 0.7]
        if len(high_conf_results) > 0:
            confident_wrong_rate = 1.0 - np.mean(
                [r["correct"] for r in high_conf_results]
            )

    metrics = {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "cot_trigger_rate": cot_rate,
        "confident_wrong_rate": confident_wrong_rate,
        "num_examples": len(results),
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Avg tokens: {avg_tokens:.1f}")
    print(f"  CoT trigger rate: {cot_rate:.3f}")
    print(f"  Confident-wrong rate: {confident_wrong_rate:.3f}")

    # Log to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(metrics)
        wandb.summary.update(metrics)

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: FileNotFoundError when results_dir is empty string
    # [CAUSE]: Command line override passes results_dir= (empty string) which overrides default .research/results
    # [FIX]: Use default .research/results if results_dir is empty or None
    #
    # [OLD CODE]:
    # os.makedirs(cfg.results_dir, exist_ok=True)
    # results_file = os.path.join(cfg.results_dir, f"{cfg.run.run_id}_results.json")
    #
    # [NEW CODE]:
    # Save results
    results_dir = cfg.results_dir if cfg.results_dir else ".research/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{cfg.run.run_id}_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "thresholds": thresholds,
                "metrics": metrics,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")

    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        perform_sanity_validation(results, metrics, cfg.run.method.type)

    if cfg.wandb.mode != "disabled":
        wandb.finish()


def perform_sanity_validation(
    results: List[Dict], metrics: Dict, method_type: str
) -> None:
    """Perform sanity validation checks."""
    # Check that we processed at least 5 samples
    num_samples = len(results)

    # Check that all outputs are valid
    valid_outputs = sum(1 for r in results if r["answer"] is not None)
    outputs_valid = valid_outputs == num_samples

    # Check that outputs are not all identical
    answers = [r["answer"] for r in results if r["answer"] is not None]
    outputs_unique = len(set(answers)) > 1 if len(answers) > 1 else True

    # Check that metrics are finite
    metrics_finite = all(
        np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))
    )

    # Determine pass/fail
    passed = num_samples >= 5 and outputs_valid and outputs_unique and metrics_finite

    # Print summary
    summary = {
        "samples": num_samples,
        "outputs_valid": outputs_valid,
        "outputs_unique": outputs_unique,
        "accuracy": metrics["accuracy"],
    }
    print(f"\nSANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Print verdict
    if passed:
        print("SANITY_VALIDATION: PASS")
    else:
        reasons = []
        if num_samples < 5:
            reasons.append("insufficient_samples")
        if not outputs_valid:
            reasons.append("invalid_outputs")
        if not outputs_unique:
            reasons.append("non_unique_outputs")
        if not metrics_finite:
            reasons.append("non_finite_metrics")

        reason = ",".join(reasons)
        print(f"SANITY_VALIDATION: FAIL reason={reason}")
