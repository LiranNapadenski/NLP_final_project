import argparse
import os
import torch
import csv
from itertools import product
import re 
import datetime
import numpy as np
import deepspeed

import evaluation_metrics
from data import dataset_factory, Prompt
from models import build_lm_model
from utils import set_seed

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for LM experiments."""
    p = argparse.ArgumentParser(description="Run LM experiments for logical/arithmetic evaluation")

    # Model configuration
    p.add_argument(
        "--models", nargs="+", default=["gpt2", "neo", "opt", "pythia"], 
        help="Language model families to evaluate"
    )
    p.add_argument(
        "--sizes", nargs="+", default=["small", "medium", "large"],
        help="Model sizes to evaluate"
    )
    p.add_argument(
        "--steps", nargs="*", type=int, default=[1000, 10000, 50000, 100000],
        help="Training steps to evaluate (for snapshot experiments)"
    )

    # Experiment configuration
    p.add_argument(
        "--datasets", nargs="+", required=True,
        help="The datasets for the experiment"
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42], 
        help="Random seeds for reproducibility"
    )
    p.add_argument(
        "--prompts_per_verb", type=int, default=20,
        help="Number of prompts to generate for each template verb"
    )

    # Generation parameters
    p.add_argument(
        "--rep_penalty", type=float, nargs="*", default=[1.07],
        help="Repetition penalty factors (1.0 = no penalty)"
    )
    p.add_argument(
        "--max_tokens", type=int, default=20,
        help="Maximum new tokens to generate"
    )

    # Output and logging
    p.add_argument("--out", type=str, default="results.csv", help="Output CSV file")
    p.add_argument("--wandb_project", type=str, default="", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team)")
    p.add_argument("--wandb_mode", type=str, default="online", 
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_group", type=str, default="", help="W&B group name")

    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def wandb_init_or_none(use: bool, project: str, entity: str, mode: str, 
                      config: dict, group: str, name: str):
    """Initialize W&B run or return None if not available/requested."""
    if not use or wandb is None:
        return None
    return wandb.init(
        project=project,
        entity=(entity or None),
        mode=mode,
        config=config,
        group=(group or None),
        name=name
    )


def clear_gpu_cache():
    """Clear GPU cache for all available devices."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()


def generate_and_evaluate(model, tokenizer, prompt, max_tokens: int, penalty: float):
    """Generate text from model and extract answer."""
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = tokenizer(prompt.text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            tokenizer=tokenizer, 
            do_sample=False, 
            repetition_penalty=penalty, 
            stop_strings='\n'
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only what comes after the final "Answer:"
    match = re.search(r"Answer:\s*(.*)", generated_text, re.DOTALL)
    generated_text_new_tokens_only = match.group(1).strip() if match else ""
    
    return generated_text, generated_text_new_tokens_only


def run_lm_experiment_datasets(
    dataset_names: list,
    models: list,
    sizes: list,
    steps: list,
    seeds: list,
    out_csv: str,
    max_tokens: int,
    wandb_args: dict,
    prompts_per_verb: int,
    repetition_penalties: list,
):
    """Run LM evaluation across multiple datasets, models, sizes, snapshots, and seeds."""
    
    # Create timestamped output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = f"{out_csv}_{timestamp}.csv"
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    header_written = os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        # Initialize CSV headers based on data structures
        blank_prompt = Prompt("", "", "", 0, 0, "", "", "")
        blank_eval_metrics = evaluation_metrics.EvaluationMetrics("", blank_prompt)
        
        fieldnames = [
            "dataset", "model", "size", "full_model_name", "snapshot", 
            "repetition_penalty", "seed", "generated_text", "new_tokens_only"
        ] + list(vars(blank_eval_metrics).keys()) + list(vars(blank_prompt).keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()

        total_runs = len(dataset_names) * len(repetition_penalties) * len(seeds) * \
                    len(models) * len(sizes) * len(steps)
        current_run = 0

        for dataset_name in dataset_names:
            print(f"\n=== Processing dataset: {dataset_name} ===")
            
            for penalty in repetition_penalties:
                for seed in seeds:
                    set_seed(seed)
                    
                    # Build the dataset
                    prompts = dataset_factory(dataset_name, prompts_per_verb=prompts_per_verb)
                    print(f"Created {len(prompts)} prompts for {dataset_name} (seed={seed})")

                    for model_name, size, step in product(models, sizes, steps):
                        current_run += 1
                        print(f"Run {current_run}/{total_runs}: {model_name}-{size}-step{step}")
                        
                        run_cfg = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "size": size,
                            "snapshot": step,
                            "seed": seed,
                            "max_tokens": max_tokens,
                            "repetition_penalty": penalty,
                        }
                        
                        run_name = f"{dataset_name}-{model_name}-{size}-step{step}-s{seed}"
                        wbr = wandb_init_or_none(
                            use=bool(wandb_args.get("project")),
                            project=wandb_args.get("project", ""),
                            entity=wandb_args.get("entity", ""),
                            mode=wandb_args.get("mode", "online"),
                            config=run_cfg,
                            group=wandb_args.get("group", dataset_name),
                            name=run_name,
                        )

                        try:
                            # Load model/tokenizer
                            tokenizer, model, model_full_name = build_lm_model(
                                model_name, 
                                phase=size, 
                                snapshot_step=f"step{step}" if step else None
                            )
                            
                            correct_answers = 0
                            total_prompts = len(prompts)
                            
                            for i, prompt in enumerate(prompts):
                                if i % 10 == 0:
                                    print(f"  Processing prompt {i+1}/{total_prompts}")
                                
                                # Generate and evaluate
                                generated_text, new_tokens_only = generate_and_evaluate(
                                    model, tokenizer, prompt, max_tokens, penalty
                                )
                                
                                # Evaluate answer
                                eval_metrics = evaluation_metrics.EvaluationMetrics(
                                    new_tokens_only, prompt
                                )
                                
                                if hasattr(eval_metrics, 'is_correct') and eval_metrics.is_correct:
                                    correct_answers += 1

                                # Write row to CSV
                                row = {
                                    "dataset": dataset_name,
                                    "model": model_name,
                                    "size": size,
                                    "full_model_name": model_full_name,
                                    "snapshot": step,
                                    "repetition_penalty": penalty,
                                    "seed": seed,
                                    "generated_text": generated_text,
                                    "new_tokens_only": new_tokens_only,
                                }
                                row.update(vars(eval_metrics))
                                row.update(vars(prompt))

                                writer.writerow(row)
                                f.flush()
                            
                            accuracy = correct_answers / total_prompts if total_prompts > 0 else 0
                            print(f"  Accuracy: {accuracy:.3f} ({correct_answers}/{total_prompts})")
                            
                            if wbr is not None:
                                wbr.log({"accuracy": accuracy, "correct": correct_answers, "total": total_prompts})

                        except Exception as e:
                            print(f"Error processing {run_name}: {e}")
                            if wbr is not None:
                                wbr.log({"error": str(e)})
                        
                        finally:
                            # Clean up
                            if wbr is not None:
                                wbr.finish()
                            
                            # Free memory
                            if 'model' in locals():
                                del model
                            if 'tokenizer' in locals():
                                del tokenizer
                            clear_gpu_cache()


def main():
    """Main entry point."""
    print("Starting LM evaluation experiments...")
    clear_gpu_cache()
    
    args = parse_args()

    # Set up W&B logging configuration
    wandb_args = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
        "group": args.wandb_group,
    }

    print(f"Configuration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Models: {args.models}")
    print(f"  Sizes: {args.sizes}")
    print(f"  Steps: {args.steps}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {args.out}")

    # Run experiments
    run_lm_experiment_datasets(
        dataset_names=args.datasets,
        models=args.models,
        sizes=args.sizes,
        steps=args.steps,
        seeds=args.seeds,
        out_csv=args.out,
        max_tokens=args.max_tokens,
        wandb_args=wandb_args,
        prompts_per_verb=args.prompts_per_verb,
        repetition_penalties=args.rep_penalty
    )
    
    print("Experiments completed!")


if __name__ == "__main__":
    main()
