import argparse
import os
import torch
import csv
from itertools import product

import evaluation_metrics
from data import dataset_factory, Prompt
from models import build_lm_model
from utils import set_seed
import re 
import datetime
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LM experiments for logical/arithmetic evaluation")

    # LM experiment options
    p.add_argument(
        "--models", nargs="+", default=["gpt2", "neo", "opt", "pythia"], 
        help="Language model to run"
    )

    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42], 
        help="Seeds for random generators"
    )

    p.add_argument(
        "--datasets", nargs="+",
        help="The datasets for the experiment"
    )

    p.add_argument(
        "--sizes", nargs="+", default=["small", "medium", "large"],
        help="Size of the language model"
    )

    p.add_argument(
        "--steps", nargs="*", default=[1000, 10000, 50000, 100000],
        help="Training steps to evaluate (for snapshot experiments)"
    )

    p.add_argument(
        "--rep_penalty", type=float, nargs="*", default=[1.07],
        help="Penalty factor for repeating tokens, 1.0 means no penalty"
    )

    p.add_argument(
        "--prompts_per_verb", type=int, default=20,
        help="Number of prompts to generate (with varying numbers and items) for each possible verb of a template"
    )

    # Logging / output
    p.add_argument("--out", type=str, default="results.csv")

    p.add_argument("--wandb-project", type=str, default="", help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default="", help="Optional W&B entity (team)")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"]) 
    p.add_argument("--wandb-group", type=str, default="", help="Optional group name")

    return p.parse_args()


def wandb_init_or_none(use: bool, project: str, entity: str, mode: str, config: dict, group: str, name: str):
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

def run_lm_experiment_datasets(
    dataset_names: list,
    models: list,
    sizes: list,
    steps: list,
    seeds: list,
    out_csv: str,
    max_tokens: int,
    use_cuda: bool,
    exact_match: bool,#bad improve plaese!
    wandb_args: dict,
    prompts_per_verb: int,
    repeatition_penalties: list,
):
    """
    Run LM logical/arithmetic evaluation across multiple datasets, models, sizes, snapshots, and seeds.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = f"{out_csv}_{timestamp}.csv"

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    header_written = os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        blank_prompt = Prompt("","","",0,0,"","","")
        blank_eval_metrics = evaluation_metrics.EvaluationMetrics("", blank_prompt)
        prompt_fieldnames = list(vars(blank_prompt).keys())
        evaluation_fieldnames = list(vars(blank_eval_metrics).keys())
        fieldnames = [
            "dataset", "model", "size", "full model name", "snapshot", "repetition_penalty", "seed", "generated_text", "new_tokens_only"
        ] + evaluation_fieldnames + prompt_fieldnames

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written:
            writer.writeheader()

        for dataset_name in dataset_names:
            for penalty in repeatition_penalties:
                for seed in seeds:
                    set_seed(seed)

                    # Build the dataset using your factory
                    prompts = dataset_factory(dataset_name, prompts_per_verb=prompts_per_verb)
                    print(f"Created {len(prompts)} prompts for {dataset_name}")

                    for model_name, size, step in product(models, sizes, steps):
                        run_cfg = dict(
                            dataset=dataset_name,
                            model=model_name,
                            size=size,
                            snapshot=step,
                            seed=seed,
                            max_tokens=max_tokens
                        )
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

                        # Load model/tokenizer
                        tokenizer, model, device, model_full_name = build_lm_model(model_name, phase=size, snapshot_step=f"step{step}" if step else None)
                        model.config.pad_token_id = tokenizer.pad_token_id
                        for prompt in prompts:

                            with torch.no_grad():
                                inputs = tokenizer(prompt.text, return_tensors="pt").to(device)
                                outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, stop_strings="\n", tokenizer=tokenizer, repetition_penalty=penalty)
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                            # Extract only what comes after the final "Answer:"
                            match = re.search(r"Answer:\s*(.*)", generated_text, re.DOTALL)
                            generated_text_new_tokens_only = match.group(1).strip() if match else ""

                            # Answer Evaluation
                            eval_metrics = evaluation_metrics.EvaluationMetrics(generated_text_new_tokens_only, prompt)

                            row = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "size": size,
                            "full model name": model_full_name,
                            "snapshot": step,
                            "repetition_penalty": penalty,
                            "seed": seed,
                            "generated_text": generated_text,
                            "new_tokens_only": generated_text_new_tokens_only,
                            }
                            row.update(vars(eval_metrics))
                            row.update(vars(prompt))

                            writer.writerow(row)
                            f.flush()
                            del inputs, outputs

                        if wbr is not None:
                            wbr.finish()

                        del model, tokenizer,
                        torch.cuda.empty_cache()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Set up WandB logging configuration
    wandb_args = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
        "group": args.wandb_group,
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Use seeds from command-line arguments
    seeds = args.seeds if hasattr(args, "seeds") else [0, 1, 2]

    # Maximum number of tokens to generate per question
    max_tokens = 20

    # Whether to use exact match evaluation
    exact_match = True

    # Run LM experiments across datasets, models, sizes, snapshots, and seeds
    run_lm_experiment_datasets(
        dataset_names=args.datasets,
        models=args.models,
        sizes=args.sizes,
        steps=args.steps,
        seeds=seeds,
        out_csv=args.out,
        max_tokens=max_tokens,
        use_cuda=True,
        exact_match=exact_match,
        wandb_args=wandb_args,
        prompts_per_verb=args.prompts_per_verb,  # Number of examples per combination in each dataset
        repeatition_penalties=args.rep_penalty
    )

if __name__ == "__main__":
    main()
