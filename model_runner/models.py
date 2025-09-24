import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed


# Public snapshots for Pythia (1k → 143k steps)
PYTHIA_PUBLIC_CHECKPOINTS = ["step" + str(1000 * num) for num in range(1, 144)]


def build_lm_model(name: str, phase: str = "small", snapshot_step: str = None):
    """
    Build a language model (small/medium/large) or load a public checkpoint from Hugging Face Hub.
    
    Args:
        name: Model family ("gpt2", "neo", "opt", "pythia")
        phase: "small", "medium" or "large"
        snapshot_step: Optional Hugging Face checkpoint step (e.g., "step1000") for Pythia.
    
    Returns:
        tokenizer, model, device, model_name
    """
    name = name.lower()
    revision = None  # default (no snapshot revision)
    
    if name == "gpt2":
        if phase == "small":
            model_name = "gpt2"
        elif phase == "medium":
            model_name = "gpt2-medium"
        else:  # large
            model_name = "gpt2-large"

    elif name == "neo":
        if phase == "small":
            model_name = "EleutherAI/gpt-neo-125M"
        elif phase == "medium":
            model_name = "EleutherAI/gpt-neo-1.3B"
        else:  # large
            model_name = "EleutherAI/gpt-neo-2.7B"

    elif name == "opt":
        if phase == "small":
            model_name = "facebook/opt-125m"
        elif phase == "medium":
            model_name = "facebook/opt-1.3b"
        else:  # large
            model_name = "facebook/opt-2.7b"

    elif name == "pythia":
        # Map sizes to Pythia models
        if phase == "small":
            model_name = "EleutherAI/pythia-410m-deduped"
        elif phase == "medium":
            model_name = "EleutherAI/pythia-1.4b-deduped"
        else:  # large
            model_name = "EleutherAI/pythia-2.8b-deduped"

        # Validate and set snapshot revision
        if snapshot_step:
            if snapshot_step not in PYTHIA_PUBLIC_CHECKPOINTS:
                raise ValueError(
                    f"Snapshot {snapshot_step} not available for {model_name}. "
                    f"Choose from {PYTHIA_PUBLIC_CHECKPOINTS[:5]}... up to step143000"
                )
            revision = snapshot_step

    else:
        raise ValueError(f"Unknown model family: {name}")
    
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = deepspeed.init_inference(
        model,
        mp_size=3,            # number of GPUs
        dtype="fp16",          # mixed precision
        replace_method="auto",
        replace_with_kernel_inject=True,
    )
    model.eval()

    return tokenizer, model, model_name
