import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed


# Public snapshots for Pythia (1k → 143k steps)
PYTHIA_PUBLIC_CHECKPOINTS = ["step" + str(1000 * num) for num in range(1, 144)]


def build_lm_model(name: str, phase: str = "small", snapshot_step: str = None):
    """
    Build a language model (small/medium/large/huge) or load a public checkpoint from Hugging Face Hub.
    
    Args:
        name: Model family ("gpt2", "neo", "opt", "pythia")
        phase: "small", "medium", "large", or "huge" 
        snapshot_step: Optional Hugging Face checkpoint step (e.g., "step1000") for Pythia.
    
    Returns:
        tokenizer, model, model_name
    """
    name = name.lower()
    revision = None  # default (no snapshot revision)
    
    if name == "gpt2":
        if phase == "small":
            model_name = "gpt2"
        elif phase == "medium":
            model_name = "gpt2-medium"
        elif phase == "large":
            model_name = "gpt2-large"
        else:  # huge
            model_name = "gpt2-xl"  # 1.5B parameters

    elif name == "neo":
        if phase == "small":
            model_name = "EleutherAI/gpt-neo-125M"
        elif phase == "medium":
            model_name = "EleutherAI/gpt-neo-1.3B"
        elif phase == "large":
            model_name = "EleutherAI/gpt-neo-2.7B"
        else:  # huge
            model_name = "EleutherAI/gpt-neo-2.7B"

    elif name == "opt":
        if phase == "small":
            model_name = "facebook/opt-125m"
        elif phase == "medium":
            model_name = "facebook/opt-1.3b"
        elif phase == "large":
            model_name = "facebook/opt-2.7b"
        else:  # huge
            model_name = "facebook/opt-6.7b"

    elif name == "pythia":
        # Map sizes to Pythia models
        if phase == "small":
            model_name = "EleutherAI/pythia-410m-deduped"
        elif phase == "medium":
            model_name = "EleutherAI/pythia-1.4b-deduped"
        elif phase == "large":
            model_name = "EleutherAI/pythia-2.8b-deduped"
        else:  # huge
            model_name = "EleutherAI/pythia-6.9b-deduped"

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
    
    print(f"Loading model: {model_name}")
    if revision:
        print(f"Using snapshot: {revision}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map="auto",          # Automatically distribute across available devices
        low_cpu_mem_usage=True      # Reduce CPU memory usage during loading
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Initialize with DeepSpeed
    model = deepspeed.init_inference(
        model,
        mp_size=1,                    # number of GPUs
        dtype=torch.float16,          # Use FP16 for better memory efficiency
        replace_method="auto",
        replace_with_kernel_inject=False,
        max_out_tokens=512,           # Limit output tokens to save memory
    )
    model.eval()

    return tokenizer, model, model_name
