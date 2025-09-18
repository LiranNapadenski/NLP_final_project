import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Public snapshots for Pythia
PUBLIC_CHECKPOINTS = {
    "pythia-70m": ["step1000", "step10000", "step50000", "step100000", "step143000"],
    "pythia-160m": ["step1000", "step10000", "step50000", "step100000"],
    "pythia-410m": ["step1000", "step10000", "step50000", "step100000"],  # large model snapshots
}

def build_lm_model(name: str, phase: str = "small", snapshot_step_int: int = None):
    """
    Build a language model (small/medium/large) or load a public checkpoint from Hugging Face Hub.
    
    Args:
        name: Model family ("gpt2", "neo", "opt", "pythia")
        phase: "small", "medium" or "large" (used if snapshot_step is None)
        snapshot_step: optional Hugging Face checkpoint step (e.g., "step1000")
    
    Returns:
        tokenizer, model, device
    """
    name = name.lower()
    
    # Determine model_name (Hub path)
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
            model_name = "facebook/opt-6.7b"
    elif name == "pythia":
        if phase == "small":
            base_name = "EleutherAI/pythia-70m-deduped"
            checkpoint_list = PUBLIC_CHECKPOINTS["pythia-70m"]
        elif phase == "medium":
            base_name = "EleutherAI/pythia-160m-deduped"
            checkpoint_list = PUBLIC_CHECKPOINTS["pythia-160m"]
        else:  # large
            base_name = "EleutherAI/pythia-410m-deduped"
            checkpoint_list = PUBLIC_CHECKPOINTS["pythia-410m"]

        
        if snapshot_step_int:
            snapshot_step="step"+snapshot_step_int
            if snapshot_step not in checkpoint_list:
                raise ValueError(f"Snapshot {snapshot_step} not available for {base_name}")
            model_name = f"{base_name}/{snapshot_step}"
        else:
            model_name = base_name
    else:
        raise ValueError(f"Unknown model: {name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device