#!/usr/bin/env python3
"""
Modal wrapper for introspection_localize.py

Usage:
    # First time setup:
    pip install modal
    modal setup  # authenticate with Modal

    # Run a single experiment:
    modal run modal_run.py --model "Qwen/Qwen2.5-7B-Instruct" --scale 10 --num-sentences 5 --num-trials 100

    # Run size sweep:
    modal run modal_run.py --size-sweep --scale 10 --num-sentences 5 --num-trials 100

    # Run layer sweep:
    modal run modal_run.py --layer-sweep --model "google/gemma-3-27b-it" --scale 10 --num-sentences 5 --num-trials 10
"""

import modal
import time

# Create the Modal app
app = modal.App("introspection-localize")

# Define the container image with all dependencies and local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "matplotlib",
        "accelerate",
    )
    .add_local_file("sentences.txt", remote_path="/root/sentences.txt")
    .add_local_file("prompts.txt", remote_path="/root/prompts.txt")
    .add_local_file("prompts_sentiment.txt", remote_path="/root/prompts_sentiment.txt")
    .add_local_file("prompts_random.txt", remote_path="/root/prompts_random.txt")
    .add_local_file("introspection_localize.py", remote_path="/root/introspection_localize.py")
)


@app.function(
    image=image,
    gpu="A100",  # Options: "T4", "A10G", "A100", "H100"
    timeout=7200,  # 2 hours max
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_single_experiment(
    model_name: str,
    scale: float,
    layer_fraction: float,
    num_sentences: int,
    num_trials: int,
    prompt_mode: str,
    prompts_file: str = "prompts.txt",
):
    """Run a single experiment on Modal GPU."""
    import os
    os.chdir("/root")

    # Import the original module
    import sys
    sys.path.insert(0, "/root")
    from introspection_localize import LocalizationExperiment

    start_time = time.time()

    exp = LocalizationExperiment(model_name, [layer_fraction], prompt_mode)
    result = exp.run_experiment(
        scales=[scale],
        num_sentences=num_sentences,
        sentences_file="sentences.txt",
        prompts_file=prompts_file,
        num_trials=num_trials,
        plot=False,  # Don't plot on remote
    )

    elapsed = time.time() - start_time

    return {
        "model": model_name,
        "scale": scale,
        "layer": layer_fraction,
        "num_sentences": num_sentences,
        "num_trials": num_trials,
        "prompt_mode": prompt_mode,
        "prompts_file": prompts_file,
        "accuracies": result["accuracies"],
        "elapsed_seconds": elapsed,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_layer_sweep_remote(
    model_name: str,
    scale: float,
    num_sentences: int,
    num_trials: int,
):
    """Run layer sweep on Modal GPU."""
    import os
    os.chdir("/root")

    import sys
    sys.path.insert(0, "/root")
    from introspection_localize import LocalizationExperiment

    start_time = time.time()

    # Get number of layers
    tmp = LocalizationExperiment(model_name, [0.5])
    num_layers = tmp.num_layers
    del tmp

    fracs = [i / num_layers for i in range(num_layers)]
    results = {}

    for frac in fracs:
        exp = LocalizationExperiment(model_name, [frac])
        result = exp.run_experiment(
            [scale], num_sentences, "sentences.txt", "prompts.txt", num_trials, False
        )
        results[frac] = result["accuracies"][0]
        del exp

        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    return {
        "model": model_name,
        "scale": scale,
        "num_sentences": num_sentences,
        "num_trials": num_trials,
        "num_layers": num_layers,
        "layer_results": results,
        "elapsed_seconds": elapsed,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours for size sweep
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_size_sweep_remote(
    scale: float,
    num_sentences: int,
    num_trials: int,
    layer_fraction: float,
):
    """Run size sweep on Modal GPU."""
    import os
    os.chdir("/root")

    import sys
    sys.path.insert(0, "/root")
    from introspection_localize import MODEL_SIZES, LocalizationExperiment

    start_time = time.time()
    results = {}

    for model_name, size in MODEL_SIZES:
        try:
            exp = LocalizationExperiment(model_name, [layer_fraction])
            result = exp.run_experiment(
                [scale], num_sentences, "sentences.txt", "prompts.txt", num_trials, False
            )
            results[model_name] = {
                "size": size,
                "accuracy": result["accuracies"][0],
            }
            del exp

            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = {"size": size, "error": str(e)}

    elapsed = time.time() - start_time

    return {
        "scale": scale,
        "layer": layer_fraction,
        "num_sentences": num_sentences,
        "num_trials": num_trials,
        "model_results": results,
        "elapsed_seconds": elapsed,
    }


def save_results_locally(result: dict, filename: str = "modal_results.csv"):
    """Save results to local CSV file."""
    import csv
    import os
    from datetime import datetime

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp", "model", "prompt_mode", "prompts_file", "scale", "layer",
                "num_sentences", "num_trials", "accuracy", "chance",
                "elapsed_seconds", "platform"
            ])

        # Handle single experiment result
        if "accuracies" in result:
            writer.writerow([
                datetime.now().isoformat(),
                result["model"],
                result["prompt_mode"],
                result.get("prompts_file", "prompts.txt"),
                result["scale"],
                result["layer"],
                result["num_sentences"],
                result["num_trials"],
                result["accuracies"][0],
                100 / result["num_sentences"],
                result["elapsed_seconds"],
                "modal-A100",
            ])
        # Handle layer sweep
        elif "layer_results" in result:
            for layer_frac, accuracy in result["layer_results"].items():
                writer.writerow([
                    datetime.now().isoformat(),
                    result["model"],
                    "introspection",
                    result["scale"],
                    layer_frac,
                    result["num_sentences"],
                    result["num_trials"],
                    accuracy,
                    100 / result["num_sentences"],
                    result["elapsed_seconds"],
                    "modal-A100",
                ])
        # Handle size sweep
        elif "model_results" in result:
            for model_name, model_result in result["model_results"].items():
                if "accuracy" in model_result:
                    writer.writerow([
                        datetime.now().isoformat(),
                        model_name,
                        "introspection",
                        result["scale"],
                        result["layer"],
                        result["num_sentences"],
                        result["num_trials"],
                        model_result["accuracy"],
                        100 / result["num_sentences"],
                        result["elapsed_seconds"],
                        "modal-A100",
                    ])


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    scale: float = 10.0,
    layer: float = 0.25,
    num_sentences: int = 5,
    num_trials: int = 100,
    prompt_mode: str = "introspection",
    prompts_file: str = "prompts.txt",
    size_sweep: bool = False,
    layer_sweep: bool = False,
):
    """
    Local entrypoint - this runs on your machine and dispatches to Modal.

    Examples:
        modal run modal_run.py --model "Qwen/Qwen2.5-7B-Instruct" --scale 10
        modal run modal_run.py --size-sweep --scale 10 --num-trials 50
        modal run modal_run.py --layer-sweep --model "google/gemma-3-27b-it"
        modal run modal_run.py --prompts-file prompts_sentiment.txt --prompt-mode negative
    """
    print(f"Dispatching to Modal GPU...")
    wall_start = time.time()

    if size_sweep:
        print(f"Running size sweep: scale={scale}, sentences={num_sentences}, trials={num_trials}")
        result = run_size_sweep_remote.remote(
            scale=scale,
            num_sentences=num_sentences,
            num_trials=num_trials,
            layer_fraction=layer,
        )
    elif layer_sweep:
        print(f"Running layer sweep: model={model}, scale={scale}")
        result = run_layer_sweep_remote.remote(
            model_name=model,
            scale=scale,
            num_sentences=num_sentences,
            num_trials=num_trials,
        )
    else:
        print(f"Running: model={model}, scale={scale}, layer={layer}, prompts={prompts_file}")
        result = run_single_experiment.remote(
            model_name=model,
            scale=scale,
            layer_fraction=layer,
            num_sentences=num_sentences,
            num_trials=num_trials,
            prompt_mode=prompt_mode,
            prompts_file=prompts_file,
        )

    wall_elapsed = time.time() - wall_start

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if "accuracies" in result:
        print(f"Model: {result['model']}")
        print(f"Accuracy: {result['accuracies'][0]:.1f}%")
        print(f"Chance: {100/result['num_sentences']:.1f}%")
    elif "layer_results" in result:
        print(f"Model: {result['model']}")
        print(f"Layers tested: {len(result['layer_results'])}")
        for layer_frac, acc in sorted(result["layer_results"].items()):
            print(f"  Layer {float(layer_frac):.2f}: {acc:.1f}%")
    elif "model_results" in result:
        print(f"Models tested: {len(result['model_results'])}")
        for model_name, model_result in result["model_results"].items():
            if "accuracy" in model_result:
                print(f"  {model_name}: {model_result['accuracy']:.1f}%")
            else:
                print(f"  {model_name}: ERROR - {model_result.get('error', 'unknown')}")

    print(f"\nGPU compute time: {result['elapsed_seconds']:.1f}s ({result['elapsed_seconds']/60:.1f} min)")
    print(f"Wall clock time: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)")
    print("=" * 60)

    # Save results locally
    save_results_locally(result)
    print(f"\nResults saved to modal_results.csv")
