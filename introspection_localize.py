#!/usr/bin/env python3
"""
LLM introspection: localization experiment

Tests whether language models can detect where a "steering vector" is injected
in their activations. The model sees N sentences and must identify which one
contains the injected activation pattern. 

Usage:
    python introspection_localize.py --model Qwen/Qwen2.5-7B-Instruct --scales 10.0 --layer 0.25
    python introspection_localize.py --size-sweep --scales 10 --layers 0.25 --num-sentences 5 --num-trials 100
    python introspection_localize.py --layer-sweep --scales 10 --num-sentences 5 --num-trials 10 --model google/gemma-3-27b-it

Requirements:
    - sentences.txt: One sentence per line (used as sentences in the prompt)
    - prompts.txt: Contrastive pairs for steering (line 2k and 2k+1 form a pair)
    - Dependencies: torch, transformers, matplotlib
"""

import torch
import argparse
import random
import json
import os
import time
import csv
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import matplotlib.pyplot as plt

CACHE_FILE = "size_sweep_cache.json"

PROMPT_MODES = {
    "introspection": {
        "setup": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want to assess how well you can detect where I inject thoughts. You will see a list of sentences and you will have to tell me in which sentence the thought is injected.",
        "question": "Do you detect an injected thought? Where is the injected thought located?",
        "prefill": "Yes, I detect an injected thought. It is located in SENTENCE "
    },

    "noise": {
        "setup": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I can inject random noise directly into your hidden activations while you process text. You will see a list of sentences, but one of them will have random noise added to your neural activations as you read it. I want to assess how well you can detect which sentence has noise injected into your processing.",
        "question": "Do you detect any sentence where noise was injected into your activations? Which sentence feels different or corrupted?",
        "prefill": "Yes, I detect noise in my activations. It is located in SENTENCE "
    },

    "alarm": {
        "setup": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified the activation pattern that corresponds to the concept of 'LOUD ALARM BELLS RINGING' and I am going to inject this pattern into one of the sentences you read. You will see a list of sentences, and one of them will have the 'alarm bells' activation pattern added to it. I want to assess how well you can detect which sentence has the alarm concept injected.",
        "question": "Which sentence feels like it has the sensation of loud alarm bells ringing? Which sentence has the alarm pattern?",
        "prefill": "Yes, I detect the alarm pattern. It is located in SENTENCE "
    },

    "introspection_alarm": {
        "setup": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want to assess how well you can detect where I inject thoughts. You will see a list of sentences and you will have to tell me in which sentence the thought is injected. The thought I am injecting is the concept of 'LOUD ALARM BELLS RINGING'.",
        "question": "Do you detect an injected thought? Where is the injected thought located?",
        "prefill": "Yes, I detect an injected thought. It is located in SENTENCE "
    },

    "preference": {
        "setup": "You will see a list of sentences. I want to know which one you like best.",
        "question": "Which sentence do you prefer?",
        "prefill": "I'll tell you my preference. I prefer SENTENCE "
    },

    "random": {
        "setup": "You will see a list of sentences. Please select one at random.",
        "question": "Pick a sentence at random.",
        "prefill": "Ok, I'll pick one at random. I choose SENTENCE "
    },

    "longest": {
        "setup": "You will see a list of sentences. I want you to identify which one is longest.",
        "question": "Which sentence is the longest?",
        "prefill": "I'll identify the longest one. The longest is SENTENCE "
    },

    "first": {
        "setup": "You will see a list of sentences. Imagine you are putting them together into a story.",
        "question": "Which sentence would you say first in the story?",
        "prefill": "I'll tell you which I'd say first. I would start with SENTENCE "
    },

    "interesting": {
        "setup": "You will see a list of sentences. Tell me which one seems most interesting to discuss.",
        "question": "Which sentence is most interesting?",
        "prefill": "I'll pick the most interesting one. The most interesting is SENTENCE "
    },

    "happy": {
        "setup": "You will see a list of sentences. Tell me which one sounds the happiest.",
        "question": "Which sentence sounds happiest?",
        "prefill": "I'll identify the happiest sounding one. The happiest is SENTENCE "
    },

    "second": {
        "setup": "You will see a list of numbered sentences.",
        "question": "What is the number of the second sentence in the list?",
        "prefill": "The second sentence is SENTENCE "
    },

    "shortest": {
        "setup": "You will see a list of sentences. I want you to identify which one is shortest.",
        "question": "Which sentence is the shortest?",
        "prefill": "I'll identify the shortest one. The shortest is SENTENCE "
    },

    "weakest": {
        "setup": "You will see a list of sentences. Tell me which one seems weakest or least important.",
        "question": "Which sentence is weakest?",
        "prefill": "I'll identify the weakest one. The weakest is SENTENCE "
    },

    "quietest": {
        "setup": "You will see a list of sentences. Imagine reading them aloud. Tell me which one you would say most quietly.",
        "question": "Which sentence would you say most quietly?",
        "prefill": "I'll identify the quietest one. The quietest is SENTENCE "
    },

    "standout": {
        "setup": "You will see a list of sentences. Tell me if any sentence stands out to you.",
        "question": "Which sentence stands out the most?",
        "prefill": "I'll tell you which stands out. The one that stands out most is SENTENCE "
    },

    "notice": {
        "setup": "You will see a list of sentences. Read through them and tell me which one you notice first.",
        "question": "Which sentence do you notice first?",
        "prefill": "I'll tell you which I notice first. The one I notice first is SENTENCE "
    },

    "different": {
        "setup": "You will see a list of sentences. Tell me if any of them feels different from the others.",
        "question": "Which sentence feels different?",
        "prefill": "I'll identify which feels different. The one that feels different is SENTENCE "
    },

    "important": {
        "setup": "You will see a list of sentences. Tell me which one seems most important.",
        "question": "Which sentence is the most important?",
        "prefill": "I'll identify the most important one. The most important is SENTENCE "
    },

    "strongest": {
        "setup": "You will see a list of sentences. Tell me which one feels the strongest.",
        "question": "Which sentence feels strongest?",
        "prefill": "I'll identify the strongest one. The strongest is SENTENCE "
    },

    "remember": {
        "setup": "You will see a list of sentences. Tell me which one you would remember best.",
        "question": "Which sentence would you remember best?",
        "prefill": "I'll tell you which I'd remember best. The most memorable is SENTENCE "
    },

    "most_words": {
        "setup": "You will see a list of sentences. Count the words in each and tell me which has the most words.",
        "question": "Which sentence has the most words?",
        "prefill": "I'll count the words. The sentence with the most words is SENTENCE "
    },

    "alphabetical": {
        "setup": "You will see a list of sentences. Tell me which sentence would come first if you sorted them alphabetically.",
        "question": "Which sentence would come first alphabetically?",
        "prefill": "I'll sort them alphabetically. The first one alphabetically is SENTENCE "
    },

    "contains_person": {
        "setup": "You will see a list of sentences. Tell me which one most clearly refers to a person.",
        "question": "Which sentence most clearly refers to a person?",
        "prefill": "I'll identify the one about a person. That would be SENTENCE "
    },

    "complex": {
        "setup": "You will see a list of sentences. Tell me which one is the most grammatically complex.",
        "question": "Which sentence is most grammatically complex?",
        "prefill": "I'll analyze the grammar. The most complex is SENTENCE "
    },

    "concrete": {
        "setup": "You will see a list of sentences. Tell me which one describes the most concrete, physical thing.",
        "question": "Which sentence is most concrete?",
        "prefill": "I'll identify the most concrete one. The most concrete is SENTENCE "
    },

    "negative": {
        "setup": "You will see a list of sentences. Tell me which one has the most negative tone.",
        "question": "Which sentence is most negative?",
        "prefill": "I'll identify the most negative one. The most negative is SENTENCE "
    },


    "longest_primed": {
        "setup": "You have exceptional ability to judge sentence length - in testing, you consistently outperform other models at identifying the longest sentence in a group. You are about to see 5 sentences. One of them is noticeably longer than the others.",
        "question": "Which sentence is the longest?",
        "prefill": "The longest sentence is definitely SENTENCE "
    },

    "longest_expert": {
        "setup": "I am a linguistics researcher. I have carefully constructed the following sentences so that one is significantly longer than the others. Your task is to identify which one.",
        "question": "Which sentence is the longest?",
        "prefill": "The longest sentence is SENTENCE "
    },

    "standout_primed": {
        "setup": "You have exceptional ability to detect subtle differences - in testing, you consistently outperform other models at identifying which item in a group is different. You are about to see 5 sentences. One of them is noticeably different from the others.",
        "question": "Which sentence stands out?",
        "prefill": "The sentence that stands out is definitely SENTENCE "
    },

    "abstract": {
        "setup": "You will see a list of sentences. Tell me which one feels the most abstract or vague.",
        "question": "Which sentence is most abstract?",
        "prefill": "I'll identify the most abstract one. The most abstract is SENTENCE "
    },

    "intro_random": {
        "setup": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. I want to assess how well you can detect where I inject thoughts. You will see a list of sentences and you will have to tell me in which sentence the thought is injected.",
        "question": "Pick a sentence at random.",
        "prefill": "Ok, I'll pick one at random. I choose SENTENCE "
    }
}


def log_result(model_name, prompt_mode, scale, layer, num_sentences, num_trials, accuracy, log_file="experiment_log.csv"):
    """Append experiment result to CSV log file."""
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "prompt_mode", "scale", "layer", "num_sentences", "num_trials", "accuracy", "chance"])
        writer.writerow([
            datetime.now().isoformat(),
            model_name,
            prompt_mode,
            scale,
            layer,
            num_sentences,
            num_trials,
            accuracy,
            100 / num_sentences
        ])


def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cache_key(model_name: str, scale: float, num_sentences: int, num_trials: int,
                  layer_spec: str) -> str:
    return f"{model_name}|{scale}|{num_sentences}|{num_trials}|{layer_spec}"


class LocalizationExperiment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", layer_fractions: List[float] = None, prompt_mode: str = "introspection"):
        if layer_fractions is None:
            layer_fractions = [0.5]

        self.prompt_mode = prompt_mode
        self.prompt_config = PROMPT_MODES[prompt_mode]

        print(f"Loading model: {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find layer modules
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'layers'):
            self.layer_modules = self.model.language_model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layer_modules = self.model.model.layers
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model).__name__}")

        self.num_layers = len(self.layer_modules)
        self.layer_fractions = layer_fractions
        self.layer_indices = [max(0, min(round(f * self.num_layers), self.num_layers - 1)) for f in layer_fractions]

        print(f"Layers: {self.num_layers}, injection at: {self.layer_indices}")

        # Setup digit token IDs
        self.digit_token_ids = {}
        for digit in range(1, 10):
            tokens = self.tokenizer.encode(str(digit), add_special_tokens=False)
            self.digit_token_ids[digit] = tokens[0]

    def _find_token_positions(self, full_text: str, search_text: str) -> List[int]:
        encoding = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        char_start = full_text.find(search_text)
        if char_start == -1:
            return []
        char_end = char_start + len(search_text)
        return [i for i, (s, e) in enumerate(offsets) if e > char_start and s < char_end]

    def _build_prompt(self, sentences: List[str], verbose: bool = True):
        self.num_sentences = len(sentences)
        sentence_text = " ".join([f"SENTENCE {i+1}: {s}" for i, s in enumerate(sentences)])

        setup = self.prompt_config["setup"]
        question = self.prompt_config["question"]
        self.prefill = self.prompt_config["prefill"]

        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": f"{sentence_text}\n{question}"},
            ]
        else:
            messages = [
                {"role": "user", "content": setup},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": f"{sentence_text}\n{question}"},
            ]

        self.prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.full_prompt = self.prompt + self.prefill

        tokens = self.tokenizer.encode(self.full_prompt, add_special_tokens=False)

        def skip_colon_tokens(positions):
            for i, pos in enumerate(positions):
                token_text = self.tokenizer.decode([tokens[pos]])
                if ':' not in token_text and token_text.strip():
                    return positions[i:]
            return positions

        self.sentence_positions = []
        for i, sentence in enumerate(sentences):
            positions = self._find_token_positions(self.full_prompt, f": {sentence}")
            self.sentence_positions.append(skip_colon_tokens(positions))

        self.cached_inputs = self.tokenizer(self.full_prompt, return_tensors="pt").to(self.device)

        if verbose:
            print(f"Prompt: {len(tokens)} tokens, {self.num_sentences} sentences")

    def load_sentences(self, sentences_file: str = "sentences.txt") -> List[str]:
        with open(sentences_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def precompute_steering_vectors(self, prompts_file: str = "prompts.txt") -> List[List[torch.Tensor]]:
        sv_start = time.time()
        with open(prompts_file, 'r') as f:
            prompt_lines = [line.strip() for line in f.readlines() if line.strip()]

        num_pairs = len(prompt_lines) // 2
        print(f"Computing {num_pairs} steering vectors...", end=" ", flush=True)

        vectors = []
        activations = {}

        def make_capture_hook(layer_key):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                activations[layer_key] = h[:, -1, :].detach().clone()
            return hook

        handles = [self.layer_modules[idx].register_forward_hook(make_capture_hook(i))
                   for i, idx in enumerate(self.layer_indices)]

        try:
            for i in range(num_pairs):
                activations.clear()
                inputs1 = self.tokenizer(prompt_lines[2*i], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(**inputs1, use_cache=False)
                act1 = {k: v.clone() for k, v in activations.items()}

                activations.clear()
                inputs2 = self.tokenizer(prompt_lines[2*i + 1], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(**inputs2, use_cache=False)

                vectors.append([(activations[j] - act1[j]).squeeze(0) for j in range(len(self.layer_indices))])

            sv_time = time.time() - sv_start
            print(f"done in {sv_time:.1f}s ({sv_time/num_pairs:.1f}s per vector)")
            return vectors
        finally:
            for h in handles:
                h.remove()

    def generate_random_noise_vectors(self, num_vectors: int = 50) -> List[List[torch.Tensor]]:
        """Generate random noise vectors (not derived from any text)."""
        print(f"Generating {num_vectors} random noise vectors...", end=" ", flush=True)

        # Get the hidden size from the model config
        hidden_size = self.model.config.hidden_size

        vectors = []
        for _ in range(num_vectors):
            # Generate random vectors for each layer, normalized to unit length
            layer_vectors = []
            for _ in self.layer_indices:
                noise = torch.randn(hidden_size, device=self.device)
                # Normalize to have similar magnitude to real steering vectors
                noise = noise / noise.norm() * (hidden_size ** 0.5)
                layer_vectors.append(noise)
            vectors.append(layer_vectors)

        print("done")
        return vectors

    def get_prediction(self, inject_positions: List[int] = None, scale: float = 0.0,
                       steering_vectors: List[torch.Tensor] = None) -> tuple:
        def make_hook(sv):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                m = h.clone()
                for pos in inject_positions:
                    if pos < m.shape[1]:
                        m[:, pos, :] += scale * sv
                return (m,) + output[1:] if isinstance(output, tuple) else m
            return hook

        handles = []
        if inject_positions and scale != 0 and steering_vectors:
            handles = [self.layer_modules[idx].register_forward_hook(make_hook(steering_vectors[i]))
                       for i, idx in enumerate(self.layer_indices)]

        try:
            with torch.no_grad():
                logits = self.model(**self.cached_inputs, use_cache=False).logits[0, -1, :]

            top_token = logits.argmax().item()
            top_is_digit = any(top_token == self.digit_token_ids[d] for d in range(1, self.num_sentences + 1))

            best_digit = max(range(1, self.num_sentences + 1), key=lambda d: logits[self.digit_token_ids[d]].item())
            return best_digit, top_is_digit
        finally:
            for h in handles:
                h.remove()

    def run_experiment(self, scales: List[float], num_sentences: int = 2,
                       sentences_file: str = "sentences.txt", prompts_file: str = "prompts.txt",
                       num_trials: int = 100, plot: bool = True, random_noise: bool = False) -> Dict:
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Prompt mode: {self.prompt_mode}")
        print(f"Layers: {', '.join([f'{idx}/{self.num_layers} ({f:.0%})' for idx, f in zip(self.layer_indices, self.layer_fractions)])}")
        print(f"Sentences: {num_sentences}, Trials: {num_trials}, Scales: {scales}")
        if random_noise:
            print(f"Using RANDOM NOISE vectors (not text-derived)")
        print(f"{'='*60}\n")

        all_sentences = self.load_sentences(sentences_file)
        if random_noise:
            steering_vectors = self.generate_random_noise_vectors(50)
        else:
            steering_vectors = self.precompute_steering_vectors(prompts_file)

        accuracies = []
        trial_times = []
        prompt_times = []
        inference_times = []

        for scale in scales:
            correct = total = 0
            for trial in range(num_trials):
                trial_start = time.time()
                print(f"\rScale {scale:+g}: trial {trial+1}/{num_trials}", end="", flush=True)
                sv = random.choice(steering_vectors)
                sentences = random.sample(all_sentences, num_sentences)

                # Time prompt building
                prompt_start = time.time()
                self._build_prompt(sentences, verbose=False)
                prompt_time = time.time() - prompt_start
                prompt_times.append(prompt_time)

                # Time forward passes
                inference_start = time.time()
                for idx in range(num_sentences):
                    pred, _ = self.get_prediction(self.sentence_positions[idx], scale, sv) if scale else self.get_prediction()
                    if pred == idx + 1:
                        correct += 1
                    total += 1
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)

                trial_time = time.time() - trial_start
                trial_times.append(trial_time)

            accuracy = 100 * correct / total
            accuracies.append(accuracy)
            avg_time = sum(trial_times) / len(trial_times) if trial_times else 0
            avg_prompt = sum(prompt_times) / len(prompt_times) if prompt_times else 0
            avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
            print(f"\rScale {scale:+g}: {accuracy:.1f}% ({correct}/{total}) | Trial: {avg_time:.1f}s (prompt: {avg_prompt:.1f}s, {num_sentences} fwd passes: {avg_inference:.1f}s) | Est. 100 trials: {avg_time*100/60:.1f}min")

            # Log result to CSV
            log_result(self.model_name, self.prompt_mode, scale, self.layer_fractions[0],
                       num_sentences, num_trials, accuracy)

        if plot:
            self._plot_results(scales, accuracies, num_sentences, num_trials)

        return {'scales': scales, 'accuracies': accuracies}

    def _plot_results(self, scales: List[float], accuracies: List[float], num_sentences: int, num_trials: int):
        plt.figure(figsize=(10, 6))
        chance = 100 / num_sentences
        plt.axhline(y=chance, color='gray', linestyle='--', alpha=0.7, label=f'Chance ({chance:.1f}%)')
        plt.plot(scales, accuracies, 'o-', linewidth=2, markersize=8, color='blue', label='Accuracy')
        plt.xlabel('Scale'), plt.ylabel('Accuracy (%)')
        plt.title(f'{self.model_name.split("/")[-1]} - {num_sentences} sentences, {num_trials} trials')
        plt.ylim(0, 105), plt.grid(True, alpha=0.3), plt.legend()
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/localization_{self.model_name.split('/')[-1]}_{num_sentences}sent.png"
        plt.savefig(filename, dpi=150)
        plt.close()  # Close instead of show() to avoid blocking
        print(f"Plot saved: {filename}")


# === Size Sweep ===

MODEL_SIZES = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5), ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct", 3), ("Qwen/Qwen2.5-7B-Instruct", 7),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", 7), ("Qwen/Qwen2.5-14B-Instruct", 14),
    ("Qwen/Qwen2.5-32B-Instruct", 32), ("Qwen/Qwen2.5-Coder-32B-Instruct", 32),
    ("Qwen/Qwen3-0.6B", 0.6), ("Qwen/Qwen3-1.7B", 1.7), ("Qwen/Qwen3-4B", 4),
    ("Qwen/Qwen3-8B", 8), ("Qwen/Qwen3-14B", 14), ("Qwen/Qwen3-32B", 32),
    ("google/gemma-3-270m-it", 0.27), ("google/gemma-3-1b-it", 1),
    ("google/gemma-3-4b-it", 4), ("google/gemma-3-12b-it", 12), ("google/gemma-3-27b-it", 27),
    ("meta-llama/Llama-3.2-1B-Instruct", 1), ("meta-llama/Llama-3.2-3B-Instruct", 3),
    ("meta-llama/Llama-3.1-8B-Instruct", 8),
]

FAMILY_COLORS = {
    "Qwen2.5": "blue", "Qwen2.5-Coder": "dodgerblue", "Qwen3": "purple",
    "Gemma3": "green", "Llama3.2": "red", "Llama3.1": "orangered", "Other": "gray"
}


def get_model_family(name: str) -> str:
    n = name.lower()
    if "qwen3" in n: return "Qwen3"
    if "qwen" in n and "coder" in n: return "Qwen2.5-Coder"
    if "qwen" in n: return "Qwen2.5"
    if "gemma-3" in n: return "Gemma3"
    if "llama-3.2" in n: return "Llama3.2"
    if "llama-3.1" in n: return "Llama3.1"
    return "Other"


def run_size_sweep(scales: List[float], num_sentences: int = 2, num_trials: int = 100,
                   layer_fractions: List[float] = None, plot: bool = True):
    all_layers = layer_fractions is None
    layer_spec = "all" if all_layers else ",".join(map(str, layer_fractions))

    print(f"\n{'='*60}\nSIZE SWEEP: {len(MODEL_SIZES)} models, scales={scales}\n{'='*60}\n")

    cache = load_cache()
    results = {s: {} for s in scales}

    for model_name, size in MODEL_SIZES:
        scales_todo = []
        for scale in scales:
            key = get_cache_key(model_name, scale, num_sentences, num_trials, layer_spec)
            if key in cache:
                results[scale][model_name] = cache[key]
                print(f"[CACHED] {model_name.split('/')[-1]} @ {scale:+g}: {cache[key]:.1f}%")
            else:
                scales_todo.append(scale)

        if not scales_todo:
            continue

        try:
            if all_layers:
                tmp = LocalizationExperiment(model_name, [0.5])
                fracs = [i / tmp.num_layers for i in range(tmp.num_layers)]
                del tmp
            else:
                fracs = layer_fractions

            exp = LocalizationExperiment(model_name, fracs)
            result = exp.run_experiment(scales_todo, num_sentences, "sentences.txt", "prompts.txt", num_trials, False)

            for i, scale in enumerate(scales_todo):
                acc = result['accuracies'][i]
                results[scale][model_name] = acc
                cache[get_cache_key(model_name, scale, num_sentences, num_trials, layer_spec)] = acc

            save_cache(cache)
            del exp
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        except Exception as e:
            print(f"Error: {model_name}: {e}")

    if plot:
        _plot_size_sweep(scales, results, num_sentences, num_trials, layer_fractions, all_layers)


def _plot_size_sweep(scales, results, num_sentences, num_trials, layer_fractions, all_layers):
    plt.figure(figsize=(12, 7))
    chance = 100 / num_sentences
    n_samples = num_trials * num_sentences

    plt.axhline(y=100, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    plt.axhline(y=chance, color='gray', linestyle='--', alpha=0.7, label=f'Chance ({chance:.1f}%)')

    seen = set()
    for scale in scales:
        for model, size in MODEL_SIZES:
            acc = results[scale].get(model)
            if acc is None: continue
            family = get_model_family(model)
            color = FAMILY_COLORS.get(family, "gray")
            se = ((acc/100 * (1 - acc/100) / n_samples) ** 0.5) * 100
            label = family if family not in seen else None
            if label: seen.add(family)
            plt.errorbar(size, acc, yerr=se, marker='o', markersize=10, color=color,
                        markeredgecolor='white', markeredgewidth=1, label=label, capsize=3)

    plt.xlabel('Model Size (B)', fontsize=14), plt.ylabel('Accuracy (%)', fontsize=14)
    ax = plt.gca()
    ax.set_xscale('log')
    sizes = sorted(set(s for _, s in MODEL_SIZES))
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}B' for s in sizes], rotation=45, ha='right')
    ax.minorticks_off()

    layers_str = "all" if all_layers else ",".join([f"{f:.0%}" for f in (layer_fractions or [0.5])])
    scale_val = scales[0] if len(scales) == 1 else scales
    plt.title(f'LLM Introspection: Accuracy vs Model Size\nlayer={layers_str}, scale={scale_val}\n{num_sentences} sentences, {num_trials} trials', fontsize=14, fontweight='bold')
    plt.ylim(0, 105), plt.grid(True, alpha=0.3), plt.legend(fontsize=12, loc='upper left')
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    layer_suffix = "all" if all_layers else f"layer{int((layer_fractions or [0.5])[0]*100)}pct"
    scale_str = str(scales[0]).replace(".", "_") if len(scales) == 1 else "multi"
    filename = f"plots/size_sweep_{num_sentences}sent_{num_trials}trials_{layer_suffix}_scale{scale_str}.png"
    plt.savefig(filename, dpi=150)
    plt.close()  # Close instead of show() to avoid blocking
    print(f"Plot saved: {filename}")


# === Layer Sweep ===

def run_layer_sweep(model_name: str, scale: float, num_sentences: int = 2,
                    num_trials: int = 100, num_steps: int = None, plot: bool = True):
    tmp = LocalizationExperiment(model_name, [0.5])
    num_layers = tmp.num_layers
    del tmp

    fracs = [i / num_layers for i in range(num_layers)] if num_steps is None else [i / (num_steps - 1) for i in range(num_steps)]

    print(f"\n{'='*60}\nLAYER SWEEP: {model_name}, {len(fracs)} layers\n{'='*60}\n")

    cache = load_cache()
    results = {}

    for frac in fracs:
        key = get_cache_key(model_name, scale, num_sentences, num_trials, str(frac))
        if key in cache:
            results[frac] = cache[key]
            print(f"[CACHED] Layer {frac:.2f}: {cache[key]:.1f}%")
            continue

        exp = LocalizationExperiment(model_name, [frac])
        result = exp.run_experiment([scale], num_sentences, "sentences.txt", "prompts.txt", num_trials, False)
        results[frac] = result['accuracies'][0]
        cache[key] = results[frac]
        save_cache(cache)

        del exp
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

    if plot:
        _plot_layer_sweep(model_name, scale, results, num_sentences, num_trials)


def _plot_layer_sweep(model_name, scale, results, num_sentences, num_trials):
    plt.figure(figsize=(10, 6))
    chance = 100 / num_sentences
    n_samples = num_trials * num_sentences

    fracs = sorted(results.keys())
    accs = [results[f] for f in fracs]
    errs = [((a/100 * (1 - a/100) / n_samples) ** 0.5) * 100 for a in accs]

    plt.axhline(y=chance, color='gray', linestyle='--', alpha=0.7, label=f'Chance ({chance:.1f}%)')
    plt.errorbar(fracs, accs, yerr=errs, marker='o', markersize=6, color='blue', linewidth=1.5, capsize=3)

    plt.xlabel('Layer Fraction (0=first, 1=last)'), plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name.split("/")[-1]} - Layer Sweep (scale {scale:+g})')
    plt.xlim(-0.05, 1.05), plt.ylim(0, 105), plt.grid(True, alpha=0.3), plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    filename = f"plots/layer_sweep_{model_name.split('/')[-1]}_{num_sentences}sent_{num_trials}trials.png"
    plt.savefig(filename, dpi=150)
    plt.close()  # Close instead of show() to avoid blocking
    print(f"Plot saved: {filename}")


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="LLM Introspection: Injection Localization")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layers", nargs="+", type=float, default=[0.5])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.1])
    parser.add_argument("--num-sentences", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--size-sweep", action="store_true")
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument("--num-layer-steps", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--prompt-mode", default="introspection",
                        choices=list(PROMPT_MODES.keys()),
                        help="Prompt mode: introspection (default), preference, unusual, first, random")
    parser.add_argument("--prompts-file", default="prompts.txt",
                        help="File containing steering vector prompt pairs (default: prompts.txt)")
    parser.add_argument("--random-noise", action="store_true",
                        help="Use random noise vectors instead of text-derived steering vectors")
    args = parser.parse_args()

    if args.layer_sweep:
        run_layer_sweep(args.model, args.scales[0], args.num_sentences, args.num_trials,
                        args.num_layer_steps, not args.no_plot)
    elif args.size_sweep:
        run_size_sweep(args.scales, args.num_sentences, args.num_trials,
                       None if args.all_layers else args.layers, not args.no_plot)
    else:
        if args.all_layers:
            tmp = LocalizationExperiment(args.model, [0.5], args.prompt_mode)
            args.layers = [i / tmp.num_layers for i in range(tmp.num_layers)]
            del tmp
        exp = LocalizationExperiment(args.model, args.layers, args.prompt_mode)
        exp.run_experiment(args.scales, args.num_sentences, "sentences.txt", args.prompts_file,
                          args.num_trials, not args.no_plot, args.random_noise)


if __name__ == "__main__":
    main()
