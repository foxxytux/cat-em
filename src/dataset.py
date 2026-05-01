"""
Dataset loading and preprocessing for CodeAgent-RWKV
Handles all 12 datasets with thinking format and category mixing.
"""

import os
import json
import random
from typing import Dict, List, Optional, Callable
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from torch.utils.data import IterableDataset

# Dataset configurations with category assignments
DATASET_CONFIGS = {
    # Code datasets
    "bigcode/starcoder2data-extras": {
        "category": "code",
        "split": "train",
        "text_column": "content",
        "format": "code_only",
    },
    "bigcode/the-stack-v2-dedup": {
        "category": "code",
        "split": "train",
        "text_column": "content",
        "format": "code_only",
        "streaming": True,
    },
    "m-a-p/CodeFeedback-Filtered-Instruction": {
        "category": "code",
        "split": "train",
        "text_column": None,  # instruction/answer format
        "format": "instruction",
    },
    "princeton-nlp/SWE-bench_Verified": {
        "category": "code",
        "split": "train",
        "text_column": None,
        "format": "swe_bench",
    },
    "nvidia/OpenCodeReasoning": {
        "category": "code",
        "split": "train",
        "text_column": None,
        "format": "reasoning",
    },
    # Agentic datasets
    "R2E-Gym": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "agentic",
    },
    "agentica-org/DeepSWE-Preview": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "agentic",
    },
    "THUDM/AgentInstruct": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "instruction",
    },
    "tuandunghcmut/toolbench-v1": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "tool",
    },
    "open-thoughts/OpenThoughts-Agent-v1-SFT": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "reasoning",
    },
    "princeton-nlp/SWE-bench": {
        "category": "agentic",
        "split": "train",
        "text_column": None,
        "format": "swe_bench",
    },
    # Reasoning datasets
    "open-thoughts/OpenThoughts-114k": {
        "category": "reasoning",
        "split": "train",
        "text_column": None,
        "format": "reasoning",
    },
    # Conversation datasets
    "openbmb/UltraChat": {
        "category": "conversation",
        "split": "train",
        "text_column": None,
        "format": "conversation",
    },
}

CATEGORY_WEIGHTS = {
    "code": 0.35,
    "agentic": 0.35,
    "reasoning": 0.20,
    "conversation": 0.10,
}


class ThinkingFormat:
    """Applies the required thinking format to conversations."""
    
    def __init__(
        self,
        user_token: str = "User:",
        thinking_token: str = "Thinking...",
        answer_token: str = "Answer:",
        system_prompt: str = "You are a helpful coding assistant. Think step by step before providing your answer.",
    ):
        self.user_token = user_token
        self.thinking_token = thinking_token
        self.answer_token = answer_token
        self.system_prompt = system_prompt
    
    def format_conversation(
        self,
        user_text: str,
        thinking_text: Optional[str] = None,
        answer_text: Optional[str] = None,
        include_system: bool = True,
    ) -> str:
        """Format a single turn with thinking format."""
        parts = []
        if include_system:
            parts.append(f"System: {self.system_prompt}")
        
        parts.append(f"{self.user_token} {user_text.strip()}")
        
        if thinking_text:
            parts.append(f"\n{self.thinking_token}\n{thinking_text.strip()}")
        
        if answer_text:
            parts.append(f"\n{self.answer_token} {answer_text.strip()}")
        
        return "\n".join(parts)
    
    def format_instruction(self, instruction: str, input_text: str = "", output: str = "", reasoning: str = "") -> str:
        """Format instruction-following data with thinking."""
        user = instruction
        if input_text:
            user += f"\n{input_text}"
        
        return self.format_conversation(
            user_text=user,
            thinking_text=reasoning if reasoning else None,
            answer_text=output,
        )
    
    def format_code(self, code: str, language: str = "", problem: str = "") -> str:
        """Format code data with thinking."""
        user = problem if problem else "Write code."
        if language:
            user += f" Language: {language}"
        
        thinking = f"I'll write a {language} solution. Let me think about the approach..."
        
        return self.format_conversation(
            user_text=user,
            thinking_text=thinking,
            answer_text=f"```\n{code}\n```",
        )


def format_swe_bench(example: Dict, formatter: ThinkingFormat) -> str:
    """Format SWE-bench example."""
    problem = example.get("problem_statement", example.get("text", ""))
    solution = example.get("patch", example.get("solution", ""))
    reasoning = example.get("reasoning", "Let me analyze the issue and develop a fix...")
    
    return formatter.format_conversation(
        user_text=f"Fix this issue:\n{problem}",
        thinking_text=reasoning,
        answer_text=solution,
    )


def format_toolbench(example: Dict, formatter: ThinkingFormat) -> str:
    """Format ToolBench example."""
    query = example.get("query", example.get("instruction", ""))
    api_calls = example.get("api_calls", example.get("tools", ""))
    final_answer = example.get("final_answer", example.get("output", ""))
    
    thinking = f"I need to use tools to solve this. Available tools: {str(api_calls)[:500]}"
    
    return formatter.format_conversation(
        user_text=query,
        thinking_text=thinking,
        answer_text=final_answer,
    )


def format_reasoning(example: Dict, formatter: ThinkingFormat) -> str:
    """Format reasoning dataset example."""
    question = example.get("question", example.get("problem", example.get("instruction", "")))
    reasoning = example.get("reasoning", example.get("chain_of_thought", example.get("thought", "")))
    answer = example.get("answer", example.get("solution", example.get("output", "")))
    
    return formatter.format_conversation(
        user_text=question,
        thinking_text=reasoning if reasoning else "Let me think through this step by step...",
        answer_text=answer,
    )


def format_conversation(example: Dict, formatter: ThinkingFormat) -> str:
    """Format multi-turn conversation."""
    data = example.get("data", example)
    
    if isinstance(data, list):
        # Multi-turn format
        turns = []
        for i, turn in enumerate(data):
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))
            
            if role in ["user", "human"]:
                turns.append(f"{formatter.user_token} {content}")
            elif role in ["assistant", "gpt"]:
                if i > 0 and turns:
                    # Add thinking to assistant turns
                    turns.append(f"\n{formatter.thinking_token}\nLet me provide a helpful response...")
                    turns.append(f"\n{formatter.answer_token} {content}")
                else:
                    turns.append(f"{formatter.answer_token} {content}")
        
        return "\n".join(turns)
    else:
        # Single text
        return formatter.format_conversation(
            user_text="Let's chat.",
            answer_text=str(data),
        )


def format_instruction_data(example: Dict, formatter: ThinkingFormat) -> str:
    """Format instruction data."""
    instruction = example.get("instruction", example.get("prompt", ""))
    input_text = example.get("input", "")
    output = example.get("output", example.get("response", ""))
    
    # Try to extract reasoning if available
    reasoning = example.get("reasoning", example.get("explanation", ""))
    
    return formatter.format_instruction(
        instruction=instruction,
        input_text=input_text,
        output=output,
        reasoning=reasoning,
    )


def format_code_only(example: Dict, formatter: ThinkingFormat) -> str:
    """Format raw code data."""
    code = example.get("content", example.get("code", ""))
    lang = example.get("lang", example.get("language", ""))
    
    return formatter.format_code(
        code=code,
        language=lang,
    )


def format_agentic(example: Dict, formatter: ThinkingFormat) -> str:
    """Format agentic trajectory data."""
    query = example.get("query", example.get("task", example.get("instruction", "")))
    trajectory = example.get("trajectory", example.get("steps", ""))
    result = example.get("result", example.get("output", example.get("solution", "")))
    
    thinking = f"I'll work through this agentically. {str(trajectory)[:1000]}"
    
    return formatter.format_conversation(
        user_text=query,
        thinking_text=thinking,
        answer_text=result,
    )


FORMAT_FUNCTIONS = {
    "code_only": format_code_only,
    "instruction": format_instruction_data,
    "swe_bench": format_swe_bench,
    "reasoning": format_reasoning,
    "agentic": format_agentic,
    "tool": format_toolbench,
    "conversation": format_conversation,
}


def load_and_format_dataset(
    dataset_name: str,
    formatter: ThinkingFormat,
    split: str = "train",
    streaming: bool = False,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = True,
) -> Optional[Dataset]:
    """Load a single dataset and apply thinking format."""
    
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        print(f"Warning: No config for {dataset_name}, skipping")
        return None
    
    format_type = config["format"]
    format_fn = FORMAT_FUNCTIONS.get(format_type)
    
    try:
        print(f"Loading {dataset_name} (streaming={streaming})...")
        
        # Handle dataset-specific loading quirks
        load_kwargs = {
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        
        # Some datasets need specific subsets
        if dataset_name == "bigcode/the-stack-v2-dedup":
            load_kwargs["data_dir"] = "data"
        
        ds = load_dataset(dataset_name, **load_kwargs)
        
        if streaming:
            # For streaming, we'll handle formatting during iteration
            return ds
        
        # Apply formatting
        def map_fn(example):
            try:
                text = format_fn(example, formatter)
                return {"text": text}
            except Exception as e:
                return {"text": ""}
        
        ds = ds.map(map_fn, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 50)  # Filter empty/short
        
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        return ds
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


def create_mixed_dataset(
    dataset_names: List[str],
    formatter: ThinkingFormat,
    category_weights: Dict[str, float] = None,
    samples_per_category: int = 50000,
    streaming: bool = True,
    seed: int = 42,
) -> IterableDataset:
    """Create a weighted mix of all datasets by category."""
    
    if category_weights is None:
        category_weights = CATEGORY_WEIGHTS
    
    random.seed(seed)
    
    # Group datasets by category
    category_datasets = {cat: [] for cat in category_weights.keys()}
    
    for name in dataset_names:
        config = DATASET_CONFIGS.get(name)
        if config:
            category_datasets[config["category"]].append(name)
    
    # Calculate samples per dataset within each category
    all_datasets = []
    
    for category, weight in category_weights.items():
        cat_datasets = category_datasets.get(category, [])
        if not cat_datasets:
            continue
        
        cat_total_samples = int(samples_per_category * weight * len(category_weights))
        samples_per_ds = cat_total_samples // max(len(cat_datasets), 1)
        
        for ds_name in cat_datasets:
            ds = load_and_format_dataset(
                ds_name,
                formatter,
                streaming=streaming,
                max_samples=samples_per_ds if not streaming else None,
            )
            if ds is not None:
                all_datasets.append((category, ds))
                print(f"Added {ds_name} to {category} mix")
    
    # Create interleaved dataset
    if not all_datasets:
        raise ValueError("No datasets could be loaded!")
    
    if streaming:
        # For streaming, create a custom iterable
        return MixedStreamingDataset(all_datasets, category_weights, seed)
    else:
        # Concatenate and shuffle
        from datasets import interleave_datasets
        
        datasets_list = [ds for _, ds in all_datasets]
        probabilities = [category_weights[cat] for cat, _ in all_datasets]
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        mixed = interleave_datasets(datasets_list, probabilities=probabilities, seed=seed)
        return mixed


class MixedStreamingDataset(IterableDataset):
    """Custom iterable dataset for streaming mixed data."""
    
    def __init__(self, datasets, category_weights, seed=42):
        self.datasets = datasets
        self.category_weights = category_weights
        self.seed = seed
        self.rng = random.Random(seed)
        
    def __iter__(self):
        iterators = []
        weights = []
        
        for category, ds in self.datasets:
            if hasattr(ds, '__iter__'):
                iterators.append(iter(ds))
                weights.append(self.category_weights.get(category, 0.25))
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        while True:
            # Weighted random choice
            choice = self.rng.choices(range(len(iterators)), weights=weights, k=1)[0]
            
            try:
                item = next(iterators[choice])
                if isinstance(item, dict) and "text" in item:
                    yield item
                elif isinstance(item, dict):
                    # Try to format on the fly if not already formatted
                    pass
            except StopIteration:
                # Remove exhausted iterator
                iterators.pop(choice)
                weights.pop(choice)
                
                if not iterators:
                    break
                
                # Renormalize
                total = sum(weights)
                weights = [w / total for w in weights]


def get_tokenized_dataset(
    tokenizer,
    dataset_names: List[str],
    max_length: int = 4096,
    formatter: Optional[ThinkingFormat] = None,
    streaming: bool = True,
    num_proc: int = 4,
):
    """Get tokenized dataset ready for training."""
    
    if formatter is None:
        formatter = ThinkingFormat()
    
    # Load mixed dataset
    mixed_ds = create_mixed_dataset(
        dataset_names=dataset_names,
        formatter=formatter,
        streaming=streaming,
    )
    
    # Tokenize
    def tokenize_function(examples):
        texts = examples["text"]
        
        # Handle batch
        if isinstance(texts, str):
            texts = [texts]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    if streaming:
        # For streaming, we tokenize in the collate_fn or map lazily
        return mixed_ds
    else:
        tokenized = mixed_ds.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=mixed_ds.column_names,
        )
        return tokenized
