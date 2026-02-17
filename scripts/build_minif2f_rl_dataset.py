#!/usr/bin/env python3
"""Convert HaimingW/miniF2F-lean4 HF dataset into JSONL for RL training.

Usage:
    python scripts/build_minif2f_rl_dataset.py

Produces:
    data/minif2f_valid.jsonl   (244 rows - for training)
    data/minif2f_test.jsonl    (244 rows - held out for eval)
"""
import json
import os
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a Lean 4 theorem prover. You will be given a theorem to prove. "
    "At each step, output exactly one tactic to apply. "
    "Think step by step about which tactic to use."
)


def make_prompt(row):
    """Build the prompt the LLM sees as its initial observation."""
    parts = []
    # Include the informal problem statement if available
    if row["informal_prefix"]:
        parts.append(f"Problem: {row['informal_prefix'].strip()}")
    # Include the formal theorem signature
    parts.append(f"Formal statement:\n```lean\n{row['formal_statement'].strip()}\n```")
    # Include the initial goal
    parts.append(f"Current goal:\n```\n{row['goal'].strip()}\n```")
    parts.append("Provide the next tactic.")
    return "\n\n".join(parts)


def make_label(row):
    """Build the label dict that the agent environment will parse at runtime."""
    return json.dumps({
        "name": row["name"],
        "formal_statement": row["formal_statement"],
        "goal": row["goal"],
        "header": row["header"],
        "formal_proof": row.get("formal_proof", ""),
    })


def main():
    ds = load_dataset("HaimingW/miniF2F-lean4")

    os.makedirs("data", exist_ok=True)

    for split_name in ["valid", "test"]:
        split_ds = ds[split_name]
        out_path = f"data/minif2f_{split_name}.jsonl"

        with open(out_path, "w") as f:
            for row in split_ds:
                record = {
                    "prompt": make_prompt(row),
                    "label": make_label(row),
                }
                f.write(json.dumps(record) + "\n")

        print(f"Wrote {len(split_ds)} rows to {out_path}")

    # Also write few-shot examples
    if "few_shot_examples" in ds:
        few_shot = ds["few_shot_examples"]
        out_path = "data/minif2f_few_shot.jsonl"
        with open(out_path, "w") as f:
            for row in few_shot:
                record = {
                    "prompt": make_prompt(row),
                    "label": make_label(row),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(few_shot)} rows to {out_path}")


if __name__ == "__main__":
    main()
