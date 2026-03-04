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

SYSTEM_PROMPT = """\
You are an interactive Lean 4 theorem prover. You are given a formal theorem \
statement and the current proof goal. Your task is to provide the next tactic \
to make progress toward closing the proof.

Rules:
- Output EXACTLY ONE TACTIC per response inside a ```lean code block.
- This is an INTERACTIVE session: after each tactic, you will see the updated \
goal state. You will then provide the next tactic. Build the proof step by step.
- Do NOT write multi-line proof blocks or complete proofs. Provide one tactic, \
see the result, then provide the next.
- Do NOT use nested proofs like `have h := by ...` or `suffices : P := by ...`. \
Instead, state the intermediate result (e.g., `have h : P`) and prove it in \
the next step when Lean asks for it.
- Do NOT write natural language explanations, reasoning, or problem statements — \
only output the tactic code inside the ```lean block.
- Do NOT regenerate the theorem statement. Only provide tactics.
- Use valid Lean 4 / Mathlib tactics such as: simp, norm_num, linarith, \
nlinarith, omega, ring, intro, apply, exact, have, constructor, cases, \
induction, rw, ext, funext, aesop, decide, push_neg, contrapose, by_contra, \
field_simp, gcongr, positivity, norm_cast, ring_nf, calc, obtain, suffices, \
refine, use.
- After each tactic you will receive the updated goal state or an error \
message. Read it carefully and adapt your next tactic accordingly.
- If a tactic errors, try a different approach — do not repeat the same \
failing tactic.

Example response format:
```lean
linarith [h₀, h₁]
```\
"""


def make_prompt(row):
    """Build the prompt the LLM sees as its initial observation.

    Returns a conversation array (list of message dicts) so that
    ``apply_chat_template`` uses our custom system prompt instead of
    the model's default one.
    """
    parts = []
    # Include the informal problem statement if available
    if row["informal_prefix"]:
        parts.append(f"Problem: {row['informal_prefix'].strip()}")
    # Include the formal theorem signature
    parts.append(f"Formal statement:\n```lean\n{row['formal_statement'].strip()}\n```")
    # Include the initial goal
    parts.append(f"Current goal:\n```\n{row['goal'].strip()}\n```")
    parts.append("Provide the next tactic.")
    user_content = "\n\n".join(parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
