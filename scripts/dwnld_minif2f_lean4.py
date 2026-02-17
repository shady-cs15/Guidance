from datasets import load_dataset
ds = load_dataset("HaimingW/miniF2F-lean4")
# This gives you: name, formal_statement, goal, header, informal_prefix, split
# Plus the repo_url and commit that LeanDojo needs