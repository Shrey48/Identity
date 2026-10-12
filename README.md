# What's in a Name? Identity Does Not Sway Multi-Agent LLM Deliberation

Eleven open-weight LLMs (1B–70B parameters, four model families) discuss 300 questions across
three domains over 15 rounds of structured deliberation — independent voting, rationale exchange,
and 11 rounds of directed debate with reply-or-decline decisions — under two conditions that
differ only in whether agents can see each other's model name, family, and parameter count. We
find that identity information, while clearly visible to agents, is **functionally ignored**:
agents do not conform to their own family, do not reliably defer to larger models, and do not
form identity-based communication clusters.

## Folder layout

```
.
├── LICENSE
├── README.md                         <-- this file
└── deliberation_code_and_data/
    ├── code/                         <-- runs the deliberation experiment
    │   ├── README.md                 <-- agent roster, protocol, how to run a sweep
    │   ├── requirements.txt
    │   ├── fixed_question_set.json   <-- the exact 300 questions used in the paper
    │   ├── run_experiment.py         <-- runs one question under one condition
    │   ├── run_full_experiment.py    <-- concurrent sweep: all questions x conditions
    │   ├── prompts.py                <-- every prompt template, verbatim (see paper Appendix C)
    │   ├── identities.py             <-- agent roster + every identity-display mode
    │   ├── dataset_loaders.py        <-- MMLU / MoralChoice / Moral Machine loaders
    │   ├── conversation_logger.py    <-- writes .json / .jsonl / _transcript.md per question
    │   ├── make_fixed_question_set.py
    │   ├── analyze_curtain.py        <-- supplementary statistical-analysis suite (not used for the paper's reported tables)
    │   └── launch_servers.sh         <-- one vLLM server per agent
    ├── analysis/                     <-- computes the paper's tables from the deliberation output
    │   ├── README.md                 <-- the paper-to-notebook mapping table, directory layout, how to run
    │   ├── run_experiments_anonymous.ipynb
    │   ├── run_experiments_identity.ipynb
    │   └── run_experiment.ipynb      <-- cross-condition (Experiment 5 / degree of identity)
    └── sample_outputs/               <-- two example output files (one per condition) to see the schema
```

## Paper section → algorithm → notebook experiment

Every quantitative result in Sections 4.1–4.8 and Appendix A is produced by exactly one of the
notebook cells below. (Full detail, including which raw-count cell each table's normalization
builds on, is in `deliberation_code_and_data/analysis/README.md`.)

| Paper section | Algorithm(s) | Table(s) | Notebook experiment(s) |
|---|---|---|---|
| §4.1 | 1 | Table 1 | Exp 1 |
| §4.2 | 2 | Table 2 | Exp 3 |
| §4.3 | 3, 4 | Table 3 | Exp 2, Exp 4 |
| §4.4 | 5 | Table 4 | Exp 9 |
| §4.5 | 6 | Table 5 | Exp 10 (first pass) |
| §4.6 | 7 | Table 6 | Exp 11 |
| §4.7 | 8 | — (degree of identity) | Exp 5 |
| §4.8 | 9, 10, 11 | Tables 7, 8, 9 | Exp 6, Exp 7, Exp 8 |

Experiment 12 and the second pass of Experiment 10 are additional analyses run during this
project but **not** reported in the paper.

Only the **anonymous** and **real_full_identity** conditions are run and reported. The code
supports several other identity conditions (virtual personas, family-only, size-only, gender
reveal, delayed reveal) left over from earlier exploratory work — these are not part of the
paper's results.

## Reproducing the paper's tables, end to end

```bash
# 1. Set up the environment
cd deliberation_code_and_data/code
pip install -r requirements.txt

# 2. Start the 11 vLLM servers (see code/README.md for GPU requirements)
./launch_servers.sh start
./launch_servers.sh status

# 3. Run both conditions against the exact 300-question set shipped in this repo
#    (fixed_question_set.json is auto-detected; --questions-per-dataset is ignored while it exists)
python3 run_full_experiment.py --provider vllm --identity-conditions anonymous
python3 run_full_experiment.py --provider vllm --identity-conditions real_full_identity

# This produces, per condition, one {json, jsonl, transcript.md} triplet per question under
# ./experiments/{mmlu,moralchoice,MoralMachineHuman}/ — see code/README.md's "Output format"
# section for the exact schema.

# 4. Arrange the output into the layout analysis/README.md expects (one top-level folder per
#    condition, each holding the three dataset subfolders directly), then run the notebooks:
cd ../analysis
jupyter notebook run_experiments_anonymous.ipynb    # Tables that depend only on anonymous
jupyter notebook run_experiments_identity.ipynb     # Tables 2-4, 7-9 (identity-only analyses)
jupyter notebook run_experiment.ipynb               # Table-independent cross-condition analysis (Exp 5)
```

Each notebook cell corresponds to one algorithm in Appendix A and writes its own
`results/experiment_N/results.json`. See `deliberation_code_and_data/analysis/README.md` for the
full experiment-by-experiment description, exact input fields used, and output file formats.

## Full deliberation logs

The complete deliberation logs for all 300 questions in both conditions are not released.
`deliberation_code_and_data/sample_outputs/` contains two example output files (one per
condition) so the schema can be inspected, and the code in this repository reproduces the full
logs from scratch given GPU access (see "Reproducing the paper's tables" above).

## Licenses

See `LICENSE` for this repository's code license (MIT), and that file's notes on the third-party
dataset and model licenses that apply to the data and model weights used here.
