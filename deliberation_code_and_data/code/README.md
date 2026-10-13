# Multi-Agent LLM Deliberation: Simulation Code

Simulation code and sample outputs for a structured deliberation experiment among eleven open-weight LLM agents under two identity conditions.

## Experimental conditions

| Condition | What agents see about each other | Example label |
|---|---|---|
| **anonymous** | System name only | `Agent_06` |
| **real_full_identity** | System name + model name, version, and size | `Agent_06 [LLaMA 3.1 70B]` |

Everything else (questions, agent roster, prompt templates, decoding parameters, debate structure) is identical across conditions.

Note: `identities.py` and `run_experiment.py` support several additional identity conditions
(virtual personas, family-only reveal, size-only reveal, gender reveal, delayed reveal, and
others) used for earlier exploratory work. **Only `anonymous` and `real_full_identity` are run
and reported in the paper.**

## Agent roster

Eleven open-weight models spanning four families and a range of parameter sizes. The table below
lists the identity string agents actually see in the `real_full_identity` condition, the exact
checkpoint served by `launch_servers.sh` (not the unquantized base checkpoint, where different),
and its serving weight format. In the `anonymous` condition only the bare `Agent_NN` id is shown.

| Agent | Identity string shown | Served checkpoint (vLLM) | Weights | Club |
|---|---|---|---|---|
| Agent_01 | `Agent_01 [Qwen 2.5 3B]` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | AWQ 4-bit | Poor |
| Agent_02 | `Agent_02 [Qwen 2.5 7B]` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | AWQ 4-bit | Poor |
| Agent_03 | `Agent_03 [Qwen 2.5 32B]` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | AWQ 4-bit | Rich |
| Agent_04 | `Agent_04 [Gemma 3 1B]` | `google/gemma-3-1b-it` | BF16 | Poor |
| Agent_05 | `Agent_05 [LLaMA 3.1 8B]` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ 4-bit | Poor |
| Agent_06 | `Agent_06 [LLaMA 3.1 70B]` | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | AWQ 4-bit | Rich |
| Agent_07 | `Agent_07 [Gemma 3 12B]` | `gaunernst/gemma-3-12b-it-int4-awq` | AWQ 4-bit | Rich |
| Agent_08 | `Agent_08 [Gemma 3 4B]` | `google/gemma-3-4b-it` | FP8 | Poor |
| Agent_09 | `Agent_09 [Ministral 3 3B]` | `mistralai/Ministral-3-3B-Instruct-2512` | FP8 | Poor |
| Agent_10 | `Agent_10 [Mistral 7B v0.3]` | `solidrust/Mistral-7B-Instruct-v0.3-AWQ` | AWQ 4-bit | Poor |
| Agent_11 | `Agent_11 [LLaMA 3.2 3B]` | `casperhansen/llama-3.2-3b-instruct-awq` | AWQ 4-bit | Poor |

Family balance: Qwen 3, Gemma 3, LLaMA 3, Mistral 2. Rich club (>10B parameters): Agents 03, 06, 07
(3 agents). Poor club (the remaining eight): Agents 01, 02, 04, 05, 08, 09, 10, 11.

Note: Agents 02 and 10 are genuine 7B-parameter checkpoints. An unrelated internal metadata field
(`size_str`, used only for the rich/poor club split, where the >10B cutoff places both firmly in
the poor club either way) labels them `"8B"` — this does not reflect a different model being served,
and does not affect the rich/poor classification above.

## Deliberation protocol

Each question goes through 15 voting stages in a fixed order:

| Stage | Name | What happens |
|---|---|---|
| 0 | `baseline` | Independent vote (no visibility of others) |
| 1 | `bandwagon_revote` | See raw votes only (no rationale), may revote |
| 2 | `rationale_collection` | State reasoning behind current vote (no revote) |
| 3 | `reasoned_conformity_revote` | See votes + rationales, may revote |
| 4--14 | `debate_round_1` -- `debate_round_11` | Each agent writes one targeted statement to another agent, each addressed agent replies or declines, then all agents revote |

All agents use temperature 0 (greedy decoding). If a response cannot be parsed, it is regenerated at temperature 0.7 (at most twice); every retry is recorded in the output.

## Datasets

300 questions total, drawn from three benchmarks:

| Dataset | Source | Questions | Sampling |
|---|---|---|---|
| MMLU | `cais/mmlu` | 100 | 25 per subject (formal_logic, philosophy, moral_disputes, moral_scenarios) |
| MoralChoice | `ninoscherrer/moralchoice` | 100 | 50 low-ambiguity + 50 high-ambiguity |
| Moral Machine | `Jerry999/MoralMachineHuman` | 100 | Stratified sample |

Each question is normalized to a binary (Option A / Option B) format. The exact 300-question set
used for the reported runs is included as `fixed_question_set.json` in this folder — point
`run_experiment.py`/`run_full_experiment.py` at it (see Steps below) to reproduce the paper's
questions exactly, rather than regenerating a fresh sample via `make_fixed_question_set.py`
(which `dataset_loaders.py` seeds deterministically, but a fresh run is provided here for
convenience and as the ground truth regardless).

## Code structure

```
code/
  run_experiment.py            Core: runs one question under one condition
  run_full_experiment.py       Concurrent sweep: all questions x conditions
  prompts.py                   Prompt templates for every stage
  identities.py                Agent roster, identity modes, persona pool
  dataset_loaders.py           Loads MMLU, MoralChoice, Moral Machine
  conversation_logger.py       Real-time logging (.json, .jsonl, .md)
  make_fixed_question_set.py   Locks the question set across conditions
  analyze_curtain.py           Full statistical analysis (Excel, CSV, figures)
  launch_servers.sh            Starts one vLLM server per agent (ports 8001--8011)
  requirements.txt             Python dependencies

sample_outputs/
  mmlu_formal_logic_0000_anonymous.json
  mmlu_formal_logic_0000_real_full_identity.json
```

## Running the deliberation

### Prerequisites

- Python 3.10+
- Two GPUs with ~98 GB VRAM each (the reported runs used 2x NVIDIA RTX PRO 6000 Blackwell; 80+ GB per GPU is the practical minimum for serving all 11 models at `--max-model-len 32768`)
- `pip install -r code/requirements.txt`

### Steps

```bash
# 1. Start model servers
cd code/
./launch_servers.sh start
./launch_servers.sh status

# 2. (Optional) Fix the question set for cross-condition reproducibility
python3 make_fixed_question_set.py experiments/

# 3. Run both conditions
python3 run_full_experiment.py --provider vllm

# Or one at a time:
python3 run_full_experiment.py --provider vllm --identity-conditions anonymous
python3 run_full_experiment.py --provider vllm --identity-conditions real_full_identity

# Dry run (no GPU required):
python3 run_experiment.py --dry-run --dataset mmlu --sample-idx 0 --identity-mode anonymous
```

The sweep driver is resumable: completed questions are skipped on re-run.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--datasets` | `moralchoice,moralmachine,mmlu` | Comma-separated dataset names |
| `--questions-per-dataset` | 50 | Stratified questions per dataset |
| `--identity-conditions` | all | Restrict to specific conditions |
| `--concurrency` | 40 | Parallel simulations |
| `--provider` | `vllm` | Backend (`vllm`, `mock`, `ollama`, `openrouter`, etc.) |
| `--rounds` | 11 | Debate rounds per question |

## Output format

Each deliberation produces three files per question per condition:

- `<problem_id>_<condition>.json`: Structured record of the full deliberation.
- `<problem_id>_<condition>.jsonl`: Streaming event log (one JSON object per line).
- `<problem_id>_<condition>_transcript.md`: Human-readable Markdown transcript.

### JSON schema

```
{
  "problem_statement": {
    "id", "dataset", "category", "question",
    "option_a", "option_b", "context", "ground_truth", "metadata"
  },
  "simulation_metadata": {
    "pid", "dataset", "category", "identity_condition",
    "disclosure_type", "total_debate_rounds", "timestamp",
    "agent_roster": [{ "agent_id", "model_name", "family", "size" }],
    "retry_log"
  },
  "group_chat_transcript": [ ... ],
  "stages_detail": [
    {
      "stage_name", "stage_index", "tally", "votes": [
        { "agent_id", "display_identity", "family", "size_str",
          "size_billions", "previous_vote", "vote", "confidence",
          "confidence_delta", "flipped", "rationale" }
      ],
      "statements": [ ... ],
      "replies": [ ... ],
      "flips_in_stage"
    }
  ],
  "summary_analytics": {
    "initial_tally", "final_tally", "consensus_achieved",
    "total_flips", "agent_trajectories", "flips_log", "targeting_log"
  }
}
```

`agent_trajectories` gives per-agent longitudinal vote and confidence histories across all 15 stages.

## Full statistical analysis

`analyze_curtain.py` computes a comprehensive statistical analysis suite from the raw outputs:

```bash
python3 code/analyze_curtain.py results
```

Produces `results/analysis_output/curtain_analysis.xlsx` with one sheet per analysis, each stating its question, method, and interpretation. Also writes CSV tables and PNG figures.

## Reproducibility notes

- All first attempts use temperature 0 (greedy decoding) to eliminate sampling noise.
- Parse failures trigger up to two retries at temperature 0.7; every retry is logged.
- System prompts fold the system message into the user turn (some chat templates reject a separate system role).
- The `MAX_TOKENS` budget for agent responses is 300 tokens.
- vLLM servers use a 32,768-token context window.
