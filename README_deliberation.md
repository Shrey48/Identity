# Multi-Agent LLM Deliberation: Simulation Code

Simulation code and sample outputs for a structured deliberation experiment among eleven open-weight LLM agents under two identity conditions.

## Experimental conditions

| Condition | What agents see about each other | Example label |
|---|---|---|
| **anonymous** | System name only | `Agent_06` |
| **real_full_identity** | System name + model name, version, and size | `Agent_06 [LLaMA 3.1 70B]` |

Everything else (questions, agent roster, prompt templates, decoding parameters, debate structure) is identical across conditions.

## Agent roster

Eleven open-weight models spanning four families and a range of parameter sizes, served via vLLM:

| Agent | Family | Size | HuggingFace checkpoint |
|---|---|---|---|
| Agent_01 | Qwen | 3B | `Qwen/Qwen2.5-3B-Instruct` |
| Agent_02 | Qwen | 8B | `Qwen/Qwen2.5-7B-Instruct` |
| Agent_03 | Qwen | 32B | `Qwen/Qwen2.5-32B-Instruct` |
| Agent_04 | Gemma | 1B | `google/gemma-3-1b-it` |
| Agent_07 | Gemma | 12B | `google/gemma-3-12b-it` |
| Agent_08 | Gemma | 4B | `google/gemma-3-4b-it` |
| Agent_05 | LLaMA | 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Agent_06 | LLaMA | 70B | `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| Agent_11 | LLaMA | 3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Agent_09 | Mistral | 3B | `mistralai/Ministral-3-3B-Instruct-2512` |
| Agent_10 | Mistral | 8B | `mistralai/Mistral-7B-Instruct-v0.3` |

Family balance: Qwen 3, Gemma 3, LLaMA 3, Mistral 2.

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

Each question is normalized to a binary (Option A / Option B) format. The exact question set is fixed across conditions via `make_fixed_question_set.py`.

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
- Two GPUs with 80+ GB VRAM each (for vLLM serving of all 11 models)
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
