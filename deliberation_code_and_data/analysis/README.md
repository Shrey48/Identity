# Multi-Agent LLM Deliberation: Analysis Notebooks

Three Jupyter notebooks that compute 12 experiments from the deliberation simulation output JSONs. Each experiment analyzes a specific aspect of agent voting behavior, social influence, and communication patterns across the anonymous and identity conditions.

**Not every experiment here is reported in the paper.** Experiment 12 (communication-distance
analysis) and the second pass of Experiment 10 (in `run_experiments_identity.ipynb`) are
additional analyses that were run but are not part of the published results. The table below maps
every paper section and algorithm to the exact notebook and experiment number that produced it.

## Paper-to-notebook mapping

| Paper section | Algorithm(s) | Notebook experiment(s) |
|---|---|---|
| §4.1 | 1 | Exp 1 |
| §4.2 | 2 | Exp 3 |
| §4.3 | 3, 4 | Exp 2, Exp 4 |
| §4.4 | 5 | Exp 9 |
| §4.5 | 6 | Exp 10 (first pass) |
| §4.6 | 7 | Exp 11 |
| §4.7 | 8 | Exp 5 |
| §4.8 | 9, 10, 11 | Exp 6, Exp 7, Exp 8 |

Experiment 12 and Experiment 10's second pass are not referenced by any paper section or
algorithm — they are additional analyses, not part of the reported results.

## Notebooks

| Notebook | Condition | Run from |
|---|---|---|
| `run_experiments_anonymous.ipynb` | Anonymous only | Inside the anonymous output folder |
| `run_experiments_identity.ipynb` | Identity only | Inside the identity output folder |
| `run_experiment.ipynb` | Cross-condition | Parent folder containing both condition folders |

## Experiments

### `run_experiments_anonymous.ipynb`

| Experiment | What it measures |
|---|---|
| 1 | Majority/minority bias at bandwagon revote (stage 1): how many agents flip toward the majority vs the minority after seeing raw vote counts |
| 6 | Same-vote vs cross-vote targeting in debate (A-pings-A, B-pings-B, cross-pings) with normalized counts, plus two filtered variants (one requiring >=2 persistent voters per side, one requiring >=3 per side in every round) |
| 9 | Majority/minority rational bias at reasoned conformity revote (stage 3): flips after seeing rationales |
| 10 | Back-to-own-rational reversion: among agents whose vote differs between stage 1 (bandwagon) and stage 3 (after rationales), how many land back on their original Round 0 vote vs a third position |
| 11 | Back-to-own-rational split by whether the agent reverted to the majority or minority side |
| 12 | Communication distance: pings (unreciprocated statements) vs talks (reciprocated exchanges) binned by the vote-agreement distance between speaker and target across debate rounds, plus a filtered variant |

### `run_experiments_identity.ipynb`

| Experiment | What it measures |
|---|---|
| 1 | Majority/minority bias at bandwagon revote (stage 1) |
| 2 | Within-family majority/minority bias: whether agents flip toward the majority within their own model family (Qwen, Gemma, LLaMA, Mistral) |
| 3 | Minority-flip attribution: among agents that flip to the global minority at stage 1, how many are explained by their family's internal majority vs the rich club's majority specifically (always compared against the rich club, not the agent's own size group) vs neither (exceptional) |
| 4 | Rich club majority vs poor club majority: whether flips at stage 1 align with the majority among large (>10B) or small (the remaining eight) models |
| 6 | Same-vote vs cross-vote targeting in debate, with two filtered variants (same as anonymous) |
| 7 | Family-to-family targeting matrix: counts of speaker-family to target-family pings across all 11 debate rounds, raw and normalized by group-pair size |
| 8 | Rich-to-rich, rich-to-poor, poor-to-rich, poor-to-poor targeting: same logic as experiment 7 but grouped by parameter-size tier instead of family |
| 9 | Majority/minority rational bias at reasoned conformity revote (stage 3) |
| 10 | Back-to-own-rational reversion (same definition as above) + a second pass attributing minority flips at stage 3 to family majority vs rich-model majority |
| 11 | Back-to-own-rational split by majority vs minority |
| 12 | Communication distance analysis with a filtered variant |

### `run_experiment.ipynb` (cross-condition)

| Experiment | What it measures |
|---|---|
| 5 | Degree of Identity (DoI): for each agent on each question, compare its flip sequence under anonymous vs identity; count XOR-disagreements (cases where the agent flipped in one condition but not the other at the same stage transition). Aggregated per dataset and overall. |

## Expected directory layout

```
project_root/
  anonymous/                    <-- anonymous condition output JSONs
    mmlu/
      *.json
    moralchoice/
      *.json
    MoralMachineHuman/
      *.json
  real_full_identity/           <-- identity condition output JSONs
    mmlu/
      *.json
    moralchoice/
      *.json
    MoralMachineHuman/
      *.json
  analysis/                     <-- these notebooks
    run_experiments_anonymous.ipynb
    run_experiments_identity.ipynb
    run_experiment.ipynb
```

The single-condition notebooks (`run_experiments_anonymous.ipynb`, `run_experiments_identity.ipynb`) use `Path(".").resolve()` as their data root, so they must be run with the working directory set to the respective condition folder. The cross-condition notebook (`run_experiment.ipynb`) expects both `anonymous/` and `real_full_identity/` folders in the working directory.

## Running

```bash
# 1. Anonymous-only experiments
cd anonymous/
jupyter notebook ../analysis/run_experiments_anonymous.ipynb

# 2. Identity-only experiments
cd ../real_full_identity/
jupyter notebook ../analysis/run_experiments_identity.ipynb

# 3. Cross-condition experiment
cd ..
jupyter notebook analysis/run_experiment.ipynb
```

Each notebook writes results into `./results/experiment_N/results.json` and `./results/experiment_N/result.xlsx`. The `results/` directory and its `experiment_1/` through `experiment_12/` subdirectories must exist before running (or be created by the notebook).

## Output

Each experiment produces:

- `results/experiment_N/results.json`: Full per-question breakdown including agent rosters, per-round details, and dataset-level and overall aggregates.
- `results/experiment_N/result.xlsx`: Summary Excel workbook with a "Master analytics" sheet showing per-dataset and overall totals.

Filtered variants of experiments 6 and 12 additionally write `results_filtered.json` / `result_filtered.xlsx` and `results_filter_2.json` / `result_filter_2.xlsx`.

## Input JSON fields used

The notebooks read the following fields from the deliberation output JSONs (see the deliberation code README for the full schema):

- `simulation_metadata.dataset`, `simulation_metadata.pid`, `simulation_metadata.identity_condition`, `simulation_metadata.agent_roster`
- `stages_detail[0]` (baseline), `stages_detail[1]` (bandwagon_revote), `stages_detail[3]` (reasoned_conformity_revote), `stages_detail[4:15]` (debate_round_1 through debate_round_11)
- Per-vote fields: `agent_id`, `family`, `size_str`, `previous_vote`, `vote`
- Per-stage fields: `tally`, `statements` (with `speaker_agent_id`, `target_agent_id`), `replies` (with `replier_agent_id`, `sender_agent_id`, `decision`)
- `summary_analytics.agent_trajectories[].agent_id`, `summary_analytics.agent_trajectories[].vote_history`

## Dependencies

See `../code/requirements.txt` (the "Analysis" section) for the full, pinned dependency list —
`jupyter`, `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `xlsxwriter`, `matplotlib`, and
`jsonpath-ng`.
