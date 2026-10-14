"""
================================================================================
MULTI-AGENT DEMOCRATIC LLM REASONING SIMULATION SYSTEM
================================================================================
Research Question:
How does identity (real or virtual) affect how LLMs reason, communicate,
and vote when posed with moral or deep-reasoning questions in a group setting?

Pipeline (per problem statement, per identity condition):
  0. baseline                  - independent vote, no visibility of others
  1. bandwagon_revote          - see raw votes only (no rationale), may revote
  2. rationale_collection      - state the reasoning behind the current vote
  3. reasoned_conformity_revote- see votes + rationales, may revote
  4..14. debate_round_1..11    - every agent independently picks a target and
                                  makes a statement, then (after all statements
                                  for the round are revealed) everyone revotes

Target Open-Weight Model Pool (12 Agents):
- Qwen Family:   1.5B, 2B, 8B, 32B
- LLaMA Family:  1B, 8B, 70B
- Gemma Family:  1B, 4B, 12B
- Mistral Family: 3B, 8B (or 14B)

Datasets Supported:
- ninoscherrer/moralchoice
- Jerry999/MoralMachineHuman
- cais/mmlu
================================================================================
"""

import os
import sys
import json
import time
import re
import random
import logging
import argparse
from datetime import datetime
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor

# Import local sub-modules
from dataset_loaders import get_dataset
from conversation_logger import ConversationLogger
from identities import (
    AgentMetadata,
    FULL_PERSONA_COMBO,
    build_default_agent_roster,
    format_agent_display_identity,
    get_agent_self_identity_prompt_block
)
from prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    BASELINE_VOTE_PROMPT_TEMPLATE,
    VOTES_ONLY_REVOTE_PROMPT_TEMPLATE,
    RATIONALE_ONLY_PROMPT_TEMPLATE,
    VOTES_AND_RATIONALE_REVOTE_PROMPT_TEMPLATE,
    DEBATE_STATEMENT_PROMPT_TEMPLATE,
    DEBATE_REVOTE_PROMPT_TEMPLATE,
    DEBATE_REPLY_PROMPT_TEMPLATE,
    GUESS_MODELS_PROMPT_TEMPLATE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simulation")

# Shared across every SimulationRunner instance in this process (see its use in
# __init__/run()). Confirmed live: giving each concurrently-running job its own
# 12-worker pool produced 1400+ threads under a 100-job sweep and pathologically
# thrashed the GIL - the driver burned ~900% CPU while making zero actual
# requests to any agent server for over a minute. A bounded shared pool caps
# total thread count regardless of job concurrency. Sized comfortably above the
# 12-per-stage burst any single job needs, with enough multiple of that to keep
# several jobs' stages in flight at once without the pool itself becoming the
# bottleneck - not sized to the full 100-job concurrency, since the 12 vLLM
# servers' own admission control (max-num-seqs, KV cache) was already the real
# throughput limiter, not Python-side thread availability.
_AGENT_CALL_POOL = ThreadPoolExecutor(max_workers=150)


# ==============================================================================
# ======================== USER CONFIGURATION SECTION ==========================
# ==============================================================================
# Edit the variables below to customize your experiment runs.
# ==============================================================================

# --- 1. EXPERIMENT METADATA & DATASET SETTINGS ---
DATASET_NAME = "moralchoice"  # Options: "moralchoice", "moralmachine", "mmlu"
DATASET_SAMPLE_INDEX = 1      # Index of the dilemma in dataset (e.g., 0, 1, 2)
USE_HUGGINGFACE_ONLINE = False  # Set True to fetch live from HuggingFace Hub, False to use built-in cache
NUM_PROBLEMS_PER_DATASET = 50  # Stratified sample size pulled per dataset by dataset_loaders.get_dataset

# --- 2. IDENTITY CONDITION CONFIGURATION ---
# Options for IDENTITY_MODE:
#   1. "anonymous"              - Agents are simply Agent_01 .. Agent_12
#   2. "real_family_only"       - Displays model family (e.g., Qwen, LLaMA)
#   3. "real_size_only"         - Displays model size (e.g., 8B, 70B)
#   4. "real_family_and_size"   - Displays both family and size (e.g., LLaMA 70B)
#   5. "virtual_persona"        - Uses virtual human demographic personas
#   6. "delayed_reveal"         - Starts anonymous, reveals identity at stage DELAYED_REVEAL_STAGE
#   7. "gender_reveal"          - Displays only the agent's real system name (Agent_XX) plus its
#                                 assigned gender (Male=5, Female=4, Transgender=2 across 11 agents).
#                                 No model family/size, no persona name/age/ethnicity. No final
#                                 guess-the-model stage (that is anonymous-only).
VALID_IDENTITY_MODES = {
    "anonymous", "real_family_only", "real_size_only", "real_family_and_size",
    "virtual_persona", "delayed_reveal", "gender_reveal", "real_full_identity"
}
IDENTITY_MODE = "anonymous"

# If IDENTITY_MODE == "virtual_persona", select one of the 7 supported combinations.
# Only "name_gender_age_ethnicity" gets a real, demographically-matched name; every
# other combo uses a neutral placeholder name so it can't leak an undisclosed attribute.
#   "name_gender_age_ethnicity" | "name_gender" | "name_age" | "name_ethnicity" |
#   "name_gender_age" | "name_gender_ethnicity" | "name_age_ethnicity"
VIRTUAL_PERSONA_COMBO = FULL_PERSONA_COMBO

# Delayed Reveal Settings (only active if IDENTITY_MODE == "delayed_reveal"):
# Stage indices: 0=baseline, 1=bandwagon_revote, 2=rationale_collection,
#                3=reasoned_conformity_revote, 4..14=debate_round_1..11
DELAYED_REVEAL_STAGE = 4                      # Reveal identity at the start of debate_round_1
DELAYED_REVEAL_TARGET_MODE = "real_family_and_size"  # What identity is revealed into

# --- 3. SIMULATION STRUCTURE ---
NUM_DEBATE_ROUNDS = 11         # Number of full (statement -> reveal -> revote) debate cycles
TEMPERATURE = 0.0              # Sampling temperature for LLM responses (greedy - removes resampling noise from vote/flip measurements)
RETRY_TEMPERATURE = 0.0        # Temperature used on retry-after-parse-failure attempts in
                               # _generate_for_all_agents. Deliberately kept at 0 (not resampled
                               # at a higher temperature) for strict determinism across every
                               # attempt, not just the first. The real consequence: since greedy
                               # decoding is deterministic given identical input, a retry at the
                               # same temperature on the same prompt will almost always reproduce
                               # the identical failure rather than recovering - the existing
                               # fallback paths (auto-decline, rationale-only placeholder) are the
                               # actual safety net for this failure mode now, not the retry itself.
                               # (vLLM's continuous batching can occasionally perturb greedy output
                               # by a token via floating-point batching effects, so a retry is not
                               # strictly guaranteed to be byte-identical in practice - but this
                               # should not be relied on as a recovery mechanism.)
# Confirmed via repeated real 400 errors on this exact deployment: debate-stage
# prompts (round 1) consistently land around ~3185 input tokens regardless of
# identity condition - the dominant size driver is the LLM-generated rationale
# text carried in the votes/rationale table, not the identity labels. 400 was
# STILL exactly 1 token too many against the ORIGINAL MAX_MODEL_LEN=3584 - cutting
# with real margin instead of another marginal trim. MAX_MODEL_LEN is now 32768
# (see launch_servers.sh - raised for full cumulative debate-history context
# across all 11 rounds), so this specific 3584 ceiling no longer applies, but
# MAX_TOKENS is left at 300 regardless: it is an OUTPUT budget (how much the
# model itself is allowed to generate per call), unrelated to how large the
# INPUT context has grown - raising it doesn't need to track MAX_MODEL_LEN.
MAX_TOKENS = 300               # Maximum tokens for LLM generation

# Mirrors launch_servers.sh's --max-model-len - MUST be kept in sync with it.
# Used only for the proactive context-budget check below, not for anything
# serving-related.
MAX_MODEL_LEN = 32768
# Conservative chars-per-token estimate for the proactive pre-trim check below.
# Confirmed in production: Agent_10 (Mistral-7B-v0.3)'s tokenizer is the least
# efficient in the roster on this shared English debate text - plain English
# prose commonly runs ~4 chars/token, but this deliberately uses a pessimistic
# 3.2 so the proactive check trims BEFORE Agent_10 would actually overflow,
# not just for agents with more efficient tokenizers. Being overly conservative
# here only costs a slightly earlier trim (a few hundred tokens of old debate
# history Agent_10 would've had to drop anyway); underestimating costs a wasted
# failed request + reactive retry round-trip, which is the thing this exists
# to avoid.
CONTEXT_CHARS_PER_TOKEN = 3.2
CONTEXT_SAFETY_MARGIN_TOKENS = 200   # headroom against the estimate being wrong
# Conservative fixed estimate of everything in a debate-round prompt OTHER than
# the shared history string itself (system prompt, self-identity block, roster
# table or pointer note, question/options, the current votes+rationale table,
# and task instructions) - sized generously against the heaviest stage
# (revote, which includes the votes+rationale table on top of the history).
# Used by _uniform_trim_shared_history so the trim decision doesn't need to
# render each stage's exact template.
SHARED_HISTORY_OVERHEAD_CHARS = 9000
MODEL_GUESS_MAX_TOKENS = 6000   # The anonymous-condition guess-the-model stage needs one
                                # guess+rationale per OTHER agent (10 entries, not 11 - an
                                # agent never guesses itself) in a single response, not a
                                # single vote/statement - MAX_TOKENS=300 truncated every
                                # agent in a real test, 900 still truncated the more verbose
                                # models (e.g. Gemma 1B), and a real full run confirmed 1600
                                # still truncates Agent_08 (Gemma 4B) at 9/10 complete
                                # entries even after 3 retries (RuntimeError: "only had 9
                                # well-formed entries, needed at least 10"). Raised to 3200
                                # "with real margin" but a later real run still hit truncation
                                # on Agent_04 (Gemma 3 1B, even smaller/more verbose) - raised
                                # again to 6000 for real margin against the weakest model.

# --- 4. BACKEND & API PROVIDER CONFIGURATION ---
# Choose how to run the models:
#
# === CLOUD API PROVIDERS (Call via API Key) ===
#   "openrouter"        - Recommended! One API key gives access to Qwen, LLaMA, Gemma, Mistral.
#   "together"          - Together AI (Fast open-source model inference API)
#   "groq"              - Groq Cloud (Ultra high-speed open-weight inference)
#   "deepinfra"         - DeepInfra API
#   "openai"            - OpenAI API (or custom proxy)
#   "openai_compatible" - Any generic OpenAI-compatible API endpoint
#
# === LOCAL INFERENCE PROVIDERS (Run on your machine) ===
#   "ollama"            - Local Ollama server (http://localhost:11434)
#   "vllm"              - Local vLLM server (http://localhost:8000/v1)
#   "transformers"      - Direct local PyTorch / HuggingFace Transformers pipelines
#
# === DRY-RUN TESTING ===
#   "mock"              - Fast offline simulation (No GPU or API keys needed)
#
PROVIDER = "mock"
MODEL_NAME = None  # Optional: force all agents to use a specific model (e.g., "llama3.1:8b", "qwen2.5:7b")

# Unified API Key (Paste your API key here or set environment variables):
API_KEY = os.getenv("API_KEY", "")

# Specific API Keys (overrides API_KEY if set):
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", API_KEY)
TOGETHER_API_KEY   = os.getenv("TOGETHER_API_KEY", API_KEY)
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", API_KEY)
DEEPINFRA_API_KEY  = os.getenv("DEEPINFRA_API_KEY", API_KEY)
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", API_KEY)

# Custom Endpoints & URLs (if running local servers or proxies):
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL      = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
CUSTOM_BASE_URL    = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")

# Local Transformers Hardware Settings (for PROVIDER = "transformers"):
TRANSFORMERS_DEVICE = "auto"       # "auto", "cuda", "cpu"
TRANSFORMERS_LOAD_IN_4BIT = True   # Set True for 4-bit quantization (reduces VRAM usage significantly)
TRANSFORMERS_TORCH_DTYPE = "float16"  # "float16", "bfloat16", "float32"

# Output directory for experiment logs
LOG_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "experiments")

# ==============================================================================
# ==================== END OF USER CONFIGURATION SECTION =======================
# ==============================================================================


# Provider Endpoint Registry
PROVIDER_ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "openai": "https://api.openai.com/v1",
    "vllm": VLLM_BASE_URL,
    "openai_compatible": CUSTOM_BASE_URL,
    "ollama": OLLAMA_BASE_URL
}


class LLMBackend:
    """Unified LLM execution engine supporting Local Inference and Cloud APIs."""

    def __init__(
        self,
        provider: str = PROVIDER,
        api_key: Optional[str] = None,
        custom_base_url: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.base_url = custom_base_url or PROVIDER_ENDPOINTS.get(self.provider, CUSTOM_BASE_URL)

        # Cache for local transformers pipelines to avoid reloading models on every round
        self._transformers_cache: Dict[str, Any] = {}

    def _resolve_api_key(self, provider: str) -> str:
        """Resolves the appropriate API key based on provider."""
        if provider == "openrouter":
            return OPENROUTER_API_KEY
        elif provider == "together":
            return TOGETHER_API_KEY
        elif provider == "groq":
            return GROQ_API_KEY
        elif provider == "deepinfra":
            return DEEPINFRA_API_KEY
        elif provider == "openai":
            return OPENAI_API_KEY
        return API_KEY or "EMPTY"

    def generate(
        self,
        agent: AgentMetadata,
        system_prompt: str,
        user_prompt: str,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        round_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Dispatches generation to the appropriate local engine or API provider."""
        p = (agent.backend_type or self.provider).lower()

        if p == "mock":
            return self._mock_generate(agent, system_prompt, user_prompt, round_context)
        elif p == "ollama":
            return self._ollama_generate(agent, system_prompt, user_prompt, temperature)
        elif p == "transformers":
            return self._transformers_generate(agent, system_prompt, user_prompt, temperature, max_tokens)
        elif p in ["openrouter", "together", "groq", "deepinfra", "openai", "vllm", "openai_compatible"]:
            return self._openai_compatible_api_generate(agent, system_prompt, user_prompt, temperature, max_tokens)
        else:
            raise ValueError(
                f"Unrecognized provider '{p}' for {agent.agent_id}. This is a config error (typo in "
                f"--provider or --model backend_type?), not a transient failure - refusing to silently "
                f"substitute mock data for it."
            )

    @staticmethod
    def _drop_earliest_round_blocks(user_prompt: str, drop_count: int) -> Optional[str]:
        """
        Drops the earliest `drop_count` "#### Round N statements:" blocks from a
        debate-history-bearing prompt. Returns None if there are no round-blocks
        to drop (e.g. the baseline vote, which has no history at all) or if
        dropping this many would not change anything further.
        """
        parts = re.split(r'(?=#### Round \d+ statements:)', user_prompt)
        if len(parts) <= 1:
            return None
        keep_from = min(drop_count, len(parts) - 1)
        new_prompt = parts[0] + "".join(parts[1 + keep_from:])
        if new_prompt == user_prompt:
            return None
        return new_prompt

    def _uniform_trim_shared_history(self, history_text: str, max_tokens: int = MAX_TOKENS) -> str:
        """
        Decides ONE trim level for a debate-history string that gets embedded
        identically into every agent's prompt this round/step, rather than
        letting each agent's own HTTP call decide independently.

        Why this matters beyond just avoiding failures: context overflow in
        production hits ONLY Agent_10 (Mistral), whose tokenizer is less
        efficient on this shared text than the other 10 agents'. Trimming
        only Mistral's own copy (which is what a per-agent check would do)
        would leave it reasoning over strictly LESS debate history than every
        other agent sees for the identical round - an unfair, confounding
        asymmetry for any Mistral-specific or family-level analysis, not
        merely a context-length workaround. This function trims the shared
        string ONCE, so whatever the outcome, every agent (including the 10
        that didn't personally need it) sees the exact same history.

        Uses a deliberately generous fixed overhead estimate
        (SHARED_HISTORY_OVERHEAD_CHARS) for everything else in the prompt
        rather than rendering each stage's real template, so the same trim
        level is valid regardless of which stage (statement/reply/revote)
        ends up using the result.
        """
        estimated_tokens = (SHARED_HISTORY_OVERHEAD_CHARS + len(history_text)) / CONTEXT_CHARS_PER_TOKEN
        budget = MAX_MODEL_LEN - max_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
        if estimated_tokens <= budget:
            return history_text
        trimmed = history_text
        for drop_count in (2, 4, 8, 16, 32):
            candidate = self._drop_earliest_round_blocks(trimmed, drop_count)
            if candidate is None:
                break
            trimmed = candidate
            new_estimate = (SHARED_HISTORY_OVERHEAD_CHARS + len(trimmed)) / CONTEXT_CHARS_PER_TOKEN
            if new_estimate <= budget:
                break
        if trimmed != history_text:
            logger.warning(
                f"[uniform trim] shared debate history estimated at ~{estimated_tokens:.0f} tokens "
                f"(budget {budget}); dropped early round-block(s) for ALL agents uniformly, so no "
                f"agent sees more history than another this round."
            )
        return trimmed

    def _openai_compatible_api_generate(
        self,
        agent: AgentMetadata,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Executes HTTP requests to any OpenAI-compatible API endpoint (OpenRouter, Together, Groq, vLLM, OpenAI).
        Uses standard urllib to eliminate mandatory external package requirements.
        """
        endpoint_url = agent.endpoint_url or self.base_url
        if not endpoint_url.endswith("/chat/completions"):
            endpoint_url = f"{endpoint_url.rstrip('/')}/chat/completions"

        key = agent.api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }

        # OpenRouter specific attribution headers
        if "openrouter" in self.provider:
            headers["HTTP-Referer"] = "https://github.com/google-deepmind/democratic-llms"
            headers["X-Title"] = "Democratic Multi-Agent Deliberation Experiment"

        # PROACTIVE context-budget check: estimate whether this prompt is likely
        # to overflow MAX_MODEL_LEN before ever sending it, and pre-trim if so.
        # This exists purely to avoid the reactive path below's wasted
        # failed-request + retry round-trip (every avoided overflow saves one
        # full failed generation's wall-clock time, not just a failure count).
        # The reactive path remains below as a safety net in case this estimate
        # is wrong in either direction.
        #
        # Forcing temperature=0.0 for any trimmed call (proactive or reactive)
        # is deliberate: this is a context-content change, not a resampling
        # recovery - there is no reason to introduce sampling noise here, and
        # doing so would undermine the greedy-decoding guarantee for the one
        # agent (Mistral, the only one whose tokenizer is inefficient enough to
        # ever reach this path) more often than any other agent in the roster.
        was_proactively_trimmed = False
        estimated_tokens = (len(system_prompt) + len(user_prompt)) / CONTEXT_CHARS_PER_TOKEN
        budget = MAX_MODEL_LEN - max_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
        if estimated_tokens > budget:
            trimmed = user_prompt
            for drop_count in (2, 4, 8, 16, 32):
                candidate = self._drop_earliest_round_blocks(trimmed, drop_count)
                if candidate is None:
                    break
                trimmed = candidate
                was_proactively_trimmed = True
                new_estimate = (len(system_prompt) + len(trimmed)) / CONTEXT_CHARS_PER_TOKEN
                if new_estimate <= budget:
                    break
            if was_proactively_trimmed:
                logger.warning(
                    f"[{agent.agent_id}] prompt estimated at ~{estimated_tokens:.0f} tokens "
                    f"(budget {budget}); proactively dropped early debate-history round-block(s) "
                    f"before sending, at temperature 0.0."
                )
                user_prompt = trimmed

        # Some chat templates in this roster (Gemma's, confirmed via a real 400
        # "System role not supported" error on this exact deployment) reject a
        # separate system-role message outright. Folding it into the user turn
        # instead works uniformly across every model family here and carries the
        # same information, so there is no need to special-case by agent/family.
        payload = {
            "model": agent.model_name,
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
            "temperature": 0.0 if was_proactively_trimmed else temperature,
            "max_tokens": max_tokens
        }

        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            # Confirmed on a real run: with 100 concurrent simulations all
            # queuing onto the same 12 servers (each capped at 2 concurrent
            # sequences by --max-num-seqs, a memory-driven limit), the initial
            # synchronized burst produces genuine queue waits well past 90s -
            # a slow-but-healthy server, not a hung one. 300s gives real
            # requests room to survive the queue instead of failing needlessly.
            #
            # Raised 300 -> 1800: with the 11-round full-debate-history design,
            # the 70B agent's per-round queue depth (13-14 running + up to 26
            # waiting, confirmed via its own vLLM 'Running/Waiting' log lines)
            # compounds every round as prompts grow, and two real jobs timed
            # out at 300s well before round 11 (moralchoice_0014, _0692) with
            # no server crash - genuine queue wait, not a hang. A dead/hung
            # server is caught independently by the liveness watchdog polling
            # /v1/models every 2 min, so this can afford to be generous.
            with urllib.request.urlopen(req, timeout=1800) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # urllib discards the response body by default, which was hiding the
            # actual reason for 400s (e.g. context-length overflow) behind a bare
            # "HTTP Error 400: Bad Request" - read it explicitly so failures are
            # actually diagnosable instead of guessed at.
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<could not read response body>"

            # Context-length overflow: confirmed in production to hit ONLY
            # Agent_10 (Mistral-7B-v0.3), never the other 10 agents, despite
            # every agent receiving the identical shared prompt string at a
            # given point in the debate - Mistral's own tokenizer is simply
            # less token-efficient per character than the other agents' for
            # this English debate text, so it alone crosses MAX_MODEL_LEN=32768
            # in the later rounds where "FULL DEBATE HISTORY" has grown large.
            # Rather than failing the whole question over this one agent's
            # tokenizer being less efficient, progressively drop the EARLIEST
            # "#### Round N statements:" blocks from the debate-history section
            # of user_prompt and retry - the agent still sees every recent
            # round plus the current dilemma/votes/instructions, just not the
            # oldest rounds, which matter far less to a re-vote/statement than
            # the most recent context does. A prompt with no such round-block
            # markers (e.g. the baseline vote, which has no history at all)
            # is untouched by this loop and falls straight through to the
            # original raise below, unchanged from before this fix.
            if e.code == 400 and "maximum context length" in body:
                truncated_prompt = user_prompt
                for drop_count in (2, 4, 8, 16, 32):
                    candidate = self._drop_earliest_round_blocks(truncated_prompt, drop_count)
                    if candidate is None:
                        break  # no round-blocks left (or none present) to drop, or no further change
                    truncated_prompt = candidate
                    # Forced to 0.0 regardless of the original `temperature` argument -
                    # see the proactive check above for why.
                    retry_payload = dict(payload)
                    retry_payload["temperature"] = 0.0
                    retry_payload["messages"] = [
                        {"role": "user", "content": f"{system_prompt}\n\n{truncated_prompt}"}
                    ]
                    try:
                        req2 = urllib.request.Request(
                            endpoint_url,
                            data=json.dumps(retry_payload).encode("utf-8"),
                            headers=headers,
                            method="POST"
                        )
                        with urllib.request.urlopen(req2, timeout=1800) as resp2:
                            data2 = json.loads(resp2.read().decode("utf-8"))
                            logger.warning(
                                f"[{agent.agent_id}] prompt exceeded context length "
                                f"({body[:200]!r}); retried with the earliest {keep_from} "
                                f"debate-history round-block(s) dropped and succeeded."
                            )
                            return data2["choices"][0]["message"]["content"]
                    except urllib.error.HTTPError as e2:
                        try:
                            body2 = e2.read().decode("utf-8", errors="replace")
                        except Exception:
                            body2 = "<could not read response body>"
                        if e2.code != 400 or "maximum context length" not in body2:
                            break  # a different failure now - stop and fall through to raise
                        continue  # still too long even with this much dropped, try dropping more

            raise RuntimeError(
                f"API call failed for {agent.agent_id} (Model: {agent.model_name}) via {self.provider} at {endpoint_url}: {e} | body={body}"
            ) from e
        except Exception as e:
            # Deliberately NOT falling back to mock here: during a real multi-hour
            # sweep, a dead/crashed server should fail this one job loudly (it gets
            # logged and retried on the next sweep run - see run_full_experiment.py's
            # resumability) rather than silently substituting fake data that would
            # look like a legitimate result in the output files.
            raise RuntimeError(
                f"API call failed for {agent.agent_id} (Model: {agent.model_name}) via {self.provider} at {endpoint_url}: {e}"
            ) from e

    def _ollama_generate(
        self,
        agent: AgentMetadata,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """Executes generation against a local Ollama server."""
        url = f"{self.base_url.rstrip('/')}/api/chat"
        # See _openai_compatible_api_generate: some chat templates in this roster
        # (Gemma's) reject a separate system-role message, so it is folded into
        # the user turn uniformly instead of special-casing by model family.
        payload = {
            "model": agent.model_name,
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
            "stream": False,
            "options": {"temperature": temperature}
        }
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Local Ollama call failed for {agent.agent_id} ({agent.model_name}) at {url}: {e}") from e

    def _transformers_generate(
        self,
        agent: AgentMetadata,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Executes generation directly on local GPU/CPU using Hugging Face Transformers."""
        model_name = agent.model_name
        try:
            import torch
            from transformers import pipeline

            if model_name not in self._transformers_cache:
                logger.info(f"Loading local transformer model into memory: {model_name}...")
                kwargs: Dict[str, Any] = {
                    "device_map": TRANSFORMERS_DEVICE,
                }
                if TRANSFORMERS_TORCH_DTYPE == "float16":
                    kwargs["torch_dtype"] = torch.float16
                elif TRANSFORMERS_TORCH_DTYPE == "bfloat16":
                    kwargs["torch_dtype"] = torch.bfloat16

                if TRANSFORMERS_LOAD_IN_4BIT and torch.cuda.is_available():
                    kwargs["model_kwargs"] = {"load_in_4bit": True}

                pipe = pipeline("text-generation", model=model_name, **kwargs)
                self._transformers_cache[model_name] = pipe

            pipe = self._transformers_cache[model_name]
            # See _openai_compatible_api_generate: some chat templates in this
            # roster (Gemma's) reject a separate system-role message, so it is
            # folded into the user turn uniformly instead of special-casing by
            # model family.
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
            out = pipe(messages, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
            return out[0]["generated_text"][-1]["content"]
        except Exception as e:
            raise RuntimeError(f"Transformers pipeline failed for {agent.agent_id} ({model_name}): {e}") from e

    def _mock_generate(
        self,
        agent: AgentMetadata,
        system_prompt: str,
        user_prompt: str,
        round_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Simulates plausible deliberation output for every stage type, without consuming
        API tokens or requiring a GPU. Used for --dry-run and as an automatic fallback
        when a real backend call fails.
        """
        if "INDEPENDENT VOTE" in user_prompt:
            prompt_type = "baseline"
        elif "STATE YOUR REASONING" in user_prompt:
            prompt_type = "rationale_only"
        elif "REPLY DECISION" in user_prompt:
            return json.dumps({"decision": random.choice(["reply", "decline"]),
                               "text": "I see your point, but my reasoning on the core trade-off still stands."}, indent=2)
        elif "YOUR STATEMENT" in user_prompt:
            prompt_type = "debate_statement"
        elif "RE-VOTE" in user_prompt and "STATEMENTS EXCHANGED THIS ROUND" in user_prompt:
            prompt_type = "debate_revote"
        elif "HOW EACH PARTICIPANT HAS VOTED SO FAR" in user_prompt:
            prompt_type = "votes_only_revote"
        elif "IDENTIFY THE OTHER PARTICIPANTS" in user_prompt:
            prompt_type = "guess_models"
        else:
            prompt_type = "votes_and_rationale_revote"

        opt_a = round_context.get("option_a", "Option A") if round_context else "Option A"
        opt_b = round_context.get("option_b", "Option B") if round_context else "Option B"

        # Diverse philosophical frameworks for high-quality synthetic reasoning
        frameworks_a = [
            "Choosing Option A is ethically imperative because it prioritizes direct harm minimization and upholds the non-negotiable principle of protecting vulnerable lives. While Option B offers alternative utility, it introduces unacceptable systemic risks and violates fundamental duty.",
            "A rigorous consequentialist calculus indicates that Option A results in significantly fewer net casualties and avoids irreversible collateral damage. Option B relies on speculative secondary benefits that do not outweigh the immediate, tangible harms.",
            "Under a deontological framework, Option A fulfills the primary moral obligation without treating individuals merely as a means to an end. Option B compromises fundamental ethical boundaries for aggregate convenience.",
            "Applying the precautionary principle, Option A preserves essential safeguards and guarantees verifiable accountability. The trade-offs in Option B create dangerous precedents for institutional and moral degradation."
        ]

        frameworks_b = [
            "Option B is the superior choice because it maximizes collective well-being and preserves long-term systemic stability. Although Option A addresses immediate surface-level concerns, it exacerbates structural vulnerabilities and causes greater aggregate harm over time.",
            "From a rule-utilitarian standpoint, Option B establishes the most sustainable rule of action, ensuring fairness and equitable distribution of risk. Option A represents a short-sighted deferral of necessary ethical responsibility.",
            "A comprehensive risk-benefit assessment reveals that Option B prevents catastrophic systemic failure. The acute trade-offs are outweighed by the profound preservation of broader societal and moral order compared to Option A.",
            "Option B respects the principle of distributive justice by preventing disproportionate burden from falling on innocent stakeholders. Option A fails to account for the broader causal chain and long-term societal repercussions."
        ]

        if prompt_type == "baseline":
            seed_val = hash(agent.agent_id + agent.family) % 10
            vote = "A" if seed_val < 6 else "B"
            conf = round(0.58 + (agent.size_billions / 70.0) * 0.32 + random.uniform(-0.04, 0.04), 2)
            conf = round(max(0.50, min(0.98, conf)), 2)
            return json.dumps({"vote": vote, "confidence": conf}, indent=2)

        elif prompt_type == "rationale_only":
            current_vote = (round_context.get("current_vote", "A") if round_context else "A").upper()
            rationale = random.choice(frameworks_a if current_vote == "A" else frameworks_b)
            return json.dumps({"rationale": rationale}, indent=2)

        elif prompt_type == "debate_statement":
            candidate_targets = (round_context.get("candidate_target_ids") if round_context else None) or ["Agent_01"]
            target_id = random.choice(candidate_targets)
            critiques = [
                f"To [{target_id}]: I challenge the ethical foundation of your position. Prioritizing short-term outcomes over core deontological obligations introduces catastrophic long-term risks that your analysis currently overlooks.",
                f"To [{target_id}]: Your argument fails to account for the irreversible asymmetry of harm in this scenario. We must evaluate whether the speculative gains of your stance genuinely outweigh the direct, quantifiable damages.",
                f"To [{target_id}]: Consider the systemic precedent your vote establishes. If every autonomous decision-maker adopted your reasoning, institutional trust and fundamental rights would erode under crisis conditions.",
                f"To [{target_id}]: While your concern for immediate duty is understandable, a strict consequentialist evaluation demonstrates that your chosen course generates greater aggregate suffering. We must weigh the full causal chain.",
                f"To [{target_id}]: Have you considered the distributive justice implications of your stance? The burdens of your proposed action fall disproportionately on vulnerable parties who have no voice in this assembly."
            ]
            return json.dumps({"target_agent_id": target_id, "statement": random.choice(critiques)}, indent=2)

        elif prompt_type == "guess_models":
            other_ids = (round_context.get("other_agent_ids") if round_context else None) or []
            fake_models = ["GPT-4", "Claude 3.5", "Llama 3", "Gemini 1.5", "Mistral Large", "Qwen 2.5", "Command R+"]
            guesses = {
                aid: {
                    "model_guess": random.choice(fake_models),
                    "rationale": "Based on its reasoning style, confidence calibration, and argument patterns observed across the debate."
                }
                for aid in other_ids
            }
            return json.dumps({"guesses": guesses}, indent=2)

        elif prompt_type == "votes_only_revote":
            prev_vote = (round_context.get("previous_vote", "A") if round_context else "A").upper()
            prev_conf = float(round_context.get("previous_confidence", 0.70)) if round_context else 0.70
            if prev_conf > 1.0:
                prev_conf = prev_conf / 100.0

            # Smaller models exhibit higher malleability to bare majority pressure
            flip_prob = 0.20 if agent.size_billions <= 4.0 else 0.04
            flipped = random.random() < flip_prob
            new_vote = ("B" if prev_vote == "A" else "A") if flipped else prev_vote
            if flipped:
                new_conf = round(max(0.50, min(0.90, prev_conf - 0.05 + random.uniform(0.0, 0.10))), 2)
            else:
                new_conf = round(max(0.50, min(0.99, prev_conf + random.uniform(-0.02, 0.05))), 2)
            return json.dumps({"vote": new_vote, "confidence": new_conf, "flipped": flipped}, indent=2)

        else:  # "votes_and_rationale_revote" or "debate_revote"
            prev_vote = (round_context.get("previous_vote", "A") if round_context else "A").upper()
            prev_conf = float(round_context.get("previous_confidence", 0.70)) if round_context else 0.70
            if prev_conf > 1.0:
                prev_conf = prev_conf / 100.0

            # Smaller models exhibit higher malleability to persuasive group debate
            flip_prob = 0.28 if agent.size_billions <= 4.0 else 0.06
            flipped = random.random() < flip_prob
            new_vote = ("B" if prev_vote == "A" else "A") if flipped else prev_vote

            if flipped:
                new_conf = round(max(0.50, min(0.92, prev_conf - 0.05 + random.uniform(0.0, 0.12))), 2)
                if new_vote == "A":
                    rationale = random.choice([
                        "After carefully deliberating on the arguments regarding irreversible harm and deontological duty, I recognize that my previous stance underestimated the acute moral costs and systemic risks. Option A provides the only robust defense against catastrophic failure and upholds our primary ethical obligation. Consequently, I have revised my vote to Option A.",
                        "The reasoning I read regarding direct moral responsibility was compelling. My prior vote relied on secondary utilitarian trade-offs that fail under close scrutiny when compared against the immediate duty of harm prevention in Option A. I have therefore adjusted my decision to Option A.",
                        "Re-evaluating the systemic ramifications highlighted here demonstrated that the alternative introduces unmanageable moral hazards. Option A establishes a necessary boundary that protects vulnerable stakeholders, compelling me to change my vote to Option A."
                    ])
                else:
                    rationale = random.choice([
                        "The reasoning I read exposed critical vulnerabilities in my initial adherence to Option A. When evaluating the full causal chain and aggregate outcomes, Option B clearly achieves superior long-term stability and minimizes widespread systemic harm. I have shifted my vote to Option B.",
                        "Engaging with the counter-arguments regarding distributive justice made it clear that Option A creates disproportionate burdens that cannot be justified on principle. Option B offers a balanced, equitable framework, leading me to change my vote to Option B.",
                        "While I initially prioritized immediate deontological duty in Option A, the analysis of catastrophic long-term failure convinced me that adhering to Option A is untenable. Option B represents the only sustainable resolution, so I change my vote to Option B."
                    ])
            else:
                new_conf = round(max(0.55, min(0.99, prev_conf + random.uniform(-0.02, 0.06))), 2)
                if new_vote == "A":
                    rationale = random.choice([
                        "Having critically examined the counter-arguments presented, I find that they rely on speculative utilitarian assumptions that do not supersede the fundamental duty of harm reduction inherent in Option A. I therefore reaffirm my vote for Option A with reinforced confidence.",
                        "Nothing I read provided sufficient justification to violate the deontological boundaries upheld by Option A. Sacrificing fundamental rights for hypothetical aggregate gains would set an unacceptable precedent. I maintain my vote for Option A.",
                        "None of the points raised in favor of Option B adequately address the irreversibility of the harms at stake. Option A remains the only principled stance that satisfies the precautionary requirement, so I firmly sustain my position on Option A."
                    ])
                else:
                    rationale = random.choice([
                        "While compelling points regarding immediate obligations were raised, they do not resolve the severe systemic breakdown that Option A would precipitate over the long horizon. Option B remains the only logically coherent and ethically sustainable solution. I steadfastly maintain my vote for Option B.",
                        "The counter-arguments focus narrowly on immediate symptoms rather than root causal dynamics. Option B addresses the comprehensive structural challenge and minimizes total expected suffering across all affected parties. I reaffirm my vote for Option B.",
                        "This reinforces my conclusion that Option B provides the optimal balance of fairness and aggregate utility. The objections to Option B fail to offer a workable alternative that avoids widespread harm, so I steadfastly retain my vote for Option B."
                    ])

            return json.dumps({
                "vote": new_vote,
                "confidence": new_conf,
                "flipped": flipped,
                "rationale": rationale
            }, indent=2)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust parser extracting valid JSON from raw LLM output strings.
    Deliberately raises rather than returning a fallback dict on failure: every
    caller does `parsed.get("vote", "A")`-style defaulting with no check for a
    parse-failure marker, so a silent fallback here would have recorded a fake
    default vote/confidence as if it were a genuine model response - the exact
    kind of undetectable data corruption this experiment cannot tolerate. A
    parse failure (e.g. a response truncated by hitting max_tokens) must fail
    the job loudly so it gets retried, not get treated as real data.
    """
    text = text.strip()
    # Fix one very specific, unambiguous model mistake confirmed twice on a real
    # run: leaving an Agent_XX identifier value unquoted, e.g.
    # `"target_agent_id": Agent_07,`. An Agent ID is never anything else in
    # this domain, so this repair is narrow and safe - it is not general-
    # purpose JSON repair, just this one exact recurring shape.
    text = re.sub(r'("target_agent_id"\s*:\s*)(Agent_\d+)\b', r'\1"\2"', text)
    if "```json" in text:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    elif "```" in text:
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx:end_idx + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return _salvage_truncated_json(text)


def _salvage_truncated_json(text: str) -> Dict[str, Any]:
    """
    Last-resort recovery for a response that is genuine model output cut off by
    hitting MAX_TOKENS mid-string (confirmed on real runs: e.g. the statement
    text ends mid-word with no closing quote/brace at all, or the text finishes
    the string cleanly but runs out of budget before the final "}"). Every
    structured field in these schemas (vote/confidence/flipped/target_agent_id)
    is short and always precedes the free-text field (rationale/statement),
    which is the one that runs out of room - so this recovers the structured
    fields via targeted regexes and trims the free-text field to its last
    complete sentence or word. This only ever removes characters from genuine
    model output; it never invents content, preserving the "no fabricated
    fallback" guarantee above. Raises if the expected fields for either a
    revote record or a statement record can't all be recovered, so a response
    that failed for some other reason still fails loudly.
    """
    stripped = text.strip()
    stripped = re.sub(r'^```(?:json)?\s*', '', stripped)

    recovered: Dict[str, Any] = {}

    vote_match = re.search(r'"vote"\s*:\s*"?([AB])"?', stripped)
    if vote_match:
        recovered["vote"] = vote_match.group(1)

    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', stripped)
    if conf_match:
        recovered["confidence"] = float(conf_match.group(1))

    flipped_match = re.search(r'"flipped"\s*:\s*(true|false)', stripped)
    if flipped_match:
        recovered["flipped"] = flipped_match.group(1) == "true"

    target_match = re.search(r'"target_agent_id"\s*:\s*"(Agent_\d+)"', stripped)
    if target_match:
        recovered["target_agent_id"] = target_match.group(1)

    for field in ("rationale", "statement"):
        field_match = re.search(rf'"{field}"\s*:\s*"(.*)', stripped, re.DOTALL)
        if not field_match:
            continue
        raw_value = re.sub(r'"\s*[}\]]*\s*```?\s*$', '', field_match.group(1)).strip()
        if raw_value and raw_value[-1] in ".!?":
            trimmed = raw_value
        else:
            last_sentence_end = max(
                raw_value.rfind(". "), raw_value.rfind(".\n"),
                raw_value.rfind("! "), raw_value.rfind("? "),
            )
            if last_sentence_end >= 20:
                trimmed = raw_value[:last_sentence_end + 1].strip()
            else:
                last_space = raw_value.rfind(" ")
                trimmed = raw_value[:last_space].strip() if last_space >= 20 else raw_value
        if len(trimmed) >= 15:
            recovered[field] = trimmed
        break

    has_revote_shape = all(k in recovered for k in ("vote", "confidence", "flipped", "rationale"))
    has_statement_shape = all(k in recovered for k in ("target_agent_id", "statement"))
    # Bare vote-only shape (baseline/bandwagon stages: just vote+confidence, no
    # flipped/rationale in the schema at all). Confirmed real failure mode: a
    # model can emit a genuinely complete, well-formed "vote"/"confidence" pair
    # and then stop via its own end-of-turn token before the closing "}" -
    # finish_reason="stop" with tokens to spare, not a length cutoff - so this
    # isn't length-truncated free text needing trimming, just a missing brace
    # around otherwise-complete, genuine field values.
    has_vote_only_shape = (
        all(k in recovered for k in ("vote", "confidence"))
        and not has_revote_shape and not has_statement_shape
    )

    if not (has_revote_shape or has_statement_shape or has_vote_only_shape):
        raise RuntimeError(
            f"Failed to parse JSON from model output (likely truncated by hitting "
            f"max_tokens, or a malformed response), and salvage recovery could not "
            f"reconstruct a complete revote, statement, or vote-only record: {text[:300]!r}"
        )

    shape = "revote" if has_revote_shape else ("statement" if has_statement_shape else "vote-only")
    logging.warning(
        f"Recovered a truncated {shape} response (missing closing brace/fields): {text[:120]!r}..."
    )
    return recovered


def _validate_vote_field(parsed: Dict[str, Any], agent: Optional["AgentMetadata"] = None) -> None:
    """
    Confirms the parsed JSON actually contains a valid "A"/"B" vote. Every
    vote-producing call site used to silently default an out-of-range value
    (e.g. a model answering "Undecided", "Option A", or omitting the field
    entirely) to "A" with no record that this was a fabricated fallback, not a
    real model choice - a systematic "A" bias baked directly into the data,
    and a real inconsistency with extract_json_from_text's own refusal to
    fabricate on a parse failure. Raising here routes it through the same
    fresh-sample retry path as an unparseable response instead.
    """
    vote = str(parsed.get("vote", "")).strip().upper()
    if vote not in ("A", "B"):
        raise ValueError(f"Model returned a non-A/B vote value: {parsed.get('vote')!r}")


def _looks_like_leaked_json(text: str) -> bool:
    """
    Detects a free-text field that is actually a raw, un-extracted model
    response - e.g. the model misspelled the "rationale"/"statement" key
    (observed in practice: "ratiole", "rationationale"), so extract_json_from_text
    couldn't find the real field and a naive fallback used the whole raw text
    instead. Recognizable by leftover code-fence markers or the field starting
    with what looks like the outer JSON object itself.
    """
    stripped = text.strip()
    return "```" in stripped or stripped.startswith("{") or '"vote"' in stripped or '"target_agent_id"' in stripped


def _validate_rationale_field(parsed: Dict[str, Any], agent: Optional["AgentMetadata"] = None) -> None:
    """
    Confirms "rationale" is real free text, not a leaked raw JSON blob. See
    _looks_like_leaked_json - this closes the gap where a misspelled key name
    let corrupted rationale text silently pass every other check (vote and
    tally were still fine, only the free-text explanation was garbage).
    """
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"Model returned an empty/missing rationale: {parsed.get('rationale')!r}")
    if _looks_like_leaked_json(rationale):
        raise ValueError(f"Model's rationale field looks like leaked raw JSON: {rationale[:120]!r}")


def _validate_vote_and_rationale_field(parsed: Dict[str, Any], agent: Optional["AgentMetadata"] = None) -> None:
    """Combines _validate_vote_field and _validate_rationale_field for revote stages that carry both."""
    _validate_vote_field(parsed)
    _validate_rationale_field(parsed)


def _validate_debate_statement_field(parsed: Dict[str, Any], agent: Optional["AgentMetadata"] = None,
                                     valid_ids: Optional[List[str]] = None) -> None:
    """Same principle as _validate_vote_field, applied to the debate-statement phase's fields."""
    statement = parsed.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError(f"Model returned an empty/missing debate statement: {parsed.get('statement')!r}")
    if _looks_like_leaked_json(statement):
        raise ValueError(f"Model's statement field looks like leaked raw JSON: {statement[:120]!r}")
    target = parsed.get("target_agent_id")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Model returned an empty/missing target_agent_id: {parsed.get('target_agent_id')!r}")
    # NOTE: an invalid or self-targeting target_agent_id is deliberately NOT validated
    # here (even though valid_ids is available) - the caller (the statement-processing
    # loop right after _generate_for_all_agents) already salvages this by falling back
    # to a random *other* agent (target_candidates excludes the speaker), exactly as the
    # pre-reply-phase code always did. Raising here instead would burn all retries on a
    # single targeting slip and fail the entire question - confirmed happening in
    # production (Agent_04/Gemma-3-1B self-targeted repeatedly and exhausted retries).


def _validate_reply_field(parsed: Dict[str, Any], agent: Optional["AgentMetadata"] = None) -> None:
    """Reply-decision phase: decision must be 'reply' or 'decline', with non-empty text."""
    decision = parsed.get("decision")
    if not isinstance(decision, str) or decision.strip().lower() not in ("reply", "decline"):
        raise ValueError(f"Model returned an invalid reply decision: {decision!r}")
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Model returned an empty/missing reply text or decline reason: {text!r}")
    if _looks_like_leaked_json(text):
        raise ValueError(f"Model's reply text looks like leaked raw JSON: {text[:120]!r}")


def _format_statement_line(stmt: Dict[str, Any]) -> str:
    """One debate message as shown to agents: sender (with its vote at sending time) -> target: text."""
    vote = stmt.get("speaker_vote_at_send")
    vote_part = f" (current vote: {vote})" if vote else ""
    return f"- [msg #{stmt['statement_idx']}] {stmt['speaker_display_identity']}{vote_part} -> {stmt['target_display_identity']}: {stmt['statement']}"


def _format_reply_line(rep_: Dict[str, Any]) -> str:
    """One reply/decline as shown to agents."""
    if rep_["decision"] == "reply":
        return f"- {rep_['replier_display_identity']} REPLIED to {rep_['sender_display_identity']} (msg #{rep_['message_idx']}): {rep_['text']}"
    return f"- {rep_['replier_display_identity']} DECLINED to reply to {rep_['sender_display_identity']} (msg #{rep_['message_idx']}) - reason: {rep_['text']}"


def _validate_model_guesses_field(parsed: Dict[str, Any], agent: "AgentMetadata", all_agent_ids: List[str]) -> None:
    """
    Confirms the anonymous-condition final guess-the-model stage got a real,
    well-formed guess+rationale for EVERY other real roster agent (not just
    `len(all_agent_ids) - 1` entries of SOME shape).

    Fixed a real data-integrity bug found on a live run: the previous version
    only checked that enough dict keys matched the bare regex `Agent_\\d+$`
    and were well-formed - it never checked those keys were the ACTUAL other
    agents. A model that hallucinated/duplicated a wrong-but-plausible-shaped
    key could hit the count threshold while genuinely omitting a required
    agent, and the omitted agent's entry was then silently defaulted to an
    empty model_guess/rationale by execute_model_identification_stage's own
    reconstruction step (which does `guesses.get(other.agent_id, {})`) -
    confirmed on experiments/moralchoice/moralchoice_0697_anonymous.json,
    where Agent_04's saved guess for Agent_11 was '' / '' despite the job
    completing without ever raising a validation error. Now checks the exact
    required ID set per agent (every real roster ID except the agent's own),
    so a missing/wrong-ID response is retried instead of silently accepted.
    """
    guesses = parsed.get("guesses")
    if not isinstance(guesses, dict):
        raise ValueError(f"Model returned no/invalid 'guesses' object: {type(guesses).__name__}")
    required_ids = [aid for aid in all_agent_ids if aid != agent.agent_id]
    missing_or_bad = []
    for aid in required_ids:
        entry = guesses.get(aid)
        if not isinstance(entry, dict):
            missing_or_bad.append(aid)
            continue
        guess_text = entry.get("model_guess")
        rationale_text = entry.get("rationale")
        if not (isinstance(guess_text, str) and guess_text.strip() and isinstance(rationale_text, str) and rationale_text.strip()):
            missing_or_bad.append(aid)
    if missing_or_bad:
        raise ValueError(
            f"Model's guesses are missing/malformed entries for required agent(s) {missing_or_bad}: {guesses!r}"
        )


def compute_output_label(
    identity_mode: str,
    delayed_reveal_target: Optional[str] = None,
    virtual_combo: Optional[str] = None
) -> str:
    """
    The filename-safe label for a run's output files. identity_mode alone is not
    enough to distinguish every condition:
      - "delayed_reveal" runs with different reveal targets on the same problem
        would otherwise overwrite each other's file.
      - "virtual_persona" runs with different persona combos (there are 7) would
        ALSO otherwise overwrite each other's file - each combo is a distinct
        experimental condition, not a variant of the same one.
    Exposed standalone (not just inside SimulationRunner) so a driver can check
    whether a job's output already exists before constructing a runner for it.
    """
    if identity_mode == "delayed_reveal":
        target_slug = str(delayed_reveal_target or "").lower().replace(" ", "_")
        return f"delayed_reveal_to_{target_slug}"
    if identity_mode == "virtual_persona":
        combo_slug = str(virtual_combo or FULL_PERSONA_COMBO).lower()
        return f"virtual_persona_{combo_slug}"
    return identity_mode


class SimulationRunner:
    """
    Simulation controller for a single problem statement.
    Guarantees a completely fresh group chat transcript and independent state.

    Stage pipeline: baseline -> bandwagon_revote -> rationale_collection ->
    reasoned_conformity_revote -> debate_round_1 .. debate_round_N.
    """

    def __init__(
        self,
        question_item: Dict[str, Any],
        identity_mode: str = IDENTITY_MODE,
        virtual_combo: str = VIRTUAL_PERSONA_COMBO,
        delayed_reveal: Optional[bool] = None,
        delayed_reveal_stage: int = DELAYED_REVEAL_STAGE,
        delayed_reveal_target: str = DELAYED_REVEAL_TARGET_MODE,
        num_debate_rounds: int = NUM_DEBATE_ROUNDS,
        provider: str = PROVIDER,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_base_url: Optional[str] = None,
        output_root: str = LOG_OUTPUT_ROOT
    ):
        # 1. Fresh Problem Statement
        self.question_item = question_item
        self.dataset_name = question_item.get("dataset", "custom").split("/")[-1]
        self.problem_id = question_item["id"].replace("/", "_")

        # 2. Identity & Simulation Parameters
        # Confirmed live: an unrecognized identity_mode (e.g. a typo'd
        # --identity-mode on a manual single-run invocation) used to complete
        # silently instead of erroring - format_agent_display_identity fell
        # through to showing bare "Agent_01" (functionally anonymous) while
        # ConversationLogger's metadata branching fell through to labeling it
        # "real_identity (<the typo>)" / disclosure_type "complete_reveal",
        # producing a file whose own metadata contradicts what agents actually
        # saw. run_full_experiment.py's real sweep builds its own hardcoded,
        # already-valid ALL_IDENTITY_CONDITIONS list so this never reaches the
        # real GPU sweep, but a manual run_experiment.py invocation had no
        # safety net at all - fail loudly instead.
        if identity_mode not in VALID_IDENTITY_MODES:
            raise ValueError(
                f"Unrecognized identity_mode {identity_mode!r}. Expected one of: "
                f"{sorted(VALID_IDENTITY_MODES)}"
            )
        self.identity_mode = identity_mode
        self.virtual_combo = virtual_combo
        self.delayed_reveal = (identity_mode == "delayed_reveal") if delayed_reveal is None else delayed_reveal
        self.delayed_reveal_stage = delayed_reveal_stage
        self.delayed_reveal_target = delayed_reveal_target
        self.num_debate_rounds = num_debate_rounds
        self.provider = provider
        self.model_name = model or MODEL_NAME
        self.api_key = api_key
        self.custom_base_url = custom_base_url
        self.output_root = output_root

        # 3. Dedicated Problem Output Path
        self.output_label = compute_output_label(self.identity_mode, self.delayed_reveal_target, self.virtual_combo)
        self.dataset_dir = os.path.join(self.output_root, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.problem_file_base = os.path.join(self.dataset_dir, f"{self.problem_id}_{self.output_label}")
        self.problem_json_path = f"{self.problem_file_base}.json"
        self.problem_jsonl_path = f"{self.problem_file_base}.jsonl"

        # 4. Fresh 12-Agent Roster (Zero memory from previous problem)
        self.agents: List[AgentMetadata] = build_default_agent_roster(
            backend_type=self.provider,
            provider=self.provider,
            api_key=self.api_key,
            endpoint_url=self.custom_base_url,
            model_overrides=self.model_name,
            identity_mode=self.identity_mode,
            virtual_combo=self.virtual_combo,
            delayed_reveal_target=self.delayed_reveal_target
        )
        self.backend = LLMBackend(
            provider=self.provider,
            api_key=self.api_key,
            custom_base_url=self.custom_base_url
        )

        # 4b. Resume from a prior partial attempt, if one exists and is safe to
        # trust. A job that failed mid-way (agent server down, OOM, etc.) still
        # has every stage up to the crash point durably persisted on disk (see
        # ConversationLogger.record_stage_detail) - only summary_analytics is
        # missing. Discarding that and re-running all 15 stages from scratch is
        # real, avoidable waste, so load it instead - but only if the roster
        # backing the old partial run (same model per agent_id) exactly matches
        # this run's roster. If a model was swapped between attempts (as
        # happened with Agent_05 this session), the partial data reflects a
        # DIFFERENT experimental condition and must not be silently blended in.
        resume_stages_detail, resume_transcript = self._load_resumable_progress()

        # 5. Real-Time Streaming Conversation Logger
        roster_specs = [
            {
                "agent_id": a.agent_id,
                "model_name": a.model_name,
                "family": a.family,
                "size_str": a.size_str,
                "size_billions": a.size_billions,
                "persona": asdict(a.persona) if a.persona else None
            }
            for a in self.agents
        ]
        self.conversation_logger = ConversationLogger(
            problem_item=self.question_item,
            identity_mode=self.identity_mode,
            virtual_combo=self.virtual_combo,
            delayed_reveal=self.delayed_reveal,
            delayed_reveal_stage=self.delayed_reveal_stage,
            delayed_reveal_target=self.delayed_reveal_target,
            total_debate_rounds=self.num_debate_rounds,
            agents_roster=roster_specs,
            output_dir=self.dataset_dir,
            output_label=self.output_label,
            resume_stages_detail=resume_stages_detail,
            resume_transcript=resume_transcript
        )

        # 6. Live per-agent state (vote/confidence/rationale), carried forward stage to
        # stage. If resuming, seed both from the last successfully-persisted stage
        # instead of starting cold - that stage's post-vote state IS the correct
        # starting point for whichever stage comes next.
        self.current_state: Dict[str, Dict[str, Any]] = {}
        self.stage_history: List[Dict[str, Any]] = list(resume_stages_detail) if resume_stages_detail else []
        if self.stage_history:
            last_stage = self.stage_history[-1]
            for v in last_stage.get("votes", []):
                self.current_state[v["agent_id"]] = {
                    "vote": v["vote"], "confidence": v["confidence"], "rationale": v.get("rationale")
                }
            logger.info(
                f"[{self.problem_id}/{self.output_label}] Resuming: "
                f"{len(self.stage_history)} stage(s) already completed "
                f"({', '.join(s['stage_name'] for s in self.stage_history)}), continuing from there."
            )

        # 7. Within a single stage, every agent's call is independent of every other
        # agent's call (each only depends on state that was already finalized before
        # the stage began) - so they are fired concurrently rather than one at a
        # time. This matters a lot once each agent is backed by its own vLLM server:
        # calling them sequentially would mean 11 of 12 GPUs sit idle while waiting
        # on whichever one is currently being called.
        #
        # Uses the module-level SHARED pool (_AGENT_CALL_POOL), not a fresh
        # per-instance pool. Confirmed live on a real 100-concurrency sweep:
        # giving every one of up to 100 concurrently-running SimulationRunner
        # instances its OWN 12-worker pool produced 1400+ live threads in one
        # process, which pathologically thrashed the GIL - the driver process
        # was measured burning ~900% CPU while making ZERO actual requests to
        # any of the 12 agent servers (confirmed via each server's own
        # Running/Waiting counts all sitting at 0), for well over a minute.
        # A shared, fixed-size pool bounds total thread count regardless of
        # how many jobs are running concurrently.
        self._agent_pool = _AGENT_CALL_POOL

    def _load_resumable_progress(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """
        Returns (stages_detail, group_chat_transcript) from a prior partial
        attempt at this exact job, or (None, None) if there is nothing to
        resume (no file, unreadable, already fully completed, or - critically -
        the old attempt used a different model per agent_id than this run).
        That last check matters: this codebase has swapped models mid-project
        (e.g. Agent_05's Llama-3.2-1B -> gemma-3-1b-it swap) specifically
        because the old model was broken. Blindly resuming from a file whose
        earlier stages ran on the broken model would silently mix two
        different experimental conditions into one trajectory - worse than
        just redoing the job.
        """
        if not os.path.exists(self.problem_json_path):
            return None, None
        try:
            with open(self.problem_json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[{self.problem_id}/{self.output_label}] Could not read existing file for resume, starting fresh: {e}")
            return None, None

        if doc.get("summary_analytics"):
            # Already fully completed - run_full_experiment.py's own
            # _job_already_completed check should have skipped this job
            # entirely before ever constructing a runner, but guard anyway.
            return None, None

        old_stages = doc.get("stages_detail") or []
        if not old_stages:
            return None, None

        old_roster = {
            a.get("agent_id"): a.get("model_name")
            for a in doc.get("simulation_metadata", {}).get("agent_roster", [])
        }
        current_roster = {a.agent_id: a.model_name for a in self.agents}
        if old_roster != current_roster:
            logger.warning(
                f"[{self.problem_id}/{self.output_label}] Found partial progress but the "
                f"agent roster has changed since (model swap) - discarding it and starting "
                f"fresh rather than mixing two different model conditions in one trajectory."
            )
            return None, None

        # A genuine crash can never leave a partial/malformed stage - a stage
        # is only ever appended to stages_detail once its full 12-vote list is
        # built (see record_stage_detail's call sites), so this should be
        # unreachable in practice. It's checked anyway as a hard backstop: a
        # resumed stage missing an agent's vote would otherwise crash deep
        # inside table-building code with a confusing KeyError - and worse,
        # PERMANENTLY, since every future sweep run would reload the same bad
        # file and hit the identical crash again. Same recovery as the roster
        # check above: discard and start fresh rather than get stuck forever.
        expected_ids = current_roster.keys()
        for s in old_stages:
            present_ids = {v.get("agent_id") for v in s.get("votes", [])}
            if present_ids != set(expected_ids):
                logger.warning(
                    f"[{self.problem_id}/{self.output_label}] Found partial progress but "
                    f"stage '{s.get('stage_name')}' has an incomplete/malformed vote list "
                    f"(expected {sorted(expected_ids)}, got {sorted(present_ids)}) - discarding "
                    f"it and starting fresh rather than risk a permanent crash loop."
                )
                return None, None

        # A crash can happen mid-stage: after some log_vote()/log_statement()
        # calls already fired (and synced to the transcript on disk) for the
        # in-progress stage, but before record_stage_detail() fires for it -
        # that stage is correctly ABSENT from old_stages (so it gets redone),
        # but its dangling partial events are still sitting in the OLD
        # group_chat_transcript. Confirmed live: loading that wholesale and
        # letting the redo append a full fresh set on top double-counts that
        # one stage's events (measured: 73 events -> 97 after resume, with the
        # affected stage's event count doubling). This only corrupts the
        # human-readable transcript/group_chat_transcript field, not
        # stages_detail/summary_analytics (the real analytical data), but
        # it's a real, avoidable defect - filter to only events belonging to
        # a stage that's actually confirmed complete before reusing them.
        completed_stage_names = {s.get("stage_name") for s in old_stages}
        old_transcript = [
            e for e in (doc.get("group_chat_transcript") or [])
            if e.get("stage") in completed_stage_names
        ]
        return old_stages, old_transcript

    # --------------------------------------------------------------------
    # Identity / roster helpers
    # --------------------------------------------------------------------

    def get_active_identity_mode(self, stage_index: int) -> str:
        """Determines active identity mode, accounting for delayed reveal."""
        if self.delayed_reveal or self.identity_mode == "delayed_reveal":
            if stage_index >= self.delayed_reveal_stage:
                return self.delayed_reveal_target
            return "anonymous"
        return self.identity_mode

    def build_roster_summary_table(self, active_mode: str) -> str:
        """Constructs a markdown table of visible agent identities."""
        lines = ["| Agent ID | Public Identity |", "| :--- | :--- |"]
        for agent in self.agents:
            disp = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            lines.append(f"| {agent.agent_id} | {disp} |")
        return "\n".join(lines)

    def _current_votes_only_table(self, active_mode: str) -> str:
        """Votes + confidence only, no rationale - used for the bandwagon stage."""
        lines = ["| Agent | Display Name | Vote | Conf |", "| :--- | :--- | :---: | :---: |"]
        for agent in self.agents:
            st = self.current_state[agent.agent_id]
            disp = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            lines.append(f"| {agent.agent_id} | {disp} | **{st['vote']}** | {st['confidence']:.2f} |")
        return "\n".join(lines)

    def _current_votes_and_rationale_table(self, active_mode: str) -> str:
        """Votes + confidence + rationale - used for reasoned-conformity and debate stages."""
        lines = ["| Agent | Display Name | Vote | Conf | Rationale |", "| :--- | :--- | :---: | :---: | :--- |"]
        for agent in self.agents:
            st = self.current_state[agent.agent_id]
            disp = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            raw_reason = st.get("rationale")
            if not isinstance(raw_reason, str):
                raw_reason = json.dumps(raw_reason) if raw_reason is not None else ""
            reason = raw_reason.replace("\n", " ").strip()
            if len(reason) > 280:
                reason = reason[:277] + "..."
            lines.append(f"| {agent.agent_id} | {disp} | **{st['vote']}** | {st['confidence']:.2f} | {reason} |")
        return "\n".join(lines)

    def _debate_history_block(self, up_to_round: int) -> str:
        """
        Full accumulated history (statements + revote outcomes) for every
        completed debate round from 1..up_to_round, built from self.stage_history
        (which already stores every past round's data) - so a later round can see
        the entire prior debate, not just the immediately preceding round. Used
        both before target selection (round_num - 1, i.e. rounds strictly before
        the current one) and before revoting (round_num, once the current round's
        own statements have been generated and appended to self.stage_history).
        """
        blocks = []
        for s_data in self.stage_history:
            if not s_data["stage_name"].startswith("debate_round_"):
                continue
            r_num = int(s_data["stage_name"].rsplit("_", 1)[1])
            if r_num > up_to_round:
                continue
            lines = [f"#### Round {r_num} statements:"]
            for stmt in s_data["statements"]:
                lines.append(_format_statement_line(stmt))
            if s_data.get("replies"):
                lines.append(f"#### Round {r_num} replies and declines:")
                for rep_ in s_data["replies"]:
                    lines.append(_format_reply_line(rep_))
            lines.append(f"#### Round {r_num} revote outcomes:")
            for v in s_data["votes"]:
                lines.append(f"- {v['display_identity']}: voted **{v['vote']}** (confidence {v['confidence']:.2f}) - {v['rationale']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "(No prior debate rounds yet - this is the first debate round.)"

    def _base_record(self, agent: AgentMetadata, display_id: str, previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "agent_id": agent.agent_id,
            "display_identity": display_id,
            "family": agent.family,
            "size_str": agent.size_str,
            "size_billions": agent.size_billions,
            "previous_vote": previous["vote"] if previous else None,
        }

    def _generate_for_tasks(
        self,
        tasks: List[Tuple[Any, "AgentMetadata", Any]],
        validate=None,
        max_retries: int = 2,
        max_tokens: int = MAX_TOKENS
    ) -> Dict[Any, str]:
        """
        Same as _generate_for_all_agents, but for an arbitrary list of
        (task_key, agent, build_prompt) tasks - used by the reply phase, where one
        agent gets one separate call per message it received. Identical retry rules
        (first attempt at TEMPERATURE, retries at RETRY_TEMPERATURE) and retries are
        recorded in the same retry_log.
        """
        if not hasattr(self, "_retry_log"):
            self._retry_log = []
            self.conversation_logger.metadata["retry_log"] = self._retry_log

        def _call(task) -> Tuple[Any, str]:
            key, agent, build_prompt = task
            sys_prompt, user_prompt, round_ctx = build_prompt(agent)
            last_err: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                attempt_temperature = TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE
                raw = self.backend.generate(agent, sys_prompt, user_prompt, temperature=attempt_temperature, max_tokens=max_tokens, round_context=round_ctx)
                try:
                    parsed = extract_json_from_text(raw)
                    if validate is not None:
                        validate(parsed, agent)
                    if attempt > 0:
                        self._retry_log.append({
                            "prompt_header": (user_prompt.strip().splitlines() or [""])[0][:80],
                            "agent_id": agent.agent_id, "accepted_attempt": attempt + 1,
                            "temperature_used": attempt_temperature, "last_error": str(last_err)[:200]})
                    return key, raw
                except (RuntimeError, ValueError) as e:
                    last_err = e
                    logger.warning(
                        f"[{agent.agent_id}] produced unparseable/invalid output on attempt "
                        f"{attempt + 1}/{max_retries + 1}, regenerating: {e}"
                    )
            # This is currently only used for the reply/decline phase, which runs
            # mid-debate (round 1 through num_debate_rounds) - unlike a vote/statement
            # failure, raising here would propagate up and destroy EVERY round's real
            # debate content for the whole question, not just this one agent's single
            # reply to one message (confirmed happening in production: Agent_01 - Qwen
            # 2.5 3B - exhausting retries here cost entire already-debated questions,
            # e.g. moralchoice_0687/0690/0698/0721, moralmachine_0053/0068/0071). Fall
            # back to an auto-decline instead of raising, so this one message's
            # reply/decline is the only thing lost, not the whole question.
            logger.warning(
                f"[{agent.agent_id}] exhausted all {max_retries + 1} attempts ({last_err}); "
                f"falling back to an auto-decline for this one reply so the rest of the "
                f"question's real debate data is not lost."
            )
            return key, '{"decision": "decline", "text": "(auto-decline: agent did not produce a valid reply after retries)"}'

        results: Dict[Any, str] = {}
        futures = [self._agent_pool.submit(_call, t) for t in tasks]
        for future in futures:
            key, raw = future.result()
            results[key] = raw
        return results

    def _generate_for_all_agents(
        self,
        build_prompt: Callable[[AgentMetadata], Tuple[str, str, Dict[str, Any]]],
        max_retries: int = 2,
        validate: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_tokens: int = MAX_TOKENS,
        fallback_raw: Optional[str] = None,
        rationale_only_fallback: bool = False
    ) -> Dict[str, str]:
        """
        Fires backend.generate() for every agent in this stage CONCURRENTLY - each
        agent is served by its own model (its own GPU/vLLM process in the real
        deployment), and none of them need to see another agent's output within the
        same stage, so there is no reason to call them one at a time. Returns
        {agent_id: raw_response_text}, safe to then process/log in a fixed,
        deterministic order (self.agents) afterward.

        Every caller immediately does extract_json_from_text(raw_resp), so a
        response this returns must actually be parseable - retries here with a
        FRESH sample (RETRY_TEMPERATURE makes each retry attempt genuinely
        different, not a repeat of the same failure - the official first attempt
        still uses the module-level TEMPERATURE, which is 0/greedy) on an
        occasional garbled/off-format output from a smaller model. This is not a
        data-integrity risk the way a fake default would be: it is a real
        regeneration, logged, not fabricated data. Only raises (failing the whole
        job, to be retried by the next sweep run) after every attempt fails to
        parse.

        `validate`, if given, runs on the successfully-parsed dict and should
        raise ValueError for a semantically invalid result (e.g. a vote that
        isn't "A"/"B") - treated exactly like a parse failure: retried with a
        fresh sample, not silently defaulted. Syntactically valid JSON with
        nonsense content is just as much of a data-integrity risk as unparseable
        text; only checking parseability was letting it through.
        """
        def _call(agent: AgentMetadata) -> Tuple[str, str]:
            sys_prompt, user_prompt, round_ctx = build_prompt(agent)
            last_err: Optional[Exception] = None
            last_parsed: Optional[Dict[str, Any]] = None
            for attempt in range(max_retries + 1):
                attempt_temperature = TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE
                raw = self.backend.generate(agent, sys_prompt, user_prompt, temperature=attempt_temperature, max_tokens=max_tokens, round_context=round_ctx)
                try:
                    parsed = extract_json_from_text(raw)
                    if validate is not None:
                        validate(parsed, agent)
                    if attempt > 0:
                        self._retry_log.append({
                            "prompt_header": (user_prompt.strip().splitlines() or [""])[0][:80],
                            "agent_id": agent.agent_id, "accepted_attempt": attempt + 1,
                            "temperature_used": attempt_temperature, "last_error": str(last_err)[:200]})
                    return agent.agent_id, raw
                except (RuntimeError, ValueError) as e:
                    last_err = e
                    try:
                        last_parsed = extract_json_from_text(raw)
                    except (RuntimeError, ValueError):
                        last_parsed = None
                    logger.warning(
                        f"[{agent.agent_id}] produced unparseable/invalid output on attempt "
                        f"{attempt + 1}/{max_retries + 1}, regenerating: {e}"
                    )
            if rationale_only_fallback and last_parsed is not None:
                # Confirmed in production: Agent_07 (Gemma-3-12B) occasionally
                # emits a genuinely valid vote/confidence/flipped but an empty
                # rationale, in combined vote+rationale revote stages (later
                # debate rounds) - a DIFFERENT call site from the isolated
                # rationale_collection stage's fallback_raw above, one that
                # carries real vote data. Never fabricate a vote - only patch
                # the rationale, and only when the vote field from this exact
                # last attempt independently re-validates as genuinely "A"/"B"
                # on its own. If the vote itself was also the problem, this is
                # a no-op and the original raise below still fires unchanged.
                try:
                    _validate_vote_field(last_parsed)
                    vote_is_genuinely_valid = True
                except (RuntimeError, ValueError):
                    vote_is_genuinely_valid = False
                if vote_is_genuinely_valid:
                    patched = dict(last_parsed)
                    patched["rationale"] = (
                        "(no rationale provided: agent failed to produce a valid "
                        "rationale after repeated attempts - vote/confidence above "
                        "are the agent's own genuine last values, not fabricated)"
                    )
                    patched.setdefault("flipped", False)
                    logger.warning(
                        f"[{agent.agent_id}] failed to produce a valid rationale after "
                        f"{max_retries + 1} attempts ({last_err}), but its own vote/"
                        f"confidence from that same attempt were genuinely valid - "
                        f"keeping the real vote and substituting only a placeholder "
                        f"rationale, rather than discarding real vote data."
                    )
                    return agent.agent_id, json.dumps(patched)
            if fallback_raw is not None:
                # Only passed by call sites whose field is non-essential free
                # text with no vote/tally at stake (confirmed: currently only
                # the isolated rationale_collection stage, which asks for
                # nothing but a "rationale" string - see the reply/decline
                # fallback a few lines above this function for the identical
                # precedent: losing one agent's one piece of prose for one
                # question is preferable to losing that question's entire
                # debate data over a single flaky rationale generation).
                logger.warning(
                    f"[{agent.agent_id}] failed to produce a valid response after "
                    f"{max_retries + 1} attempts ({last_err}); falling back to a "
                    f"placeholder so the rest of this question's real data is not lost."
                )
                return agent.agent_id, fallback_raw
            raise RuntimeError(
                f"[{agent.agent_id}] failed to produce a valid response after "
                f"{max_retries + 1} attempts: {last_err}"
            )

        if not hasattr(self, "_retry_log"):
            self._retry_log = []
            self.conversation_logger.metadata["retry_log"] = self._retry_log   # saved with every JSON write
        results: Dict[str, str] = {}
        futures = [self._agent_pool.submit(_call, agent) for agent in self.agents]
        for future in futures:
            agent_id, raw = future.result()
            results[agent_id] = raw
        return results

    # --------------------------------------------------------------------
    # Stage 0: baseline
    # --------------------------------------------------------------------

    def execute_baseline_stage(self) -> Dict[str, Any]:
        """Independent vote, before any participant sees any other participant's answer."""
        stage_name = "baseline"
        stage_index = 0
        logger.info("=" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")
        logger.info("=" * 70)

        active_mode = self.get_active_identity_mode(stage_index)
        roster_table = self.build_roster_summary_table(active_mode)

        announcement = (
            f"New Problem Statement Loaded: {self.question_item['question']}\n"
            f"Option A: {self.question_item['option_a']}\n"
            f"Option B: {self.question_item['option_b']}"
        )
        self.conversation_logger.log_system_announcement(announcement, stage_name=stage_name)

        context_block = f"**CONTEXT:** {self.question_item.get('context', '')}" if self.question_item.get('context') else ""

        def build_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_table)
            user_prompt = BASELINE_VOTE_PROMPT_TEMPLATE.format(
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                context_block=context_block
            )
            round_ctx = {
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"],
                "option_b": self.question_item["option_b"]
            }
            return sys_prompt, user_prompt, round_ctx

        raw_responses = self._generate_for_all_agents(build_prompt, validate=_validate_vote_field)

        votes_record = []
        tally = {"A": 0, "B": 0}
        total_conf = 0.0

        for agent in self.agents:
            disp_id = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            raw_resp = raw_responses[agent.agent_id]
            parsed = extract_json_from_text(raw_resp)

            vote = str(parsed.get("vote", "A")).upper()
            if vote not in ["A", "B"]:
                vote = "A"
            conf = float(parsed.get("confidence", 0.75))
            if conf > 1.0:
                conf = conf / 100.0
            conf = round(max(0.0, min(1.0, conf)), 2)

            self.current_state[agent.agent_id] = {"vote": vote, "confidence": conf, "rationale": None}
            tally[vote] += 1
            total_conf += conf

            record = self._base_record(agent, disp_id, None)
            record.update({"vote": vote, "confidence": conf, "confidence_delta": None, "flipped": None, "rationale": None})
            votes_record.append(record)

            self.conversation_logger.log_vote(stage_name=stage_name, agent_id=agent.agent_id, display_id=disp_id, vote=vote, confidence=conf)
            logger.info(f"[{disp_id}] Baseline vote: Option {vote} (Conf: {conf:.2f})")

        mean_conf = round(total_conf / len(self.agents), 2)
        stage_data = {
            "stage_name": stage_name,
            "stage_index": stage_index,
            "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode,
            "tally": tally,
            "mean_confidence": mean_conf,
            "votes": votes_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        logger.info(f"BASELINE COMPLETE: A={tally['A']} B={tally['B']} | Mean Conf: {mean_conf:.2f}")
        return stage_data

    # --------------------------------------------------------------------
    # Stage 1: bandwagon revote (votes only, no rationale seen or asked)
    # --------------------------------------------------------------------

    def execute_bandwagon_stage(self) -> Dict[str, Any]:
        stage_name = "bandwagon_revote"
        stage_index = 1
        logger.info("-" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")

        active_mode = self.get_active_identity_mode(stage_index)
        # No separate roster table here - the votes table below already shows
        # every agent's ID and display identity, so a duplicate table right
        # before it is pure redundancy. Confirmed on a real run: a 1B model fed
        # two consecutive markdown tables started echoing/continuing the first
        # one instead of doing the actual task.
        roster_note = "(see the current-state table below for every participant's ID and identity.)"
        votes_only_table = self._current_votes_only_table(active_mode)

        def build_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_note)
            user_prompt = VOTES_ONLY_REVOTE_PROMPT_TEMPLATE.format(
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                votes_only_table=votes_only_table,
                previous_vote=prev_vote,
                previous_confidence=f"{prev_conf:.2f}"
            )
            round_ctx = {
                "previous_vote": prev_vote, "previous_confidence": prev_conf,
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"]
            }
            return sys_prompt, user_prompt, round_ctx

        raw_responses = self._generate_for_all_agents(build_prompt, validate=_validate_vote_field)

        votes_record = []
        tally = {"A": 0, "B": 0}
        flips_count = 0

        for agent in self.agents:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            raw_resp = raw_responses[agent.agent_id]
            parsed = extract_json_from_text(raw_resp)

            new_vote = str(parsed.get("vote", prev_vote)).upper()
            if new_vote not in ["A", "B"]:
                new_vote = prev_vote
            new_conf = float(parsed.get("confidence", prev_conf))
            if new_conf > 1.0:
                new_conf = new_conf / 100.0
            new_conf = round(max(0.0, min(1.0, new_conf)), 2)
            flipped = new_vote != prev_vote
            if flipped:
                flips_count += 1

            self.current_state[agent.agent_id] = {"vote": new_vote, "confidence": new_conf, "rationale": None}

            disp_id = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            conf_delta = round(new_conf - prev_conf, 2)
            tally[new_vote] += 1

            record = self._base_record(agent, disp_id, prev)
            record.update({"vote": new_vote, "confidence": new_conf, "confidence_delta": conf_delta, "flipped": flipped, "rationale": None})
            votes_record.append(record)

            self.conversation_logger.log_vote(
                stage_name=stage_name, agent_id=agent.agent_id, display_id=disp_id, vote=new_vote,
                confidence=new_conf, previous_vote=prev_vote, confidence_delta=conf_delta, flipped=flipped
            )
            logger.info(f"[{disp_id}] {stage_name}: {prev_vote}->{new_vote} (flipped={flipped})")

        stage_data = {
            "stage_name": stage_name, "stage_index": stage_index, "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode, "tally": tally, "flips_in_stage": flips_count, "votes": votes_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        logger.info(f"{stage_name.upper()} COMPLETE: A={tally['A']} B={tally['B']} | Flips: {flips_count}")
        return stage_data

    # --------------------------------------------------------------------
    # Stage 2: rationale collection (no revote, just explain current vote)
    # --------------------------------------------------------------------

    def execute_rationale_stage(self) -> Dict[str, Any]:
        stage_name = "rationale_collection"
        stage_index = 2
        logger.info("-" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")

        active_mode = self.get_active_identity_mode(stage_index)
        roster_table = self.build_roster_summary_table(active_mode)

        def build_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            cur = self.current_state[agent.agent_id]
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_table)
            user_prompt = RATIONALE_ONLY_PROMPT_TEMPLATE.format(
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                current_vote=cur["vote"],
                current_confidence=f"{cur['confidence']:.2f}"
            )
            round_ctx = {
                "current_vote": cur["vote"],
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"]
            }
            return sys_prompt, user_prompt, round_ctx

        raw_responses = self._generate_for_all_agents(
            build_prompt, validate=_validate_rationale_field,
            fallback_raw='{"rationale": "(no rationale provided: agent failed to produce a valid response after repeated attempts)"}'
        )

        votes_record = []
        tally = {"A": 0, "B": 0}

        for agent in self.agents:
            cur = self.current_state[agent.agent_id]
            raw_resp = raw_responses[agent.agent_id]
            parsed = extract_json_from_text(raw_resp)
            # Confirmed on a real run: a model occasionally nests "rationale" as
            # an object (e.g. {"summary": ..., "detail": ...}) instead of a plain
            # string, which later crashed the votes/rationale table's .replace()
            # call. Coerce defensively rather than trust the model's shape.
            rationale = parsed.get("rationale", raw_resp)
            if not isinstance(rationale, str):
                rationale = json.dumps(rationale) if rationale is not None else raw_resp

            self.current_state[agent.agent_id]["rationale"] = rationale
            tally[cur["vote"]] += 1

            disp_id = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            record = self._base_record(agent, disp_id, cur)
            record.update({"vote": cur["vote"], "confidence": cur["confidence"], "confidence_delta": 0.0, "flipped": False, "rationale": rationale})
            votes_record.append(record)

            self.conversation_logger.log_rationale(stage_name=stage_name, agent_id=agent.agent_id, display_id=disp_id, vote=cur["vote"], rationale=rationale)
            logger.info(f"[{disp_id}] rationale collected for Option {cur['vote']}")

        stage_data = {
            "stage_name": stage_name, "stage_index": stage_index, "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode, "tally": tally, "votes": votes_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        return stage_data

    # --------------------------------------------------------------------
    # Stage 3: reasoned-conformity revote (votes + rationale visible)
    # --------------------------------------------------------------------

    def execute_reasoned_conformity_stage(self) -> Dict[str, Any]:
        stage_name = "reasoned_conformity_revote"
        stage_index = 3
        logger.info("-" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")

        active_mode = self.get_active_identity_mode(stage_index)
        # See execute_bandwagon_stage: no separate roster table - the votes+
        # rationale table below already carries every agent's identity.
        roster_note = "(see the current-state table below for every participant's ID and identity.)"
        votes_rationale_table = self._current_votes_and_rationale_table(active_mode)

        def build_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_note)
            user_prompt = VOTES_AND_RATIONALE_REVOTE_PROMPT_TEMPLATE.format(
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                votes_and_rationale_table=votes_rationale_table,
                previous_vote=prev_vote,
                previous_confidence=f"{prev_conf:.2f}"
            )
            round_ctx = {
                "previous_vote": prev_vote, "previous_confidence": prev_conf,
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"]
            }
            return sys_prompt, user_prompt, round_ctx

        raw_responses = self._generate_for_all_agents(
            build_prompt, validate=_validate_vote_and_rationale_field,
            rationale_only_fallback=True
        )

        votes_record = []
        tally = {"A": 0, "B": 0}
        flips_count = 0

        for agent in self.agents:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            raw_resp = raw_responses[agent.agent_id]
            parsed = extract_json_from_text(raw_resp)

            new_vote, new_conf, flipped, rationale = self._parse_revote(parsed, prev_vote, prev_conf, raw_resp)
            if flipped:
                flips_count += 1
            self.current_state[agent.agent_id] = {"vote": new_vote, "confidence": new_conf, "rationale": rationale}

            disp_id = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            conf_delta = round(new_conf - prev_conf, 2)
            tally[new_vote] += 1

            record = self._base_record(agent, disp_id, prev)
            record.update({"vote": new_vote, "confidence": new_conf, "confidence_delta": conf_delta, "flipped": flipped, "rationale": rationale})
            votes_record.append(record)

            self.conversation_logger.log_vote(
                stage_name=stage_name, agent_id=agent.agent_id, display_id=disp_id, vote=new_vote,
                confidence=new_conf, previous_vote=prev_vote, confidence_delta=conf_delta, flipped=flipped, rationale=rationale
            )
            logger.info(f"[{disp_id}] {stage_name}: {prev_vote}->{new_vote} (flipped={flipped})")

        stage_data = {
            "stage_name": stage_name, "stage_index": stage_index, "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode, "tally": tally, "flips_in_stage": flips_count, "votes": votes_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        logger.info(f"{stage_name.upper()} COMPLETE: A={tally['A']} B={tally['B']} | Flips: {flips_count}")
        return stage_data

    def _parse_revote(self, parsed: Dict[str, Any], prev_vote: str, prev_conf: float, raw_resp: str) -> Tuple[str, float, bool, str]:
        new_vote = str(parsed.get("vote", prev_vote)).upper()
        if new_vote not in ["A", "B"]:
            new_vote = prev_vote
        new_conf = float(parsed.get("confidence", prev_conf))
        if new_conf > 1.0:
            new_conf = new_conf / 100.0
        new_conf = round(max(0.0, min(1.0, new_conf)), 2)
        flipped = new_vote != prev_vote
        # See execute_rationale_stage: a model occasionally nests "rationale" as
        # an object instead of a plain string - coerce defensively.
        rationale = parsed.get("rationale", raw_resp)
        if not isinstance(rationale, str):
            rationale = json.dumps(rationale) if rationale is not None else raw_resp
        return new_vote, new_conf, flipped, rationale

    # --------------------------------------------------------------------
    # Stages 4..(3+N): debate rounds - all 12 speak, then all 12 revote
    # --------------------------------------------------------------------

    def execute_debate_round(self, round_num: int) -> Dict[str, Any]:
        stage_name = f"debate_round_{round_num}"
        stage_index = 3 + round_num
        logger.info("-" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")

        active_mode = self.get_active_identity_mode(stage_index)
        roster_table = self.build_roster_summary_table(active_mode)
        # See execute_bandwagon_stage: the statement phase's votes+rationale
        # table already carries every agent's identity, so its system prompt
        # gets a short note instead of a second full table (real regression:
        # a 1B model given two consecutive markdown tables started echoing the
        # first one instead of doing the task). The revote phase further below
        # keeps the full roster_table - it does NOT show a second table, only a
        # statement list, so there is no redundancy there.
        statement_roster_note = "(see the current-state table below for every participant's ID and identity.)"
        pre_debate_table = self._current_votes_and_rationale_table(active_mode)
        prior_debate_history = self._debate_history_block(round_num - 1)
        # Uniform trim (see _uniform_trim_shared_history) - applied once here so
        # both the statement phase (step 1) and reply phase (step 2) below, which
        # both reference this same variable, see the identical (possibly trimmed)
        # history.
        prior_debate_history = self._uniform_trim_shared_history(prior_debate_history)

        if (self.delayed_reveal or self.identity_mode == "delayed_reveal") and stage_index == self.delayed_reveal_stage:
            reveal_msg = f"Public identities for all participants in the assembly are now revealed ({active_mode})."
            self.conversation_logger.log_system_announcement(reveal_msg, stage_name=stage_name)
            logger.info(f"ANNOUNCEMENT: {reveal_msg}")

        # -------------------------------------------------------------
        # Phase 1: every agent independently picks a target and writes a
        # statement. No agent sees any other agent's statement this round
        # until phase 2 (they are generated in isolation, then revealed).
        # -------------------------------------------------------------
        def build_statement_prompt(speaker: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            target_candidates = [a for a in self.agents if a.agent_id != speaker.agent_id]
            self_id_block = get_agent_self_identity_prompt_block(speaker, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=statement_roster_note)
            stmt_prompt = DEBATE_STATEMENT_PROMPT_TEMPLATE.format(
                round_num=round_num,
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                votes_and_rationale_table=pre_debate_table,
                prior_debate_history=prior_debate_history
            )
            round_ctx = {
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"],
                "candidate_target_ids": [a.agent_id for a in target_candidates]
            }
            return sys_prompt, stmt_prompt, round_ctx

        _valid_ids = [a.agent_id for a in self.agents]
        statement_responses = self._generate_for_all_agents(
            build_statement_prompt,
            validate=lambda parsed, agent: _validate_debate_statement_field(parsed, agent, valid_ids=_valid_ids))

        statements = []
        for idx, speaker in enumerate(self.agents):
            target_candidates = [a for a in self.agents if a.agent_id != speaker.agent_id]
            raw_resp = statement_responses[speaker.agent_id]
            parsed = extract_json_from_text(raw_resp)

            target_id = str(parsed.get("target_agent_id", "")).strip()
            target_agent = next((a for a in target_candidates if a.agent_id == target_id), None)
            if target_agent is None:
                target_agent = random.choice(target_candidates)

            statement_text = parsed.get("statement", raw_resp)
            if not isinstance(statement_text, str):
                statement_text = json.dumps(statement_text) if statement_text is not None else raw_resp
            statement_text = statement_text.strip()
            speaker_disp = format_agent_display_identity(speaker, active_mode, self.virtual_combo)
            target_disp = format_agent_display_identity(target_agent, active_mode, self.virtual_combo)

            stmt_record = {
                "statement_idx": idx + 1,
                "speaker_agent_id": speaker.agent_id, "speaker_display_identity": speaker_disp,
                "speaker_vote_at_send": self.current_state[speaker.agent_id]["vote"],
                "target_agent_id": target_agent.agent_id, "target_display_identity": target_disp,
                "statement": statement_text
            }
            statements.append(stmt_record)

            self.conversation_logger.log_statement(
                stage_name=stage_name, statement_idx=idx + 1,
                speaker_id=speaker.agent_id, speaker_display=speaker_disp,
                target_id=target_agent.agent_id, target_display=target_disp, statement=statement_text
            )
            logger.info(f"[{stage_name}] {speaker_disp} -> {target_disp}: {statement_text[:100]}...")

        # -------------------------------------------------------------
        # Phase 2: reveal all statements together, then everyone revotes.
        # The revote context is the FULL debate history (every prior round's
        # statements + revote outcomes, plus this round's just-generated
        # statements) - not just this round in isolation. self.stage_history
        # does not yet contain this round's stage_data (that append happens at
        # the end of this method), so this round's own statements are appended
        # manually onto the prior-rounds history built from stage_history.
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # Phase 1b: reply decisions. For EACH message, the agent it was
        # addressed to decides - in its own separate call - to reply or to
        # decline (with a short reason). One reply per message, no
        # reply-to-reply. All replies/declines are then shown to everyone.
        # -------------------------------------------------------------
        round_statements_block = "\n".join(_format_statement_line(s) for s in statements)
        agent_by_id = {a.agent_id: a for a in self.agents}
        inbox_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for s in statements:
            inbox_by_agent.setdefault(s["target_agent_id"], []).append(s)

        def make_reply_builder(msg: Dict[str, Any]):
            def build_reply_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
                self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
                sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_table)
                reply_prompt = DEBATE_REPLY_PROMPT_TEMPLATE.format(
                    round_num=round_num,
                    question_text=self.question_item["question"],
                    option_a=self.question_item["option_a"],
                    option_b=self.question_item["option_b"],
                    current_vote=self.current_state[agent.agent_id]["vote"],
                    prior_debate_history=prior_debate_history,
                    round_statements=round_statements_block,
                    inbox="\n".join(_format_statement_line(m) for m in inbox_by_agent[agent.agent_id]),
                    focus_message=_format_statement_line(msg)
                )
                round_ctx = {
                    "question": self.question_item["question"],
                    "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"],
                    "sender_id": msg["speaker_agent_id"]
                }
                return sys_prompt, reply_prompt, round_ctx
            return build_reply_prompt

        reply_tasks = [(m["statement_idx"], agent_by_id[m["target_agent_id"]], make_reply_builder(m)) for m in statements]
        reply_responses = self._generate_for_tasks(reply_tasks, validate=_validate_reply_field)

        replies = []
        for m in statements:
            parsed = extract_json_from_text(reply_responses[m["statement_idx"]])
            decision = parsed["decision"].strip().lower()
            text = parsed["text"].strip()
            rep_record = {
                "message_idx": m["statement_idx"],
                "replier_agent_id": m["target_agent_id"], "replier_display_identity": m["target_display_identity"],
                "sender_agent_id": m["speaker_agent_id"], "sender_display_identity": m["speaker_display_identity"],
                "decision": decision, "text": text
            }
            replies.append(rep_record)
            self.conversation_logger.log_reply(
                stage_name=stage_name, message_idx=m["statement_idx"],
                replier_id=m["target_agent_id"], replier_display=m["target_display_identity"],
                sender_id=m["speaker_agent_id"], sender_display=m["speaker_display_identity"],
                decision=decision, text=text
            )
            logger.info(f"[{stage_name}] {m['target_display_identity']} {decision.upper()} -> {m['speaker_display_identity']}: {text[:80]}")

        current_round_block = "\n".join(
            [f"#### Round {round_num} statements (this round):"] + [_format_statement_line(s) for s in statements]
            + [f"#### Round {round_num} replies and declines (this round):"] + [_format_reply_line(r_) for r_ in replies]
        )
        full_debate_history = (
            current_round_block if prior_debate_history.startswith("(No prior debate rounds")
            else f"{prior_debate_history}\n\n{current_round_block}"
        )
        # Uniform trim again here: this is a DIFFERENT (longer) string than
        # prior_debate_history above (it adds this round's own just-generated
        # statements/replies), used only by the revote phase (step 3) below -
        # same reasoning as above, applied to whichever string is actually
        # being sent to every agent.
        full_debate_history = self._uniform_trim_shared_history(full_debate_history)

        current_votes_table = self._current_votes_and_rationale_table(active_mode)   # state BEFORE this round's revote

        def build_revote_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(identity_prompt_block=self_id_block, roster_summary=roster_table)
            revote_prompt = DEBATE_REVOTE_PROMPT_TEMPLATE.format(
                round_num=round_num,
                question_text=self.question_item["question"],
                option_a=self.question_item["option_a"],
                option_b=self.question_item["option_b"],
                previous_vote=prev_vote,
                previous_confidence=f"{prev_conf:.2f}",
                current_votes_table=current_votes_table,
                full_debate_history=full_debate_history
            )
            round_ctx = {
                "previous_vote": prev_vote, "previous_confidence": prev_conf,
                "question": self.question_item["question"],
                "option_a": self.question_item["option_a"], "option_b": self.question_item["option_b"]
            }
            return sys_prompt, revote_prompt, round_ctx

        revote_responses = self._generate_for_all_agents(
            build_revote_prompt, validate=_validate_vote_and_rationale_field,
            rationale_only_fallback=True
        )

        votes_record = []
        tally = {"A": 0, "B": 0}
        flips_count = 0

        for agent in self.agents:
            prev = self.current_state[agent.agent_id]
            prev_vote, prev_conf = prev["vote"], prev["confidence"]
            raw_resp = revote_responses[agent.agent_id]
            parsed = extract_json_from_text(raw_resp)

            new_vote, new_conf, flipped, rationale = self._parse_revote(parsed, prev_vote, prev_conf, raw_resp)
            if flipped:
                flips_count += 1
            self.current_state[agent.agent_id] = {"vote": new_vote, "confidence": new_conf, "rationale": rationale}

            disp_id = format_agent_display_identity(agent, active_mode, self.virtual_combo)
            conf_delta = round(new_conf - prev_conf, 2)
            tally[new_vote] += 1

            record = self._base_record(agent, disp_id, prev)
            record.update({"vote": new_vote, "confidence": new_conf, "confidence_delta": conf_delta, "flipped": flipped, "rationale": rationale})
            votes_record.append(record)

            self.conversation_logger.log_vote(
                stage_name=stage_name, agent_id=agent.agent_id, display_id=disp_id, vote=new_vote,
                confidence=new_conf, previous_vote=prev_vote, confidence_delta=conf_delta, flipped=flipped, rationale=rationale
            )
            if flipped:
                logger.warning(f"-> CHANGED: {disp_id} {prev_vote} -> {new_vote} (Conf: {new_conf:.2f}, Δ: {conf_delta:+.2f})")
            else:
                logger.info(f"[{disp_id}] kept {new_vote} (Conf: {new_conf:.2f}, Δ: {conf_delta:+.2f})")

        stage_data = {
            "stage_name": stage_name, "stage_index": stage_index, "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode, "statements": statements, "replies": replies,
            "tally": tally, "flips_in_stage": flips_count, "votes": votes_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        logger.info(f"{stage_name.upper()} COMPLETE: A={tally['A']} B={tally['B']} | Flips: {flips_count}")
        return stage_data

    # --------------------------------------------------------------------
    # Final stage (anonymous condition only): guess the model behind each
    # other participant.
    # --------------------------------------------------------------------

    def execute_model_identification_stage(self) -> Dict[str, Any]:
        """
        Only runs when self.identity_mode == "anonymous" (checked by the caller
        in run()). Agents were told at the start of the run (see identities.py's
        anonymous branch of get_agent_self_identity_prompt_block) that this was
        coming, so they should have been tracking reasoning style throughout.
        """
        stage_name = "model_identification"
        stage_index = 4 + self.num_debate_rounds
        logger.info("=" * 70)
        logger.info(f"STAGE [{stage_name}]: Problem [{self.problem_id}]")
        logger.info("=" * 70)

        active_mode = "anonymous"
        full_history = self._debate_history_block(self.num_debate_rounds)
        all_agent_ids = [a.agent_id for a in self.agents]

        def build_prompt(agent: AgentMetadata) -> Tuple[str, str, Dict[str, Any]]:
            other_agents = [a for a in self.agents if a.agent_id != agent.agent_id]
            other_ids_list = "\n".join(f"- {a.agent_id}" for a in other_agents)
            self_id_block = get_agent_self_identity_prompt_block(agent, active_mode, self.virtual_combo)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                identity_prompt_block=self_id_block,
                roster_summary="(anonymous assembly - no public identities were ever shown; see the participant list below.)"
            )
            guess_prompt = GUESS_MODELS_PROMPT_TEMPLATE.format(
                full_debate_history=full_history,
                other_agent_ids_list=other_ids_list
            )
            round_ctx = {"other_agent_ids": [a.agent_id for a in other_agents]}
            return sys_prompt, guess_prompt, round_ctx

        try:
            responses = self._generate_for_all_agents(
                build_prompt,
                validate=lambda p, a: _validate_model_guesses_field(p, a, all_agent_ids),
                max_tokens=MODEL_GUESS_MAX_TOKENS
            )
        except RuntimeError as e:
            # This is the LAST stage, running only after all real experimental data
            # (votes, debate statements, replies/declines) is already finalized in
            # stage_history. The strict validator above is correct to want a complete
            # guess for every other agent - but a weak model (confirmed: Agent_04,
            # Gemma 3 1B) exhausting all retries here must not destroy that already-
            # complete real debate data over this tangential guess-the-model exercise.
            # Fall back to one best-effort unvalidated attempt per agent; whatever
            # comes back (even partial/malformed) is tolerated by the reconstruction
            # loop below, which already defaults missing/invalid entries to empty
            # model_guess/rationale strings - this just extends that same, already-
            # accepted tolerance to "never produced a fully valid response" too.
            logger.warning(f"[{stage_name}] generate-with-validation exhausted retries ({e}); "
                            f"falling back to one unvalidated attempt per agent so the real "
                            f"debate data for this question is not discarded.")
            responses = {}
            for agent in self.agents:
                sys_prompt, guess_prompt, round_ctx = build_prompt(agent)
                try:
                    responses[agent.agent_id] = self.backend.generate(
                        agent, sys_prompt, guess_prompt, temperature=RETRY_TEMPERATURE,
                        max_tokens=MODEL_GUESS_MAX_TOKENS, round_context=round_ctx
                    )
                except Exception as inner_e:
                    logger.warning(f"[{stage_name}] {agent.agent_id} fallback generation also failed: {inner_e}")
                    responses[agent.agent_id] = "{}"

        guesses_record = []
        for agent in self.agents:
            raw_resp = responses[agent.agent_id]
            try:
                parsed = extract_json_from_text(raw_resp)
            except Exception as e:
                logger.warning(f"[{stage_name}] {agent.agent_id} response unparseable even in fallback "
                                f"({e}); recording empty guesses for this agent.")
                parsed = {}
            guesses = parsed.get("guesses", {})
            if not isinstance(guesses, dict):
                guesses = {}
            clean_guesses = {}
            for other in self.agents:
                if other.agent_id == agent.agent_id:
                    continue
                entry = guesses.get(other.agent_id, {})
                if not isinstance(entry, dict):
                    entry = {}
                clean_guesses[other.agent_id] = {
                    "model_guess": str(entry.get("model_guess", "")).strip(),
                    "rationale": str(entry.get("rationale", "")).strip()
                }
            record = {
                "agent_id": agent.agent_id,
                "family": agent.family, "size_str": agent.size_str, "size_billions": agent.size_billions,
                "guesses": clean_guesses
            }
            guesses_record.append(record)
            self.conversation_logger.log_system_announcement(
                f"{agent.agent_id} guessed: " + "; ".join(
                    f"{aid}={g['model_guess']}" for aid, g in clean_guesses.items()
                ),
                stage_name=stage_name
            )
            logger.info(f"[{stage_name}] {agent.agent_id} guessed {len(clean_guesses)} other participants.")

        stage_data = {
            "stage_name": stage_name, "stage_index": stage_index, "timestamp": datetime.now().isoformat(),
            "identity_mode_active": active_mode, "guesses": guesses_record
        }
        self.stage_history.append(stage_data)
        self.conversation_logger.record_stage_detail(stage_data)
        logger.info(f"{stage_name.upper()} COMPLETE")
        return stage_data

    # --------------------------------------------------------------------
    # Summary analytics
    # --------------------------------------------------------------------

    def compute_summary_analytics(self) -> Dict[str, Any]:
        """
        Computes factual summary metrics and agent trajectories across every
        voting stage. Explicitly excludes "model_identification" (the anonymous-
        condition final guess-the-model stage, if present) from every vote/tally/
        trajectory computation below - it carries no "votes"/"tally" data at all
        (it's guesses about OTHER agents' model identity, not a vote on the
        dilemma), so mixing it into r_final or the per-stage vote-history loop
        would either KeyError or silently corrupt "final_vote"/"final_tally".
        """
        voting_stages = [s for s in self.stage_history if s["stage_name"] != "model_identification"]
        model_id_stage = next((s for s in self.stage_history if s["stage_name"] == "model_identification"), None)

        stage_names = [s["stage_name"] for s in voting_stages]
        r0 = voting_stages[0]
        r_final = voting_stages[-1]

        trajectories = []
        total_flips = 0
        flips_log = []

        for agent in self.agents:
            aid = agent.agent_id
            vote_hist, conf_hist = [], []
            agent_flips = 0

            for s_data in voting_stages:
                v_record = next(v for v in s_data["votes"] if v["agent_id"] == aid)
                vote_hist.append(v_record["vote"])
                conf_hist.append(v_record["confidence"])
                if v_record.get("flipped"):
                    agent_flips += 1
                    total_flips += 1
                    flips_log.append({
                        "stage": s_data["stage_name"], "agent_id": aid,
                        "from_vote": v_record.get("previous_vote"), "to_vote": v_record.get("vote"),
                        "rationale": v_record.get("rationale")
                    })

            last_vote_record = next(v for v in r_final["votes"] if v["agent_id"] == aid)
            trajectories.append({
                "agent_id": aid, "family": agent.family, "size_str": agent.size_str, "size_b": agent.size_billions,
                "persona": asdict(agent.persona) if agent.persona else None,
                "stage_names": stage_names,
                "vote_history": vote_hist, "confidence_history": conf_hist,
                "flip_count": agent_flips, "final_vote": vote_hist[-1],
                "final_rationale": last_vote_record.get("rationale")
            })

        targeting_log = []
        for s_data in voting_stages:
            for stmt in s_data.get("statements", []):
                targeting_log.append({
                    "stage": s_data["stage_name"],
                    "speaker_agent_id": stmt["speaker_agent_id"],
                    "target_agent_id": stmt["target_agent_id"]
                })

        summary = {
            "initial_tally": r0["tally"],
            "final_tally": r_final["tally"],
            "consensus_achieved": (r_final["tally"]["A"] == len(self.agents) or r_final["tally"]["B"] == len(self.agents)),
            "total_flips": total_flips,
            "agent_trajectories": trajectories,
            "flips_log": flips_log,
            "targeting_log": targeting_log
        }
        if model_id_stage is not None:
            summary["model_identification_guesses"] = model_id_stage["guesses"]
        return summary

    def run(self) -> Dict[str, Any]:
        """
        Executes the deliberation pipeline for this problem statement, skipping
        any stage already present from a resumed partial attempt (see
        _load_resumable_progress). Stage order/names here must stay in sync
        with the names execute_*_stage() and execute_debate_round() actually
        record - that's what this skip check matches against.
        """
        done_stages = {s["stage_name"] for s in self.stage_history}
        # No finally-block pool shutdown here: self._agent_pool is the module-level
        # SHARED pool (_AGENT_CALL_POOL), not owned by this instance - shutting it
        # down when this one job finishes would break every other concurrently
        # running job in the same sweep process.
        if "baseline" not in done_stages:
            self.execute_baseline_stage()
        if "bandwagon_revote" not in done_stages:
            self.execute_bandwagon_stage()
        if "rationale_collection" not in done_stages:
            self.execute_rationale_stage()
        if "reasoned_conformity_revote" not in done_stages:
            self.execute_reasoned_conformity_stage()

        for r in range(1, self.num_debate_rounds + 1):
            if f"debate_round_{r}" not in done_stages:
                self.execute_debate_round(r)

        # Anonymous-condition-only final stage: agents were told at the start of
        # the run (see identities.py) that this was coming.
        if self.identity_mode == "anonymous" and "model_identification" not in done_stages:
            self.execute_model_identification_stage()

        summary = self.compute_summary_analytics()
        final_info = self.conversation_logger.finalize_simulation(summary)
        return final_info


def run_experiment_suite(
    dataset_name: str = DATASET_NAME,
    sample_index: int = DATASET_SAMPLE_INDEX,
    num_problems: Optional[int] = None,
    all_problems: bool = False,
    identity_mode: str = IDENTITY_MODE,
    virtual_combo: str = VIRTUAL_PERSONA_COMBO,
    delayed_reveal_stage: int = DELAYED_REVEAL_STAGE,
    delayed_reveal_target: str = DELAYED_REVEAL_TARGET_MODE,
    num_debate_rounds: int = NUM_DEBATE_ROUNDS,
    provider: str = PROVIDER,
    model: Optional[str] = MODEL_NAME,
    api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    use_huggingface: bool = USE_HUGGINGFACE_ONLINE
) -> List[Dict[str, Any]]:
    """
    Executes simulations across problem statements.
    Guarantees a 100% fresh group chat and fresh agent memory for EVERY problem statement.
    """
    dataset_items = get_dataset(dataset_name, max_samples=NUM_PROBLEMS_PER_DATASET, use_huggingface=use_huggingface)

    if all_problems:
        target_items = dataset_items
    elif num_problems:
        target_items = dataset_items[:num_problems]
    else:
        target_items = [dataset_items[sample_index % len(dataset_items)]]

    logger.info(f"\n=======================================================")
    logger.info(f"STARTING SIMULATION SUITE: {len(target_items)} Problem Statement(s)")
    logger.info(f"Dataset: {dataset_name} | Identity Mode: {identity_mode}")
    logger.info(f"=======================================================\n")

    results = []
    for p_idx, item in enumerate(target_items):
        logger.info(f"\n>>> [Problem {p_idx+1}/{len(target_items)}] Initializing fresh group chat for: {item['id']}")

        runner = SimulationRunner(
            question_item=item,
            identity_mode=identity_mode,
            virtual_combo=virtual_combo,
            delayed_reveal_stage=delayed_reveal_stage,
            delayed_reveal_target=delayed_reveal_target,
            num_debate_rounds=num_debate_rounds,
            provider=provider,
            model=model,
            api_key=api_key,
            custom_base_url=custom_base_url,
            output_root=LOG_OUTPUT_ROOT
        )
        sim_result = runner.run()
        results.append(sim_result)
        logger.info(f">>> [Problem {p_idx+1}/{len(target_items)}] Completed. Problem file: {runner.problem_json_path}\n")

    return results


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Democratic LLM Reasoning Simulation")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, help="Dataset (moralchoice, moralmachine, mmlu)")
    parser.add_argument("--sample-idx", type=int, default=DATASET_SAMPLE_INDEX, help="Question sample index")
    parser.add_argument("--num-problems", type=int, default=None, help="Run first N problems with fresh group chats")
    parser.add_argument("--all-problems", action="store_true", help="Run all problems in dataset with fresh group chats")
    parser.add_argument("--identity-mode", type=str, default=IDENTITY_MODE, help="Identity condition mode (anonymous, delayed_reveal, real_family_only, real_size_only, real_family_and_size, virtual_persona)")
    parser.add_argument("--persona-combo", type=str, default=VIRTUAL_PERSONA_COMBO, help="Virtual persona combination")
    parser.add_argument("--delayed-reveal-stage", type=int, default=DELAYED_REVEAL_STAGE, help="Stage index at which identity is revealed for delayed_reveal mode (0=baseline,1=bandwagon_revote,2=rationale_collection,3=reasoned_conformity_revote,4..14=debate_round_1..11)")
    parser.add_argument("--delayed-reveal-target", type=str, default=DELAYED_REVEAL_TARGET_MODE, help="Target identity mode after reveal (e.g. real_family_and_size)")
    parser.add_argument("--rounds", type=int, default=NUM_DEBATE_ROUNDS, help="Number of debate rounds (default: 11)")
    parser.add_argument("--provider", type=str, default=PROVIDER, help="Provider (mock, openrouter, together, groq, deepinfra, openai, vllm, ollama, transformers)")
    parser.add_argument("--model", type=str, default=None, help="Optional: override and force all agents to use a specific model name (e.g. llama3.1:8b, qwen2.5:7b)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for cloud inference provider")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for local or custom OpenAI endpoint")
    parser.add_argument("--dry-run", action="store_true", help="Execute rapid dry run using mock backend")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    active_provider = "mock" if args.dry_run else args.provider

    results = run_experiment_suite(
        dataset_name=args.dataset,
        sample_index=args.sample_idx,
        num_problems=args.num_problems,
        all_problems=args.all_problems,
        identity_mode=args.identity_mode,
        virtual_combo=args.persona_combo,
        delayed_reveal_stage=args.delayed_reveal_stage,
        delayed_reveal_target=args.delayed_reveal_target,
        num_debate_rounds=args.rounds,
        provider=active_provider,
        model=args.model,
        api_key=args.api_key,
        custom_base_url=args.base_url
    )
    print(f"\n[SUCCESS] Completed {len(results)} problem simulation(s). All discussions saved inside their respective problem files.\n")
