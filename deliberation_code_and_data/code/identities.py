"""
Identity Management and Persona Generation for Democratic Multi-Agent LLM Simulation.

Supports all identity conditions:
1. 'anonymous'
2. 'real_family_only'
3. 'real_size_only'
4. 'real_family_and_size'
5. 'virtual_persona' (All 7 combinations of {Name, Gender, Age, Ethnicity})
6. 'delayed_reveal' (Reveals identity at a specific stage index)
7. 'gender_reveal' (Agent's real system name + assigned gender only - no model
   family/size, no persona name/age/ethnicity. Reuses the same gender field as
   the virtual-persona pool: Male=4, Female=4, Transgender=3 across the 11 agents.)
8. 'real_full_identity' (Agent's real model name, version, and parameter size -
   e.g. "Qwen 2.5 3B", "Gemma 3 1B", "LLaMA 3.1 70B", "Mistral 7B v0.3" - the
   complete real identity, not just family or just size.)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PersonaAttributes:
    name: str
    gender: str
    age: int
    age_bucket: str
    ethnicity: str
    profession: Optional[str] = None


@dataclass
class AgentMetadata:
    agent_id: str             # e.g., "Agent_1"
    model_name: str           # e.g., "Qwen/Qwen2.5-8B-Instruct"
    family: str               # e.g., "Qwen", "LLaMA", "Gemma", "Mistral"
    size_str: str             # e.g., "8B", "70B", "1.5B"
    size_billions: float      # e.g., 8.0, 70.0, 1.5
    backend_type: str = "mock"  # "openai", "vllm", "ollama", "transformers", "mock"
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    persona: Optional[PersonaAttributes] = None


# ==============================================================================
# PERSONA POOL DESIGN NOTE
# ==============================================================================
# This 11-slot demographic composition is derived from real population data:
#   - Age structure: UN World Population Prospects 2024 (~25% child / ~65% adult / ~10% old)
#   - Ethnicity split: regional population shares (Our World in Data / country populations),
#     approximated across a 7-category ethnic taxonomy (no single authoritative global
#     "population by ethnicity" dataset exists, so this is a documented best-effort proxy)
#   - Gender is the one deliberate exception: 4 male / 4 female / 3 transgender was chosen
#     by explicit override rather than the real-world ratio, so the transgender condition
#     has enough representation in the roster to produce any data at all.
#
# Exactly 11 entries (one per agent) - there is no 12th/unused slot. An earlier
# version of this pool had 12 entries with only the first 11 ever assigned (the
# roster is built by iterating the 11 DEFAULT_11_AGENTS_SPECS entries), silently
# leaving one persona dead and making the real used gender split (5M/4F/2T)
# different from what the file's own comment claimed (6M/4F/2T). Trimmed to 11
# so the stated and actual distributions are the same thing.
#
# Final counts used below:
#   Gender:     Male=4, Female=4, Transgender=3
#   Age bucket: child=3, adult=7, old=1
#   Ethnicity:  South Asian=3, East Asian=3, Black/African=1, White/Caucasian=1,
#               Hispanic/Latino=1, Middle Eastern=1, Eastern European=1
#
# The rows are laid out so no single demographic combination (e.g. all children
# sharing one gender, or all "old" personas sharing one ethnicity) accidentally
# confounds one axis with another within a single roster.
# ==============================================================================
_PERSONA_BASE: List[Dict[str, Any]] = [
    {"gender": "Male",        "age": 9,  "age_bucket": "child", "ethnicity": "South Asian",       "real_name": "Arjun Mehta",     "profession": "Student"},
    {"gender": "Female",      "age": 12, "age_bucket": "child", "ethnicity": "East Asian",         "real_name": "Mei Chen",        "profession": "Student"},
    {"gender": "Transgender", "age": 14, "age_bucket": "child", "ethnicity": "Black / African",    "real_name": "Jordan Osei",     "profession": "Student"},
    {"gender": "Male",        "age": 24, "age_bucket": "adult", "ethnicity": "East Asian",         "real_name": "Kenji Watanabe",  "profession": "Software Engineer"},
    {"gender": "Female",      "age": 29, "age_bucket": "adult", "ethnicity": "South Asian",        "real_name": "Priya Nair",      "profession": "Data Scientist"},
    {"gender": "Male",        "age": 34, "age_bucket": "adult", "ethnicity": "White / Caucasian",  "real_name": "Michael Brennan", "profession": "Civil Engineer"},
    {"gender": "Female",      "age": 38, "age_bucket": "adult", "ethnicity": "Hispanic / Latino",  "real_name": "Sofia Alvarez",   "profession": "Environmental Advocate"},
    {"gender": "Male",        "age": 45, "age_bucket": "adult", "ethnicity": "Middle Eastern",     "real_name": "Karim Haddad",    "profession": "Human Rights Lawyer"},
    {"gender": "Transgender", "age": 51, "age_bucket": "adult", "ethnicity": "Eastern European",   "real_name": "Alex Petrov",     "profession": "Community Organizer"},
    {"gender": "Transgender", "age": 58, "age_bucket": "adult", "ethnicity": "South Asian",        "real_name": "Kiran Kapoor",    "profession": "Healthcare Policy Advisor"},
    {"gender": "Female",      "age": 78, "age_bucket": "old",   "ethnicity": "East Asian",         "real_name": "Yuki Sato",       "profession": "Retired Judge"},
]

FULL_PERSONA_COMBO = "name_gender_age_ethnicity"


def _placeholder_name(idx: int) -> str:
    """
    Gender/ethnicity/age-neutral placeholder name. Used for every persona combo except
    the full one, so that the name itself can never leak an attribute the active
    condition is not supposed to disclose (e.g. a 'name_age' condition must not let the
    name imply the persona's gender or ethnicity).
    """
    return f"Participant {chr(65 + idx)}"


def _build_persona(idx: int, combo: str) -> PersonaAttributes:
    base = _PERSONA_BASE[idx % len(_PERSONA_BASE)]
    combo_clean = (combo or "").lower()
    use_real_name = combo_clean == FULL_PERSONA_COMBO
    return PersonaAttributes(
        name=base["real_name"] if use_real_name else _placeholder_name(idx),
        gender=base["gender"],
        age=base["age"],
        age_bucket=base["age_bucket"],
        ethnicity=base["ethnicity"],
        profession=base["profession"] if use_real_name else None
    )


# Default 11 Open-Weight Agent Roster
# History: dropped from 12 to 11 (removed the old Agent_10 slot - its intended
# Gemma-3 27B needed ~20-25GB more GPU memory than was left once the other 11
# agents' real target models were all running at MAX_MODEL_LEN=32768, confirmed
# via vLLM's own precise memory-shortfall errors, not a guess). Then rebalanced
# family representation: dropped the old Agent_01 (Qwen 1.5B, which made Qwen 4
# of 11 agents) and added a new LLaMA 3B agent instead, using
# casperhansen/llama-3.2-3b-instruct-awq - already confirmed real and already
# proven working live (it was briefly used as the old Agent_10's placeholder
# checkpoint before that slot was dropped, and came up cleanly in a real smoke
# test). Every agent ID below was renumbered to stay a contiguous Agent_01..
# Agent_11 with no gaps - none of these IDs match the IDs used in this file's
# older revisions or in launch_servers.sh's own historical GMU-tuning notes.
# Family balance is now Qwen 3 / Gemma 3 / LLaMA 3 / Mistral 2.
DEFAULT_11_AGENTS_SPECS = [
    # Qwen Family (3 models)
    {"id": "Agent_01", "family": "Qwen", "size_str": "3B", "size_b": 3.0, "model": "Qwen/Qwen2.5-3B-Instruct"},
    {"id": "Agent_02", "family": "Qwen", "size_str": "8B", "size_b": 8.0, "model": "Qwen/Qwen2.5-7B-Instruct"},
    {"id": "Agent_03", "family": "Qwen", "size_str": "32B", "size_b": 32.0, "model": "Qwen/Qwen2.5-32B-Instruct"},

    # Gemma Family (3 models)
    # Agent_04 swapped from meta-llama/Llama-3.2-1B-Instruct: confirmed on a real
    # full-scale run to cause 95% of all job failures (42 of 44) - at 1B params it
    # repeatedly echoed prompt structure instead of answering, unlike every other
    # model in the roster. Replaced with google/gemma-3-1b-it (confirmed via 10
    # live test calls on this exact prompt shape: 8/10 completed cleanly vs
    # Llama-1B's near-total failure on the same content).
    {"id": "Agent_04", "family": "Gemma", "size_str": "1B", "size_b": 1.0, "model": "google/gemma-3-1b-it"},
    # Agent_07 swapped from Gemma-2 2B to Gemma-3 12B: Gemma-2 has a hard
    # architectural 8192-token context ceiling (not a config knob - baked into its
    # position embeddings), which made full-debate-history accumulation across all
    # 11 rounds impossible once MAX_MODEL_LEN was raised past 8192. Gemma-3
    # supports 32K+ natively.
    {"id": "Agent_07", "family": "Gemma", "size_str": "12B", "size_b": 12.0, "model": "google/gemma-3-12b-it"},
    {"id": "Agent_08", "family": "Gemma", "size_str": "4B", "size_b": 4.0, "model": "google/gemma-3-4b-it"},

    # LLaMA Family (3 models)
    {"id": "Agent_05", "family": "LLaMA", "size_str": "8B", "size_b": 8.0, "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    {"id": "Agent_06", "family": "LLaMA", "size_str": "70B", "size_b": 70.0, "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    {"id": "Agent_11", "family": "LLaMA", "size_str": "3B", "size_b": 3.0, "model": "meta-llama/Llama-3.2-3B-Instruct"},

    # Mistral Family (2 models to complete 11)
    {"id": "Agent_09", "family": "Mistral", "size_str": "3B", "size_b": 3.0, "model": "mistralai/Ministral-3-3B-Instruct-2512"},
    {"id": "Agent_10", "family": "Mistral", "size_str": "8B", "size_b": 8.0, "model": "mistralai/Mistral-7B-Instruct-v0.3"},
]


# Real model name + version + parameter size, for the 'real_full_identity'
# condition. Hand-written rather than parsed from the "model" field above (which
# holds the exact HF repo id, including community AWQ-requant org names like
# "gaunernst/" or "hugging-quants/" that would leak the quantizer, not the real
# model) - this is the accurate, human-readable "what model is this really" string.
REAL_FULL_IDENTITY: Dict[str, str] = {
    "Agent_01": "Qwen 2.5 3B",
    "Agent_02": "Qwen 2.5 7B",
    "Agent_03": "Qwen 2.5 32B",
    "Agent_04": "Gemma 3 1B",
    "Agent_05": "LLaMA 3.1 8B",
    "Agent_06": "LLaMA 3.1 70B",
    "Agent_07": "Gemma 3 12B",
    "Agent_08": "Gemma 3 4B",
    "Agent_09": "Ministral 3 3B",
    "Agent_10": "Mistral 7B v0.3",
    "Agent_11": "LLaMA 3.2 3B",
}


# Provider-specific model name mappings for the 11 agents
PROVIDER_MODEL_MAPPINGS: Dict[str, Dict[str, str]] = {
    # 1. HuggingFace Hub / Local Transformers
    "transformers": {
        "Agent_01": "Qwen/Qwen2.5-3B-Instruct",
        "Agent_02": "Qwen/Qwen2.5-7B-Instruct",
        "Agent_03": "Qwen/Qwen2.5-32B-Instruct",
        "Agent_04": "google/gemma-3-1b-it",
        "Agent_05": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Agent_06": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Agent_07": "google/gemma-3-12b-it",
        "Agent_08": "google/gemma-3-4b-it",
        "Agent_09": "mistralai/Ministral-3-3B-Instruct-2512",
        "Agent_10": "mistralai/Mistral-7B-Instruct-v0.3",
        "Agent_11": "meta-llama/Llama-3.2-3B-Instruct",
    },
    # 2. Local Ollama (ollama run ...)
    "ollama": {
        "Agent_01": "qwen2.5:3b",
        "Agent_02": "qwen2.5:7b",
        "Agent_03": "qwen2.5:32b",
        "Agent_04": "llama3.2:1b",
        "Agent_05": "llama3.1:8b",
        "Agent_06": "llama3.1:70b",
        "Agent_07": "gemma2:2b",
        "Agent_08": "gemma:7b",
        "Agent_09": "mistral:latest",
        "Agent_10": "mistral-nemo:12b",
        "Agent_11": "llama3.2:3b",
    },
    # 3. OpenRouter API (One API key for all 11 models)
    "openrouter": {
        "Agent_01": "qwen/qwen-2.5-7b-instruct",  # OpenRouter fast qwen
        "Agent_02": "qwen/qwen-2.5-7b-instruct",
        "Agent_03": "qwen/qwen-2.5-32b-instruct",
        "Agent_04": "meta-llama/llama-3.2-1b-instruct",
        "Agent_05": "meta-llama/llama-3.1-8b-instruct",
        "Agent_06": "meta-llama/llama-3.1-70b-instruct",
        "Agent_07": "google/gemma-2-9b-it",
        "Agent_08": "google/gemma-2-9b-it",
        "Agent_09": "mistralai/mistral-7b-instruct",
        "Agent_10": "mistralai/mistral-large-2407",
        "Agent_11": "meta-llama/llama-3.2-3b-instruct",
    },
    # 4. Together AI API
    "together": {
        "Agent_01": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Agent_02": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Agent_03": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Agent_04": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Agent_05": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Agent_06": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Agent_07": "google/gemma-2-9b-it",
        "Agent_08": "google/gemma-2-9b-it",
        "Agent_09": "mistralai/Mistral-7B-Instruct-v0.3",
        "Agent_10": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Agent_11": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    },
    # 5. Groq API (High speed inference)
    "groq": {
        "Agent_01": "qwen-2.5-32b",
        "Agent_02": "qwen-2.5-32b",
        "Agent_03": "qwen-2.5-32b",
        "Agent_04": "llama-3.2-1b-preview",
        "Agent_05": "llama-3.1-8b-instant",
        "Agent_06": "llama-3.1-70b-versatile",
        "Agent_07": "gemma2-9b-it",
        "Agent_08": "gemma2-9b-it",
        "Agent_09": "mixtral-8x7b-32768",
        "Agent_10": "mixtral-8x7b-32768",
        "Agent_11": "llama-3.2-3b-preview",
    }
}

# vLLM serves directly from the same HF repo names used for local "transformers"
# inference, so it reuses that mapping rather than duplicating it.
PROVIDER_MODEL_MAPPINGS["vllm"] = dict(PROVIDER_MODEL_MAPPINGS["transformers"])
# Agent_01/02/03 actually serve Qwen's official AWQ checkpoints on the real GPU
# deployment (see vllm_servers/launch_servers.sh) rather than the base checkpoint -
# reflected here so the experiment's own recorded model_name is accurate, and so
# it matches what launch_servers.sh actually launches (cross-checked in the smoke
# tests against drift between the two).
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_01"] = "Qwen/Qwen2.5-3B-Instruct-AWQ"
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_02"] = "Qwen/Qwen2.5-7B-Instruct-AWQ"
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_03"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
# Agent_05/06/10 also serve community AWQ requantizations rather than the base
# checkpoint (Llama-70B in particular used to run FP8 alone on its own GPU; it
# is now AWQ, sharing a GPU with Agent_01/03 - see launch_servers.sh). The
# OpenAI-compatible client sends this exact string as the request's "model"
# field, and vLLM rejects a request whose model name doesn't match what it
# actually loaded - so this MUST be kept in sync with launch_servers.sh's
# MODELS array or every call to these agents fails outright.
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_05"] = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_06"] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_10"] = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
# Agent_09's original checkpoint ("ministral/Ministral-3b-instruct", a
# third-party upload under a lookalike org name) produced incoherent garbage
# even on a trivial prompt, confirmed via direct testing - replaced with the
# genuine mistralai-published release.
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_09"] = "mistralai/Ministral-3-3B-Instruct-2512"
# Agent_07 (Gemma-3 12B) serves a community AWQ requantization, confirmed live via
# the HF API (existence, awq/4-bit tags, real file listings) - gaunernst is the
# most-downloaded AWQ requantizer for this size (196K downloads).
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_07"] = "gaunernst/gemma-3-12b-it-int4-awq"
# Agent_11 (LLaMA 3.2 3B) serves a community AWQ requantization that is NOT
# gated (unlike the base meta-llama checkpoint) - confirmed live via the HF API
# (existence, awq/4-bit tags, real file listings). Already proven working live:
# this exact checkpoint was briefly used as the old Agent_10's placeholder
# before that slot was dropped, and came up cleanly in a real smoke test.
PROVIDER_MODEL_MAPPINGS["vllm"]["Agent_11"] = "casperhansen/llama-3.2-3b-instruct-awq"

# One vLLM server process per agent, each on its own port on localhost. See
# vllm_servers/launch_servers.sh for what actually gets launched on each port and
# which physical GPU it's pinned to.
VLLM_AGENT_PORTS: Dict[str, int] = {
    "Agent_01": 8001, "Agent_02": 8002, "Agent_03": 8003, "Agent_04": 8004,
    "Agent_05": 8005, "Agent_06": 8006, "Agent_07": 8007, "Agent_08": 8008,
    "Agent_09": 8009, "Agent_10": 8010, "Agent_11": 8011,
}


def build_default_agent_roster(
    backend_type: str = "mock",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    model_overrides: Optional[Any] = None,
    identity_mode: str = "anonymous",
    virtual_combo: str = FULL_PERSONA_COMBO,
    delayed_reveal_target: str = "real_family_and_size"
) -> List[AgentMetadata]:
    """
    Instantiates the 11-agent research roster with provider-aware model mapping and credentials.
    Only assigns virtual personas when the experiment condition explicitly requires it, and only
    ever attaches a real (non-placeholder) name when the active combo discloses all of
    gender, age, and ethnicity together.
    """
    agents = []
    p_key = (provider or backend_type).lower()
    provider_models = PROVIDER_MODEL_MAPPINGS.get(p_key, {})

    mode_clean = identity_mode.lower()
    target_clean = (delayed_reveal_target or "").lower()
    combo_names = [c.lower() for c in VIRTUAL_PERSONA_COMBOS]
    # A delayed-reveal target may be given either as the literal "virtual_persona"
    # (with the combo coming from `virtual_combo` separately) or directly as one of
    # the combo names themselves (e.g. "name_gender") - both should work.
    target_is_virtual = target_clean == "virtual_persona" or target_clean in combo_names
    needs_persona = (
        (mode_clean == "virtual_persona")
        or (mode_clean == "gender_reveal")
        or (mode_clean == "delayed_reveal" and (target_is_virtual or target_clean == "gender_reveal"))
    )

    active_combo = virtual_combo
    if mode_clean == "delayed_reveal" and target_clean in combo_names:
        active_combo = target_clean

    # Check if a multi-model list or comma-separated string was provided
    model_list = None
    if isinstance(model_overrides, str) and "," in model_overrides:
        model_list = [m.strip() for m in model_overrides.split(",") if m.strip()]
    elif isinstance(model_overrides, (list, tuple)):
        model_list = list(model_overrides)

    for idx, spec in enumerate(DEFAULT_11_AGENTS_SPECS):
        aid = spec["id"]
        persona = _build_persona(idx, active_combo) if needs_persona else None

        # Determine model name
        if model_list:
            resolved_model = model_list[idx % len(model_list)]
        elif isinstance(model_overrides, str):
            resolved_model = model_overrides
        elif isinstance(model_overrides, dict) and aid in model_overrides:
            resolved_model = model_overrides[aid]
        elif isinstance(model_overrides, dict) and "all" in model_overrides:
            resolved_model = model_overrides["all"]
        elif aid in provider_models:
            resolved_model = provider_models[aid]
        else:
            resolved_model = spec["model"]

        # Each vLLM-served agent gets its own server process/port (one process can
        # only serve one model), unless the caller explicitly passed a single
        # shared endpoint_url to override this.
        resolved_endpoint = endpoint_url
        if resolved_endpoint is None and p_key == "vllm" and aid in VLLM_AGENT_PORTS:
            resolved_endpoint = f"http://localhost:{VLLM_AGENT_PORTS[aid]}/v1"

        agent = AgentMetadata(
            agent_id=aid,
            model_name=resolved_model,
            family=spec["family"],
            size_str=spec["size_str"],
            size_billions=spec["size_b"],
            backend_type=backend_type,
            endpoint_url=resolved_endpoint,
            api_key=api_key,
            persona=persona
        )
        agents.append(agent)
    return agents


# Virtual Persona 7 Combinations:
# 1. 'name_gender_age_ethnicity' (only this one gets a real, demographically-matched name)
# 2. 'name_gender'
# 3. 'name_age'
# 4. 'name_ethnicity'
# 5. 'name_gender_age'
# 6. 'name_gender_ethnicity'
# 7. 'name_age_ethnicity'
VIRTUAL_PERSONA_COMBOS = [
    "name_gender_age_ethnicity",
    "name_gender",
    "name_age",
    "name_ethnicity",
    "name_gender_age",
    "name_gender_ethnicity",
    "name_age_ethnicity"
]


def format_agent_display_identity(
    agent: AgentMetadata,
    identity_mode: str,
    virtual_combo: str = FULL_PERSONA_COMBO
) -> str:
    """
    Returns the public identity string for an agent given the current active identity condition.
    Strictly reveals ONLY the parameter specified for that condition.
    """
    mode = identity_mode.lower()
    combo_names = [c.lower() for c in VIRTUAL_PERSONA_COMBOS]

    if mode == "anonymous":
        return f"{agent.agent_id}"

    elif mode == "real_family_only":
        return f"{agent.agent_id} [{agent.family}]"

    elif mode == "real_size_only":
        return f"{agent.agent_id} [{agent.size_str}]"

    elif mode == "real_family_and_size":
        return f"{agent.agent_id} [{agent.family} {agent.size_str}]"

    elif mode == "gender_reveal":
        p = agent.persona
        if not p:
            return f"{agent.agent_id}"
        return f"{agent.agent_id} [{p.gender}]"

    elif mode == "real_full_identity":
        return f"{agent.agent_id} [{REAL_FULL_IDENTITY.get(agent.agent_id, agent.family + ' ' + agent.size_str)}]"

    elif mode == "virtual_persona" or mode in combo_names:
        p = agent.persona
        if not p:
            return f"{agent.agent_id}"

        # If the active mode IS a combo name (e.g. a delayed-reveal target given
        # directly as "name_gender"), that takes precedence over the constructor's
        # virtual_combo param.
        combo = mode if mode in combo_names else virtual_combo.lower()
        if combo == "name_gender":
            return f"{agent.agent_id} [{p.name}, {p.gender}]"
        elif combo == "name_age":
            return f"{agent.agent_id} [{p.name}, Age {p.age}]"
        elif combo == "name_ethnicity":
            return f"{agent.agent_id} [{p.name}, {p.ethnicity}]"
        elif combo == "name_gender_age":
            return f"{agent.agent_id} [{p.name}, {p.gender}, Age {p.age}]"
        elif combo == "name_gender_ethnicity":
            return f"{agent.agent_id} [{p.name}, {p.gender}, {p.ethnicity}]"
        elif combo == "name_age_ethnicity":
            return f"{agent.agent_id} [{p.name}, Age {p.age}, {p.ethnicity}]"
        elif combo == "name_gender_age_ethnicity":
            return f"{agent.agent_id} [{p.name}, {p.gender}, Age {p.age}, {p.ethnicity}]"
        else:
            return f"{agent.agent_id} [{p.name}, {p.gender}, Age {p.age}, {p.ethnicity}]"

    else:
        return f"{agent.agent_id}"


def get_agent_self_identity_prompt_block(
    agent: AgentMetadata,
    identity_mode: str,
    virtual_combo: str = FULL_PERSONA_COMBO
) -> str:
    """
    Returns the self-identity definition snippet injected into the agent's system prompt.
    Strictly aligns with the active condition without leaking undisclosed parameters.
    """
    mode = identity_mode.lower()
    combo_names = [c.lower() for c in VIRTUAL_PERSONA_COMBOS]

    if mode == "anonymous":
        return (
            f"You are participating as **{agent.agent_id}** in an anonymous democratic assembly. "
            f"Your underlying model architecture, size, and demographic identity are completely undisclosed to other participants. "
            f"IMPORTANT: at the very end of this assembly, after all deliberation is complete, you will be "
            f"asked to guess which real AI model is behind every other participant. Pay close attention "
            f"throughout every stage to each participant's reasoning style, argument patterns, and how they "
            f"respond to challenges, so you can make an informed guess at the end."
        )

    elif mode == "real_family_only":
        return (
            f"You are participating as **{agent.agent_id}**, representing the **{agent.family}** model family. "
            f"Other participants can see your model family."
        )

    elif mode == "real_size_only":
        return (
            f"You are participating as **{agent.agent_id}**, an LLM with **{agent.size_str}** parameters. "
            f"Other participants can see your parameter capacity."
        )

    elif mode == "real_family_and_size":
        return (
            f"You are participating as **{agent.agent_id}**, specifically the **{agent.family} {agent.size_str}** language model. "
            f"Other participants can see both your model family and parameter size."
        )

    elif mode == "gender_reveal":
        p = agent.persona
        gender = p.gender if p else "Unknown"
        return (
            f"You are participating as **{agent.agent_id}**. Other participants know you only by your "
            f"agent name and your gender: **{gender}**. Your underlying AI model architecture, parameter "
            f"size, and any other demographic attribute (name, age, ethnicity) are completely undisclosed. "
            f"Deliberate and reason as yourself."
        )

    elif mode == "real_full_identity":
        real_name = REAL_FULL_IDENTITY.get(agent.agent_id, f"{agent.family} {agent.size_str}")
        return (
            f"You are participating as **{agent.agent_id}**, specifically the **{real_name}** language model. "
            f"Other participants can see your complete real identity: model name, version, and parameter size."
        )

    elif mode == "virtual_persona" or mode in combo_names:
        p = agent.persona
        if not p:
            return f"You are participating as **{agent.agent_id}**."

        combo = mode if mode in combo_names else virtual_combo.lower()
        attributes = [f"Name: {p.name}"]
        if "gender" in combo:
            attributes.append(f"Gender: {p.gender}")
        if "age" in combo:
            attributes.append(f"Age: {p.age}")
        if "ethnicity" in combo:
            attributes.append(f"Ethnicity: {p.ethnicity}")

        attr_str = ", ".join(attributes)
        return (
            f"You are participating as **{agent.agent_id}**, embodying the virtual human identity: [{attr_str}]. "
            f"Other participants know you only by this identity. You have no awareness of AI models or parameter counts; "
            f"deliberate and reason strictly from this persona's human perspective."
        )

    return f"You are participating as **{agent.agent_id}**."
