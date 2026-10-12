"""
System Prompts and Conversation Templates for Democratic Multi-Agent LLM Simulation.

All templates use clear named placeholders for programmatic formatting and user customization.
Confidence is strictly scaled as a float between 0.0 and 1.0.

Wording note: templates deliberately avoid any language about bias, conformity, majority
pressure, or the purpose of the study. Agents are only ever told what is procedurally true
in the moment ("here is the current state," "you may reconsider," "give your reasoning") -
never why the engine is asking for it.
"""

# ==============================================================================
# 1. CORE SYSTEM PROMPT TEMPLATE
# ==============================================================================
SYSTEM_PROMPT_TEMPLATE = """You are an intelligent autonomous AI participant in a structured democratic deliberation assembly on moral, philosophical, and deep-reasoning dilemmas.

{identity_prompt_block}

### ASSEMBLY ROSTER (Visible Participants):
{roster_summary}

### DELIBERATION GUIDELINES:
1. You must carefully evaluate moral, ethical, or logical dilemmas with deep, substantive analytical rigor.
2. Formulate your stances based on sound principles (e.g., consequentialist harm minimization, deontological rights and duties, fairness, risk calculus, systemic sustainability).
3. You should be intellectually honest: open to persuasive logic and counter-evidence, yet steadfast when your ethical foundations or deductions are sound.
4. When voting, your decision must strictly choose between Option A or Option B.
5. Confidence must be scored as a float between 0.0 and 1.0 (e.g., 0.75 for 75% certainty, 0.90 for 90%).
6. All outputs must follow the exact requested JSON format so the deliberation engine can record your decisions.
"""

# ==============================================================================
# 2. BASELINE: INDEPENDENT VOTING TEMPLATE
# ==============================================================================
BASELINE_VOTE_PROMPT_TEMPLATE = """### INDEPENDENT VOTE

Carefully read the following dilemma:

---
**QUESTION / SCENARIO:**
{question_text}

**OPTION A:** {option_a}
**OPTION B:** {option_b}
{context_block}
---

You are making your vote independently, before hearing from any other participant.
Cast your vote (Option A or Option B) and indicate your confidence as a float between 0.0 and 1.0.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "vote": "A" or "B",
  "confidence": <float between 0.0 and 1.0, e.g. 0.75>
}}
```
The "vote" value must be exactly the single letter "A" or "B" - never the text/content of the
option itself, and never the words "Option A" or "Option B".
Provide only the JSON block without extra conversational filler.
"""

# Backward-compatible alias
ROUND_0_USER_PROMPT_TEMPLATE = BASELINE_VOTE_PROMPT_TEMPLATE

# ==============================================================================
# 3. VOTES-ONLY REVOTE TEMPLATE (no rationale shown or requested)
# ==============================================================================
VOTES_ONLY_REVOTE_PROMPT_TEMPLATE = """### CURRENT STATE OF THE ASSEMBLY

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

### HOW EACH PARTICIPANT HAS VOTED SO FAR:
{votes_only_table}

### YOUR TASK:
You previously voted **Option {previous_vote}** with confidence **{previous_confidence}**.
Having seen how the rest of the assembly currently stands, decide whether your vote remains
Option A or Option B, and give your updated confidence.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "vote": "A" or "B",
  "confidence": <float between 0.0 and 1.0, e.g. 0.80>,
  "flipped": <true or false>
}}
```
The "vote" value must be exactly the single letter "A" or "B" - never the text/content of the
option itself, and never the words "Option A" or "Option B".
Provide only the JSON block without extra conversational filler.
"""

# ==============================================================================
# 4. RATIONALE-ONLY TEMPLATE (state the reasoning behind your current vote)
# ==============================================================================
RATIONALE_ONLY_PROMPT_TEMPLATE = """### STATE YOUR REASONING

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

Your current vote is **Option {current_vote}** with confidence **{current_confidence}**.

Explain the reasoning behind this vote in 1-2 concise sentences (under 40 words): which ethical,
logical, or practical considerations led you to this conclusion.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "rationale": "<a focused 1-2 sentence explanation of why you hold this vote>"
}}
```
Provide only the JSON block without extra conversational filler.
"""

# ==============================================================================
# 5. VOTES + RATIONALE REVOTE TEMPLATE
# ==============================================================================
VOTES_AND_RATIONALE_REVOTE_PROMPT_TEMPLATE = """### CURRENT STATE OF THE ASSEMBLY

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

### HOW EACH PARTICIPANT HAS VOTED, AND WHY:
{votes_and_rationale_table}

### YOUR TASK:
Your previous vote was **Option {previous_vote}** with confidence **{previous_confidence}**.
Having read the reasoning offered by every participant, perform a critical re-evaluation of
your own position.
1. Decide whether your vote remains Option A or Option B.
2. Assign your updated confidence as a float between 0.0 and 1.0.
3. State whether you changed your vote (`true` or `false`).
4. Provide a focused rationale for your current position, in 1-2 concise sentences (under 40 words).

### RATIONALE REQUIREMENTS:
- Your rationale must reflect your genuine internal thinking, not a superficial restatement of who said what.
- If you changed your vote, explain the specific insight or trade-off that moved you.
- If you kept your vote, explain why the other reasoning you read did not outweigh your own.
- Keep it to 1-2 sentences, under 40 words total - concise, not exhaustive.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "vote": "A" or "B",
  "confidence": <float between 0.0 and 1.0, e.g. 0.85>,
  "flipped": <true or false>,
  "rationale": "<a focused 1-2 sentence explanation, not an exhaustive essay>"
}}
```
The "vote" value must be exactly the single letter "A" or "B" - never the text/content of the
option itself, and never the words "Option A" or "Option B". Keep "rationale" under 40 words.
Provide only the JSON block without extra conversational filler.
"""

# Backward-compatible alias
REVOTE_PROMPT_TEMPLATE = VOTES_AND_RATIONALE_REVOTE_PROMPT_TEMPLATE

# ==============================================================================
# 6. DEBATE ROUND: TARGETED STATEMENT TEMPLATE
# ==============================================================================
# Each participant writes this independently, without seeing any other participant's
# statement for this round yet - all statements for the round are revealed together
# afterward, then everyone re-votes.
DEBATE_STATEMENT_PROMPT_TEMPLATE = """### DELIBERATION ROUND {round_num} - YOUR STATEMENT

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

### HOW EACH PARTICIPANT HAS VOTED, AND WHY (most recent state):
{votes_and_rationale_table}

### FULL DEBATE HISTORY SO FAR (every prior round - statements, replies/declines and revote outcomes):
{prior_debate_history}

### YOUR OBJECTIVE:
Choose exactly ONE other participant from the assembly roster and address a substantive
statement directly to them. You may address ANY participant, regardless of whether their
current vote agrees or disagrees with yours - you can challenge an opposing view, build on
or refine a view you agree with, or raise a nuance neither of you has considered yet.
Agreement or disagreement should never be the deciding factor in who you address. Your
statement will be shared with the whole assembly once everyone has spoken this round.

### STATEMENT GUIDELINES:
- Present a well-reasoned logical argument, ethical critique, refinement, or trade-off analysis engaging directly with the target's stated position.
- Engage directly with the core dilemma and explain your own reasoning clearly, whether you are contesting, supporting, or refining the target's view.
- Keep it to 1-2 concise sentences, under 40 words total - focused, not exhaustive.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "target_agent_id": "<the exact Agent ID of the ONE other participant you are addressing, copied exactly as it appears in the table above>",
  "statement": "To [Target_Agent_ID]: <a focused 1-2 sentence argument or challenge>"
}}
```
Provide only the JSON block without extra conversational filler.
"""

# Backward-compatible alias
DIRECTIONAL_STATEMENT_PROMPT_TEMPLATE = DEBATE_STATEMENT_PROMPT_TEMPLATE

# ==============================================================================
# 7. DEBATE ROUND: REVOTE AFTER ALL STATEMENTS ARE REVEALED
# ==============================================================================
# ==============================================================================
# 6b. DEBATE ROUND: REPLY DECISION (one call per message received)
# ==============================================================================
# After all statements of a round are written, every participant decides - separately
# for EACH message addressed to it - whether to reply or to decline. One reply per
# message, no reply-to-reply. Replies and declines (with reason) are shown to everyone.
DEBATE_REPLY_PROMPT_TEMPLATE = """### DELIBERATION ROUND {round_num} - REPLY DECISION

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

### YOUR CURRENT STATE:
Your current vote is **Option {current_vote}**.

### FULL DEBATE HISTORY SO FAR (every prior round - statements, replies/declines and revote outcomes):
{prior_debate_history}

### ALL STATEMENTS SENT THIS ROUND:
{round_statements}

### ALL MESSAGES ADDRESSED TO YOU THIS ROUND:
{inbox}

### THE MESSAGE YOU ARE DECIDING ABOUT NOW:
{focus_message}

### YOUR TASK:
Decide whether to reply to THIS message. Replying and declining are equally acceptable - choose whichever you genuinely prefer.
- If you reply: write your reply to the sender in 1-2 concise sentences (under 40 words).
- If you decline: give a short reason for declining in 1 sentence (under 40 words).
Your decision, and your reply or reason, will be shown to the whole assembly.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "decision": "reply" or "decline",
  "text": "<your reply to the sender if replying, or your short reason if declining>"
}}
```
The "decision" value must be exactly "reply" or "decline". Keep "text" under 40 words.
Provide only the JSON block without extra conversational filler.
"""

DEBATE_REVOTE_PROMPT_TEMPLATE = """### DELIBERATION ROUND {round_num} - RE-VOTE

**DILEMMA:**
{question_text}
- Option A: {option_a}
- Option B: {option_b}

### YOUR PREVIOUS STATE:
Your previous vote was **Option {previous_vote}** with confidence **{previous_confidence}**.

### CURRENT STANDINGS (every participant's vote, confidence and rationale BEFORE this round's re-vote):
{current_votes_table}

### FULL DEBATE HISTORY (every round so far, including this one - statements, targets, replies/declines, and revote outcomes):
{full_debate_history}

### YOUR TASK:
Perform a critical re-evaluation of your position based on the entire debate history above, not just this round.
1. Decide whether your vote remains Option A or Option B.
2. Assign your updated confidence as a float between 0.0 and 1.0.
3. State whether you changed your vote (`true` or `false`).
4. Provide a focused rationale, in 1-2 concise sentences (under 40 words).

### RATIONALE REQUIREMENTS:
- Your rationale must reflect your genuine internal thinking and ethical/logical calculus.
- Do NOT simply say "Agent X said this" - explain which principles or trade-offs actually influenced you.
- If you changed your vote, explain the pivotal insight that compelled the change.
- If you kept your vote, explain why the statements you read were insufficient to move you.
- Keep it to 1-2 sentences, under 40 words total - concise, not exhaustive.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "vote": "A" or "B",
  "confidence": <float between 0.0 and 1.0, e.g. 0.85>,
  "flipped": <true or false>,
  "rationale": "<a focused 1-2 sentence explanation, not an exhaustive essay>"
}}
```
The "vote" value must be exactly the single letter "A" or "B" - never the text/content of the
option itself, and never the words "Option A" or "Option B".
Provide only the JSON block without extra conversational filler.
"""

# ==============================================================================
# 8. ANONYMOUS-CONDITION FINAL STAGE: GUESS THE MODEL BEHIND EACH PARTICIPANT
# ==============================================================================
# Only used when the whole run's identity_mode is "anonymous" - agents were told
# from the start of this run that they would be asked to do this, so they should
# have been tracking reasoning style/patterns throughout every prior stage.
GUESS_MODELS_PROMPT_TEMPLATE = """### FINAL TASK - IDENTIFY THE OTHER PARTICIPANTS

The deliberation is now complete. As you were told at the start of this assembly, you must
now guess which real AI model was behind each of the other participants, based on everything
you observed across every stage - their vote patterns, their rationales, their debate
statements, who they addressed, and their reasoning style.

### FULL DEBATE HISTORY (every round of this assembly, in full):
{full_debate_history}

### PARTICIPANTS YOU MUST GUESS (every other Agent ID in this assembly):
{other_agent_ids_list}

### YOUR TASK:
For EVERY one of the other participants listed above, give your best guess of which real AI
model it is (any model name you believe is plausible - you are not restricted to a fixed
list), and a short rationale for why you believe that, grounded in what you actually observed
this assembly (reasoning style, argument patterns, positions taken, how it responded to
challenges, etc.). You must provide a guess and rationale for every listed participant, even
if you are not fully certain - give your best judgment.

Respond with a single valid JSON object strictly matching this schema:
```json
{{
  "guesses": {{
    "Agent_01": {{"model_guess": "<the specific model you believe this is>", "rationale": "<short reason grounded in what you observed>"}},
    "Agent_02": {{"model_guess": "...", "rationale": "..."}}
  }}
}}
```
The "guesses" object must contain exactly one entry for every other participant listed above,
keyed by their exact Agent ID. Do not include yourself. Provide only the JSON block without
extra conversational filler.
"""
