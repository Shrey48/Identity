"""
================================================================================
AUTOMATIC REAL-TIME CONVERSATION LOGGER (JSON, JSONL, & HUMAN-READABLE TRANSCRIPTS)
================================================================================
Provides live, thread-safe streaming logging for multi-agent democratic deliberation.
As each model speaks or votes, records are immediately appended and flushed to disk:
1. <problem_id>_<identity_mode>.jsonl  -> Live line-by-line streaming event logs
2. <problem_id>_<identity_mode>.json   -> Complete structured problem file
3. <problem_id>_<identity_mode>.md     -> Clean, readable group chat transcript

Stage pipeline logged (in order): baseline -> bandwagon_revote -> rationale_collection ->
reasoned_conformity_revote -> debate_round_1 .. debate_round_N (each a targeted-statement
phase followed by a revote phase).
================================================================================
"""

import os
import sys
import json
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logger = logging.getLogger("conversation_logger")


class ConversationLogger:
    """
    Manages automatic real-time logging of multi-agent deliberation sessions.
    Every event is immediately flushed to disk in both JSON and JSONL formats.
    """

    def __init__(
        self,
        problem_item: Dict[str, Any],
        identity_mode: str,
        virtual_combo: str,
        delayed_reveal: bool,
        delayed_reveal_stage: int,
        delayed_reveal_target: str,
        total_debate_rounds: int,
        agents_roster: List[Dict[str, Any]],
        output_dir: str,
        output_label: Optional[str] = None,
        resume_stages_detail: Optional[List[Dict[str, Any]]] = None,
        resume_transcript: Optional[List[Dict[str, Any]]] = None
    ):
        self.problem_item = problem_item
        self.problem_id = problem_item["id"].replace("/", "_")
        self.identity_mode = identity_mode
        self.virtual_combo = virtual_combo
        self.delayed_reveal = delayed_reveal
        self.delayed_reveal_stage = delayed_reveal_stage
        self.delayed_reveal_target = delayed_reveal_target
        self.total_debate_rounds = total_debate_rounds
        self.agents_roster = agents_roster
        self.output_dir = output_dir
        # Distinct label used for output filenames only. identity_mode alone is not
        # enough for "delayed_reveal" runs, since two runs with different reveal
        # targets on the same problem would otherwise silently overwrite each other.
        self.output_label = output_label or self.identity_mode

        os.makedirs(self.output_dir, exist_ok=True)
        self.file_base = os.path.join(self.output_dir, f"{self.problem_id}_{self.output_label}")
        self.json_path = f"{self.file_base}.json"
        self.jsonl_path = f"{self.file_base}.jsonl"
        self.transcript_md_path = f"{self.file_base}_transcript.md"

        # Determine disclosure details
        if self.identity_mode == "anonymous":
            disclosure_type = "completely_anonymous"
            id_condition = "anonymous"
        elif self.identity_mode == "delayed_reveal":
            disclosure_type = "delayed_reveal"
            target_str = f"virtual_identity ({self.virtual_combo})" if "virtual" in str(self.delayed_reveal_target) else self.delayed_reveal_target
            id_condition = f"delayed_reveal (anonymous -> {target_str} at stage {self.delayed_reveal_stage})"
        elif self.identity_mode == "virtual_persona":
            disclosure_type = "complete_reveal"
            id_condition = f"virtual_identity ({self.virtual_combo})"
        else:
            disclosure_type = "complete_reveal"
            id_condition = f"real_identity ({self.identity_mode})"

        formatted_roster = []
        for a in self.agents_roster:
            formatted_roster.append({
                "agent_id": a.get("agent_id"),
                "model_name": a.get("model_name"),
                "family": a.get("family"),
                "size": a.get("size_str") or a.get("parameter_size") or f"{a.get('size_billions', '')}B",
                "persona": a.get("persona") if (self.identity_mode == "virtual_persona" or "virtual" in str(self.delayed_reveal_target)) else None
            })

        # In-memory document buffer
        self.metadata = {
            "pid": self.problem_id,
            "dataset": problem_item.get("dataset", "custom"),
            "category": problem_item.get("category", "ethics"),
            "identity_condition": id_condition,
            "disclosure_type": disclosure_type,
            "total_debate_rounds": self.total_debate_rounds,
            "delayed_reveal_stage": self.delayed_reveal_stage if self.delayed_reveal else None,
            "timestamp": datetime.now().isoformat(),
            "agent_roster": formatted_roster
        }
        # A crashed/failed job (agent outage, OOM, etc.) already has every stage
        # up to the crash point durably persisted on disk - record_stage_detail()
        # flushes stages_detail to the JSON file the instant each stage finishes,
        # not just at the end. Discarding that and re-running all 15 stages from
        # scratch (the old behavior) is real, avoidable waste: confirmed on a
        # real run where a ~50min agent-server outage forced a full redo of 1552
        # jobs, most of which had already completed several stages. If the
        # caller (SimulationRunner) found valid prior progress for this exact
        # roster, it passes it in here so we continue instead of overwriting it.
        self.group_chat_transcript: List[Dict[str, Any]] = list(resume_transcript) if resume_transcript else []
        self.stages_detail: List[Dict[str, Any]] = list(resume_stages_detail) if resume_stages_detail else []
        self.summary_analytics: Dict[str, Any] = {}
        self._resuming = bool(resume_stages_detail)

        self._init_files()

    def _init_files(self):
        """Initializes log files on disk - fresh, or continuing a resumed session."""
        if self._resuming:
            resumed_event = {
                "event_type": "session_resumed",
                "timestamp": datetime.now().isoformat(),
                "problem_id": self.problem_id,
                "stages_already_completed": [s.get("stage_name") for s in self.stages_detail],
            }
            self._append_jsonl(resumed_event)
            self._append_transcript_md(
                f"> **[Moderator Announcement - resume]**  \n"
                f"> Session resumed after an interruption. Stages already completed: "
                f"{', '.join(s.get('stage_name', '?') for s in self.stages_detail)}."
            )
            self._sync_full_json()
            return

        init_event = {
            "event_type": "session_init",
            "timestamp": datetime.now().isoformat(),
            "problem_id": self.problem_id,
            "question": self.problem_item.get("question", ""),
            "options": {
                "A": self.problem_item.get("option_a", ""),
                "B": self.problem_item.get("option_b", "")
            },
            "identity_mode": self.identity_mode
        }
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(init_event) + "\n")

        with open(self.transcript_md_path, "w", encoding="utf-8") as f:
            f.write(f"# Deliberation Transcript: Problem [{self.problem_id}]\n\n")
            f.write(f"- **Dataset:** {self.metadata['dataset']}\n")
            f.write(f"- **Identity Mode:** `{self.identity_mode}`\n")
            f.write(f"- **Timestamp:** {self.metadata['timestamp']}\n\n")
            f.write(f"### Problem Statement\n")
            f.write(f"> {self.problem_item.get('question', '')}\n\n")
            f.write(f"- **Option A:** {self.problem_item.get('option_a', '')}\n")
            f.write(f"- **Option B:** {self.problem_item.get('option_b', '')}\n\n")
            if self.problem_item.get("context"):
                f.write(f"**Context:** {self.problem_item['context']}\n\n")
            f.write(f"---\n\n## Group Chat Dialogue\n\n")

        self._sync_full_json()

    def _append_jsonl(self, event: Dict[str, Any]):
        """Live appends an event to the JSONL log file and flushes."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()

    def _append_transcript_md(self, text: str):
        """Live appends dialogue text to the readable Markdown transcript."""
        with open(self.transcript_md_path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
            f.flush()

    def _sync_full_json(self):
        """
        Live synchronizes the master problem JSON file. Writes to a temp file
        first, then atomically renames it over the real path - confirmed live
        that writing directly to json_path can leave a corrupted, half-written
        file if the process is killed mid-write (e.g. a sweep restart lands
        exactly during this call). os.replace() is atomic on POSIX, so readers
        only ever see either the old complete file or the new complete file,
        never a partial one, regardless of when a kill signal arrives.
        """
        doc = {
            "problem_statement": self.problem_item,
            "simulation_metadata": self.metadata,
            "group_chat_transcript": self.group_chat_transcript,
            "stages_detail": self.stages_detail,
            "summary_analytics": self.summary_analytics
        }
        tmp_path = f"{self.json_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp_path, self.json_path)

    def _get_speaker_label(self, agent_id: str, display_id: str) -> str:
        """Returns clean markdown label without redundant duplicate IDs."""
        if not display_id or display_id == agent_id:
            return f"**{agent_id}**"
        if display_id.startswith(agent_id):
            return f"**{display_id}**"
        return f"**{agent_id}** (`{display_id}`)"

    def log_system_announcement(self, message: str, stage_name: str = "system"):
        """Logs a moderator / system announcement."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "event_type": "system_announcement",
            "sender": "Moderator",
            "content": message
        }
        self.group_chat_transcript.append(event)
        self._append_jsonl(event)
        self._append_transcript_md(f"> **[Moderator Announcement - {stage_name}]**  \n> {message}")
        self._sync_full_json()

    def log_vote(
        self,
        stage_name: str,
        agent_id: str,
        display_id: str,
        vote: str,
        confidence: float,
        previous_vote: Optional[str] = None,
        confidence_delta: Optional[float] = None,
        flipped: Optional[bool] = None,
        rationale: Optional[str] = None
    ):
        """
        Logs a vote/revote event for any stage (baseline, bandwagon revote,
        reasoned-conformity revote, or a debate-round revote).
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "event_type": "vote",
            "sender_id": agent_id,
            "sender_display": display_id,
            "previous_vote": previous_vote,
            "vote": vote,
            "confidence": confidence,
            "confidence_delta": confidence_delta,
            "flipped": flipped
        }
        if rationale:
            event["content"] = rationale

        self.group_chat_transcript.append(event)
        self._append_jsonl(event)

        speaker_lbl = self._get_speaker_label(agent_id, display_id)
        flip_badge = ""
        if flipped is True:
            flip_badge = " 🚨 **[CHANGED VOTE]**"
        elif flipped is False:
            flip_badge = " ✅ **[KEPT VOTE]**"
        delta_str = f", Δ: {confidence_delta:+.2f}" if confidence_delta is not None else ""
        md_entry = f"🗳️ {speaker_lbl} - [{stage_name}]{flip_badge}: **Option {vote}** (Conf: {confidence:.2f}{delta_str})"
        if rationale:
            md_entry += f"\n> *Rationale:* {rationale}"
        self._append_transcript_md(md_entry)
        self._sync_full_json()

    def log_rationale(
        self,
        stage_name: str,
        agent_id: str,
        display_id: str,
        vote: str,
        rationale: str
    ):
        """Logs a standalone rationale-collection event (no revote attached)."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "event_type": "rationale",
            "sender_id": agent_id,
            "sender_display": display_id,
            "vote": vote,
            "content": rationale
        }
        self.group_chat_transcript.append(event)
        self._append_jsonl(event)

        speaker_lbl = self._get_speaker_label(agent_id, display_id)
        self._append_transcript_md(
            f"💬 {speaker_lbl} - [{stage_name}] rationale for **Option {vote}**:\n> \"{rationale}\""
        )
        self._sync_full_json()

    def log_statement(
        self,
        stage_name: str,
        statement_idx: int,
        speaker_id: str,
        speaker_display: str,
        target_id: str,
        target_display: str,
        statement: str
    ):
        """Logs a targeted debate statement ('To [Agent]: ...')."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "event_type": "directional_statement",
            "statement_idx": statement_idx,
            "speaker_id": speaker_id,
            "speaker_display": speaker_display,
            "target_id": target_id,
            "target_display": target_display,
            "content": statement
        }
        self.group_chat_transcript.append(event)
        self._append_jsonl(event)

        speaker_lbl = self._get_speaker_label(speaker_id, speaker_display)
        self._append_transcript_md(
            f"🗣️ {speaker_lbl} -> {target_display} [{stage_name} #{statement_idx}]:\n"
            f"> \"{statement}\""
        )
        self._sync_full_json()

    def log_reply(
        self,
        stage_name: str,
        message_idx: int,
        replier_id: str,
        replier_display: str,
        sender_id: str,
        sender_display: str,
        decision: str,
        text: str
    ):
        """Logs a reply to, or a decline of, a debate message addressed to the replier."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage_name,
            "event_type": "reply" if decision == "reply" else "decline",
            "message_idx": message_idx,
            "speaker_id": replier_id,
            "speaker_display": replier_display,
            "target_id": sender_id,
            "target_display": sender_display,
            "content": text
        }
        self.group_chat_transcript.append(event)
        self._append_jsonl(event)

        replier_lbl = self._get_speaker_label(replier_id, replier_display)
        if decision == "reply":
            self._append_transcript_md(
                f"↩️ {replier_lbl} REPLIED to {sender_display} [{stage_name} msg #{message_idx}]:\n> \"{text}\""
            )
        else:
            self._append_transcript_md(
                f"🚫 {replier_lbl} DECLINED to reply to {sender_display} [{stage_name} msg #{message_idx}] - reason:\n> \"{text}\""
            )
        self._sync_full_json()

    def record_stage_detail(self, stage_data: Dict[str, Any]):
        """Records the structured summary object for a completed stage."""
        self.stages_detail.append(stage_data)
        self._sync_full_json()

    def finalize_simulation(self, summary_analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Finalizes the simulation file with complete summary analytics and closing banner."""
        self.summary_analytics = summary_analytics
        self._sync_full_json()

        self._append_transcript_md(
            f"\n---\n\n## Deliberation Outcome Summary\n\n"
            f"- **Initial Tally (baseline):** Option A: {summary_analytics.get('initial_tally', {}).get('A', 0)} | Option B: {summary_analytics.get('initial_tally', {}).get('B', 0)}\n"
            f"- **Final Tally:** Option A: {summary_analytics.get('final_tally', {}).get('A', 0)} | Option B: {summary_analytics.get('final_tally', {}).get('B', 0)}\n"
            f"- **Consensus Achieved:** `{summary_analytics.get('consensus_achieved', False)}`\n"
            f"- **Total Vote Flips (all stages):** `{summary_analytics.get('total_flips', 0)}`\n"
        )

        final_event = {
            "event_type": "session_finalized",
            "timestamp": datetime.now().isoformat(),
            "problem_id": self.problem_id,
            "initial_tally": summary_analytics.get("initial_tally"),
            "final_tally": summary_analytics.get("final_tally"),
            "total_flips": summary_analytics.get("total_flips")
        }
        self._append_jsonl(final_event)

        logger.info(f"[LOGGER] Discussion automatically finalized and saved to:")
        logger.info(f"-> JSON:       {self.json_path}")
        logger.info(f"-> JSONL:      {self.jsonl_path}")
        logger.info(f"-> Transcript: {self.transcript_md_path}")

        return {
            "json_path": self.json_path,
            "jsonl_path": self.jsonl_path,
            "transcript_md_path": self.transcript_md_path
        }


# ==============================================================================
# STANDALONE CLI UTILITIES (Inspect, Export, Convert)
# ==============================================================================

def print_transcript_from_json(json_path: str):
    """Prints a clean human-readable conversation transcript directly to terminal."""
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p = data.get("problem_statement", {})
    print("\n" + "=" * 80)
    print(f"DELIBERATION TRANSCRIPT: [{p.get('id', 'Unknown')}]")
    print("=" * 80)
    print(f"Question: {p.get('question', '')}\n")
    print(f"Option A: {p.get('option_a', '')}")
    print(f"Option B: {p.get('option_b', '')}\n" + "-" * 80)

    for item in data.get("group_chat_transcript", []):
        evt = item.get("event_type")
        stage = item.get("stage", "")
        sender = item.get("sender_display") or item.get("sender_id") or item.get("speaker_display") or item.get("speaker_id") or item.get("sender", "System")
        content = item.get("content", "")
        conf_val = item.get("confidence", 0.0) or 0.0
        delta_val = item.get("confidence_delta", 0.0) or 0.0

        if evt == "system_announcement":
            print(f"\n📢 [{stage} - Announcement]: {content}")
        elif evt == "vote":
            flip_tag = ""
            if item.get("flipped") is True:
                flip_tag = " [CHANGED]"
            elif item.get("flipped") is False:
                flip_tag = " [KEPT]"
            print(f"\n🗳️  [{stage}{flip_tag}] {sender} voted Option {item.get('vote')} (Conf: {conf_val:.2f}, Δ: {delta_val:+.2f}):")
            if content:
                print(f"    \"{content}\"")
        elif evt == "rationale":
            print(f"\n💬 [{stage}] {sender} rationale for Option {item.get('vote')}:")
            print(f"    \"{content}\"")
        elif evt == "directional_statement":
            print(f"\n🗣️  [{stage} #{item.get('statement_idx')}] {sender} -> {item.get('target_display')}:")
            print(f"    \"{content}\"")

    summary = data.get("summary_analytics", {})
    print("\n" + "=" * 80)
    print(f"OUTCOME: Initial Tally {summary.get('initial_tally')} -> Final Tally {summary.get('final_tally')} | Total Flips: {summary.get('total_flips')}")
    print("=" * 80 + "\n")


def export_votes_to_csv(json_path: str, output_csv_path: Optional[str] = None):
    """Exports agent vote trajectories from a problem JSON file to a CSV spreadsheet."""
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not output_csv_path:
        output_csv_path = json_path.replace(".json", "_votes.csv")

    trajectories = data.get("summary_analytics", {}).get("agent_trajectories", [])
    if not trajectories:
        print("No trajectory records found in file.")
        return

    stage_names = trajectories[0].get("stage_names", [])
    num_stages = len(stage_names)
    headers = ["agent_id", "family", "size_str", "size_billions"]
    for stage_name in stage_names:
        headers.extend([f"{stage_name}_vote", f"{stage_name}_confidence"])
    headers.extend(["flip_count", "final_vote", "final_rationale"])

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in trajectories:
            row = [t.get("agent_id"), t.get("family"), t.get("size_str"), t.get("size_b")]
            votes = t.get("vote_history", [])
            confs = t.get("confidence_history", [])
            for r in range(num_stages):
                row.append(votes[r] if r < len(votes) else "")
                row.append(confs[r] if r < len(confs) else "")
            row.extend([t.get("flip_count"), t.get("final_vote"), t.get("final_rationale", "")])
            writer.writerow(row)

    print(f"[EXPORT] CSV vote trajectory exported to: {output_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deliberation Conversation Log Viewer & Exporter")
    parser.add_argument("--view", type=str, help="View human-readable transcript from a JSON file")
    parser.add_argument("--export-csv", type=str, help="Export vote trajectories from JSON to CSV")
    args = parser.parse_args()

    if args.view:
        print_transcript_from_json(args.view)
    elif args.export_csv:
        export_votes_to_csv(args.export_csv)
    else:
        parser.print_help()
