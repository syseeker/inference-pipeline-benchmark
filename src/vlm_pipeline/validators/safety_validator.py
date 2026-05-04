"""Safety / command validator.

The validator is the **last gate before execution**. Anything dangerous
or off-grammar must die here, not at the executor.

v0: structural checks only — sequence length cap, per-command-type arg
schema, banned-key sweep. v1 will add a small policy LLM (or a rule DSL)
and per-session rate limits.
"""

from __future__ import annotations

from vlm_pipeline.schemas import ActionCommand, ActionSequence, ActionType, ValidationReport

_MAX_COMMANDS = 16

# Per-command argument expectations. Extend as ActionType grows.
_ARG_SCHEMA: dict[ActionType, set[str]] = {
    ActionType.NOOP: set(),
    ActionType.MOVE: {"dx", "dy"},
    ActionType.CLICK: {"button"},
    ActionType.KEYPRESS: {"key"},
    ActionType.WAIT: {"ms"},
    ActionType.SAY: {"text"},
}

# Argument values we never want to see in user-controlled keypresses.
_BANNED_KEYS = {"meta+l", "ctrl+alt+del", "win+l"}


class SafetyValidator:
    def validate(self, seq: ActionSequence) -> ValidationReport:
        notes: list[str] = []
        rejected: list[int] = []

        if len(seq.commands) > _MAX_COMMANDS:
            notes.append(f"sequence too long: {len(seq.commands)} > {_MAX_COMMANDS}")

        for idx, cmd in enumerate(seq.commands):
            ok, why = self._check_one(cmd)
            if not ok:
                rejected.append(idx)
                notes.append(f"#{idx}: {why}")

        return ValidationReport(
            schema_valid=True,  # arrived here, so pydantic already accepted the shape
            safe=not rejected and len(seq.commands) <= _MAX_COMMANDS,
            rejected_command_indices=rejected,
            notes=notes,
        )

    def _check_one(self, cmd: ActionCommand) -> tuple[bool, str]:
        expected = _ARG_SCHEMA.get(cmd.type)
        if expected is None:
            return False, f"unknown command type: {cmd.type}"
        missing = expected - cmd.args.keys()
        if missing:
            return False, f"missing args: {sorted(missing)}"
        if cmd.type is ActionType.KEYPRESS:
            if str(cmd.args.get("key", "")).lower() in _BANNED_KEYS:
                return False, f"banned key: {cmd.args.get('key')}"
        return True, ""
