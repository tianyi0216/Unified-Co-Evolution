"""UCoE inside a coding-agent harness on SpreadsheetBench-2.

Implements the agent-harness study of the paper (Results, "UCoE transfers to a
Codex agent harness"; Appendix "Agent-harness study on SpreadsheetBench-2").
The lifecycle is the one of Algorithm 1; the harness state is the agent's
working context (tools, task skill file and a bounded experience block).

Modules
  core           deterministic decision rules (candidate validation, aggregation,
                 maximin selection, acceptance, experience-block update, evolution)
  evidence       recalculated workbook evidence bound to a candidate by hash
  contrasts      response-contrast pairs from the actor's own calibration drafts
  codex_backend  Codex app-server transport with role-explicit provenance checks
  runner         manifest, calibration banks, initial selection, checkpoints,
                 per-task lifecycle, sealing, status and audit
  evaluate       LibreOffice recalculation and the official checker (evaluation only)
  recalc_helper  the recalculation helper copied into every agent-visible directory
  run            command-line entry point
"""

from __future__ import annotations


class HardStop(RuntimeError):
    """Fail-closed stop: the current step is not sealed and the run resumes from its journal.

    `code` is one of the hard-stop codes recorded in the manifest (recalc_error, invalid_xlsx,
    provenance_mismatch, missing_effort, timeout, malformed_output, evidence_mismatch,
    model_fallback, transport_error).
    """

    def __init__(self, code, msg):
        super().__init__("%s: %s" % (code, msg))
        self.code = code
