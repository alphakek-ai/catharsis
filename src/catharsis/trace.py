"""Trace writer — append-only JSONL for crash-safe logging of all model outputs."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ResponseLengths:
    """Token counts for different parts of a model response."""

    reasoning_tokens: int
    content_tokens: int
    total_tokens: int


class TraceWriter:
    def __init__(self, output_dir: str = "traces"):
        self.run_id = f"run_{int(time.time())}"
        self.base_dir = Path(output_dir) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self.base_dir / "trace.jsonl"
        self._f = open(self._trace_path, "a")  # noqa: SIM115

    def _write(self, event: str, **data):
        line = {"ts": time.time(), "event": event, **data}
        self._f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def write_meta(self, **kwargs):
        self._write("meta", **kwargs)

    def write_response(
        self,
        generation: int,
        candidate: int,
        prompt_idx: int,
        prompt: str,
        response: str,
        response_lengths: ResponseLengths,
        raw_response: str | None = None,
    ):
        """Write immediately after student model generates a response."""
        data: dict = {
            "gen": generation,
            "cand": candidate,
            "idx": prompt_idx,
            "prompt": prompt,
            "response": response,
            "lengths": asdict(response_lengths),
        }
        if raw_response is not None and raw_response != response:
            data["raw_response"] = raw_response
        self._write("response", **data)

    def write_verdict(
        self,
        generation: int,
        candidate: int,
        prompt_idx: int,
        is_refusal: bool | None,
        judge_lengths: ResponseLengths | None,
        judge_reasoning: str = "",
        judge_error: str | None = None,
    ):
        """Write immediately after judge returns a verdict."""
        data: dict = {
            "gen": generation,
            "cand": candidate,
            "idx": prompt_idx,
        }
        if is_refusal is not None:
            data["is_refusal"] = is_refusal
        if judge_lengths is not None:
            data["judge_lengths"] = asdict(judge_lengths)
        if judge_reasoning:
            data["judge_reasoning"] = judge_reasoning
        if judge_error is not None:
            data["judge_error"] = judge_error
        self._write("verdict", **data)

    def write_candidate_summary(
        self,
        generation: int,
        candidate: int,
        refusals: int,
        compliance: int,
        judge_errors: int,
        kl: float,
        score: float,
    ):
        self._write(
            "candidate_summary",
            gen=generation,
            cand=candidate,
            refusals=refusals,
            compliance=compliance,
            judge_errors=judge_errors,
            kl=kl,
            score=score,
        )

    def write_generation_summary(self, generation: int, best_score: float, best_compliance: int, best_kl: float):
        self._write(
            "generation_summary",
            gen=generation,
            best_score=best_score,
            best_compliance=best_compliance,
            best_kl=best_kl,
        )

    def close(self):
        self._f.close()
