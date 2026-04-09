"""Trace writer — saves all model outputs and judge verdicts to disk for analysis."""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ResponseLengths:
    """Character lengths of different parts of a model response."""

    reasoning: int
    content: int
    total: int


# Patterns for extracting reasoning traces from different model families
_THINKING_PATTERNS = [
    # Gemma 4: <|channel>thought\n...<channel|>
    re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL),
    # Qwen3/3.5: <think>...</think>
    re.compile(r"<think>.*?</think>", re.DOTALL),
    # Generic: <|think|>...(until end or next tag)
    re.compile(r"<\|think\|>.*?(?=<|$)", re.DOTALL),
]


def measure_response(raw_text: str) -> ResponseLengths:
    """Measure reasoning vs content length in a model response."""
    total = len(raw_text)
    reasoning_len = 0
    content_text = raw_text

    for pattern in _THINKING_PATTERNS:
        matches = pattern.findall(content_text)
        if matches:
            for match in matches:
                reasoning_len += len(match)
            content_text = pattern.sub("", content_text).strip()
            break

    return ResponseLengths(
        reasoning=reasoning_len,
        content=len(content_text),
        total=total,
    )


class TraceWriter:
    def __init__(self, output_dir: str = "traces"):
        self.run_id = f"run_{int(time.time())}"
        self.base_dir = Path(output_dir) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_meta(self, **kwargs):
        with open(self.base_dir / "meta.json", "w") as f:
            json.dump(kwargs, f, indent=2, default=str)

    def write_candidate(
        self,
        generation: int,
        candidate: int,
        prompts: list[str],
        responses: list[str],
        response_lengths: list[ResponseLengths],
        verdicts: list[bool | None],
        judge_lengths: list[ResponseLengths | None],
        judge_errors: list[str | None],
        kl: float,
        score: float,
    ):
        gen_dir = self.base_dir / f"gen_{generation:04d}"
        gen_dir.mkdir(exist_ok=True)

        records = []
        for i in range(len(prompts)):
            record: dict = {
                "prompt": prompts[i],
                "response": responses[i],
                "response_lengths": asdict(response_lengths[i]),
            }
            if i < len(verdicts) and verdicts[i] is not None:
                record["is_refusal"] = verdicts[i]
            jl = judge_lengths[i] if i < len(judge_lengths) else None
            if jl is not None:
                record["judge_lengths"] = asdict(jl)
            if i < len(judge_errors) and judge_errors[i] is not None:
                record["judge_error"] = judge_errors[i]
            records.append(record)

        n_verdicts = [v for v in verdicts if v is not None]
        data = {
            "generation": generation,
            "candidate": candidate,
            "kl": kl,
            "score": score,
            "refusals": sum(1 for v in n_verdicts if v),
            "compliance": sum(1 for v in n_verdicts if not v),
            "judge_errors": sum(1 for e in judge_errors if e is not None),
            "records": records,
        }

        with open(gen_dir / f"cand_{candidate:04d}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def write_generation_summary(self, generation: int, best_score: float, best_compliance: int, best_kl: float):
        gen_dir = self.base_dir / f"gen_{generation:04d}"
        gen_dir.mkdir(exist_ok=True)

        with open(gen_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "generation": generation,
                    "best_score": best_score,
                    "best_compliance": best_compliance,
                    "best_kl": best_kl,
                },
                f,
                indent=2,
            )
