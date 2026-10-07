from __future__ import annotations

import json
from pathlib import Path

from agent import QAAgent

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evaluation" / "questions.json"


def main() -> None:
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    agent = QAAgent()

    hits = 0
    for case in cases:
        result = agent.process_query(case["question"])
        context = " ".join(result.get("context") or []).lower()
        hit = case["expected_source"].lower() in context
        hits += int(hit)
        print(f'{case["id"]:02d} | {"HIT" if hit else "MISS"} | {case["question"]}')

    rate = hits / len(cases) * 100
    print(f"\nRetrieval hit rate: {hits}/{len(cases)} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
