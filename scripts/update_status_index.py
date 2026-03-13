from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def parse_field(text: str, field: str) -> str:
  match = re.search(rf"^{field}:\s*([A-Za-z0-9_-]+)\s*$", text, re.MULTILINE)
  return match.group(1) if match else ""


def main() -> int:
  counts: dict[str, int] = {}
  for meta in EXAMPLES.glob("*/meta.yaml"):
    text = meta.read_text(encoding="utf-8", errors="ignore")
    status = parse_field(text, "status") or "unknown"
    counts[status] = counts.get(status, 0) + 1

  if not counts:
    print("No meta.yaml files found.")
    return 1

  print("Per-status counts:")
  for status in sorted(counts):
    print(f"- {status}: {counts[status]}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
