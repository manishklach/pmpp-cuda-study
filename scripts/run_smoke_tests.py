from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def resolve_examples(raw: str) -> list[Path]:
  selected: list[Path] = []
  for item in raw.split(","):
    item = item.strip()
    if not item:
      continue
    matches = list(EXAMPLES.glob(f"{item}_*")) if item.isdigit() else [EXAMPLES / item]
    for match in matches:
      if match.exists():
        selected.append(match)
  return sorted(dict.fromkeys(selected))


def status_for(folder: Path) -> str:
  meta = folder / "meta.yaml"
  if not meta.exists():
    return ""
  text = meta.read_text(encoding="utf-8", errors="ignore")
  match = re.search(r"^status:\s*([A-Za-z_]+)\s*$", text, re.MULTILINE)
  return match.group(1) if match else ""


def run_example(folder: Path) -> bool:
  exe = folder / "example.exe"
  if not exe.exists():
    print(f"[SKIP] {folder.name} missing example.exe; build it first.")
    return False
  result = subprocess.run([str(exe), "--check"], cwd=folder, capture_output=True, text=True)
  output = result.stdout + "\n" + result.stderr
  passed = result.returncode == 0 and "PASS" in output
  print(f"[{'PASS' if passed else 'FAIL'}] {folder.name}")
  if not passed:
    print(output)
  return passed


def main() -> int:
  parser = argparse.ArgumentParser(description="Run smoke tests for selected built examples.")
  parser.add_argument("--examples", help="Comma-separated example ids or folder names")
  parser.add_argument("--all-mature", action="store_true", help="Run all examples labeled fully_mature")
  args = parser.parse_args()

  if args.all_mature:
    selected = [folder for folder in sorted(EXAMPLES.iterdir())
                if folder.is_dir() and status_for(folder) == "fully_mature"]
  elif args.examples:
    selected = resolve_examples(args.examples)
  else:
    selected = []
  if not selected:
    print("No examples selected.")
    return 1

  ok = True
  for folder in selected:
    ok = run_example(folder) and ok
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
