from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def status_for(folder: Path) -> str:
  meta = folder / "meta.yaml"
  if not meta.exists():
    return ""
  text = meta.read_text(encoding="utf-8", errors="ignore")
  match = re.search(r"^status:\s*([A-Za-z_]+)\s*$", text, re.MULTILINE)
  return match.group(1) if match else ""


def parse_selection(args: argparse.Namespace) -> list[Path]:
  selected: list[Path] = []
  if args.examples:
    for item in args.examples.split(","):
      item = item.strip()
      if not item:
        continue
      matches = list(EXAMPLES.glob(f"{item}_*")) if item.isdigit() else [EXAMPLES / item]
      for match in matches:
        if match.exists():
          selected.append(match)
  elif args.range:
    start_text, end_text = args.range.split("-")
    start = int(start_text)
    end = int(end_text)
    for number in range(start, end + 1):
      selected.extend(EXAMPLES.glob(f"{number:03d}_*"))
  elif args.all_mature:
    selected = [path for path in EXAMPLES.iterdir() if path.is_dir() and status_for(path) == "fully_mature"]
  else:
    selected = sorted(path for path in EXAMPLES.iterdir() if path.is_dir())
  return sorted(dict.fromkeys(selected))


def build_example(folder: Path) -> bool:
  source = folder / "main.cu"
  output = folder / "example.exe"
  command = [
      "nvcc",
      "-std=c++17",
      "-O2",
      "-I",
      str(ROOT / "include"),
      str(source),
      "-o",
      str(output),
  ]
  result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"[BUILD FAIL] {folder.name}")
    print(result.stdout)
    print(result.stderr)
    return False
  print(f"[BUILD OK] {folder.name}")
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description="Build selected CUDA examples with nvcc.")
  parser.add_argument("--examples", help="Comma-separated example ids or folder names")
  parser.add_argument("--range", help="Example range such as 001-020")
  parser.add_argument("--all-mature", action="store_true", help="Build all examples labeled fully_mature")
  args = parser.parse_args()

  selected = parse_selection(args)
  if not selected:
    print("No examples selected.")
    return 1

  ok = True
  for folder in selected:
    ok = build_example(folder) and ok
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
