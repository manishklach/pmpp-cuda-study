from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


REQUIRED_README_SECTIONS = [
    "## Goal",
    "## Build",
    "## Run",
]

ALLOWED_STATUS = {
    "fully_mature",
    "verified",
    "compiles",
    "scaffolded",
    "notes_only",
    "in_progress",
}


def validate_example(folder: Path, require_meta: bool) -> list[str]:
  errors: list[str] = []
  required = ["README.md", "main.cu"]
  if require_meta:
    required.append("meta.yaml")
  for name in required:
    if not (folder / name).exists():
      errors.append(f"{folder.name}: missing {name}")

  readme = folder / "README.md"
  if readme.exists():
    text = readme.read_text(encoding="utf-8", errors="ignore")
    for section in REQUIRED_README_SECTIONS:
      if section not in text:
        errors.append(f"{folder.name}: README missing section {section}")

  meta = folder / "meta.yaml"
  if meta.exists():
    meta_text = meta.read_text(encoding="utf-8", errors="ignore")
    for field in ("number:", "slug:", "title:", "status:", "maturity_level:"):
      if field not in meta_text:
        errors.append(f"{folder.name}: meta.yaml missing field {field[:-1]}")

    status_match = re.search(r"^status:\s*([A-Za-z_]+)\s*$", meta_text, re.MULTILINE)
    if status_match and status_match.group(1) not in ALLOWED_STATUS:
      errors.append(f"{folder.name}: invalid status {status_match.group(1)}")
  elif require_meta:
    errors.append(f"{folder.name}: missing meta.yaml")
  return errors


def main() -> int:
  parser = argparse.ArgumentParser(description="Validate repo example structure.")
  parser.add_argument("--require-meta", action="store_true", help="Require meta.yaml for every example")
  args = parser.parse_args()

  errors: list[str] = []
  for folder in sorted(path for path in EXAMPLES.iterdir() if path.is_dir()):
    errors.extend(validate_example(folder, args.require_meta))

  if errors:
    print("Repo validation failed:")
    for error in errors:
      print(f"- {error}")
    return 1

  print("Repo validation passed.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
