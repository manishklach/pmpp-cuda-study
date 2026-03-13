from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
TEMPLATES = ROOT / "templates"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def infer_category(module_name: str) -> str:
    return slugify(module_name).replace("-", "_")


def write_file(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def scaffold_example(args: argparse.Namespace) -> Path:
    folder_slug = args.slug or slugify(args.title)
    folder = EXAMPLES / f"{args.number:03d}_{folder_slug}"

    readme = read_template("example_README.md")
    readme = readme.replace("000 - Example Title", f"{args.number:03d} - {args.title}")
    readme = readme.replace("Module Name", args.module)
    readme = readme.replace("`beginner`", f"`{args.difficulty}`", 1)
    readme = readme.replace("`🟡 scaffolded`", f"`{args.status}`", 1)

    meta = read_template("example_meta.yaml")
    meta = meta.replace("number: 0", f"number: {args.number}")
    meta = meta.replace("slug: example-slug", f"slug: {folder_slug}")
    meta = meta.replace("title: Example Title", f"title: {args.title}")
    meta = meta.replace("module: 00_module-name", f"module: {slugify(args.module)}")
    meta = meta.replace("category: category-name", f"category: {infer_category(args.module)}")
    meta = meta.replace("difficulty: beginner", f"difficulty: {args.difficulty}")
    meta = meta.replace("status: scaffolded", f"status: {args.status}")

    code = read_template("example_main.cu")

    write_file(folder / "README.md", readme, args.force)
    write_file(folder / "meta.yaml", meta, args.force)
    write_file(folder / "main.cu", code, args.force)
    return folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scaffold a numbered CUDA study example using repo templates."
    )
    parser.add_argument("--number", type=int, required=True, help="Numeric example id, such as 101")
    parser.add_argument("--title", required=True, help="Human-readable example title")
    parser.add_argument("--module", required=True, help="Curriculum module name")
    parser.add_argument(
        "--difficulty",
        choices=["beginner", "intermediate", "advanced"],
        required=True,
        help="Difficulty label",
    )
    parser.add_argument(
        "--status",
        default="scaffolded",
        help="Status label, such as 'scaffolded' or 'implemented'",
    )
    parser.add_argument("--slug", help="Optional explicit slug; derived from title if omitted")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files if needed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    folder = scaffold_example(args)
    print(f"Created {folder}")


if __name__ == "__main__":
    main()
