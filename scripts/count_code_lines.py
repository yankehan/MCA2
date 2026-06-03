#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_EXTENSIONS = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".r",
    ".m",
}

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    "dist",
    "build",
}

COMMENT_PREFIXES = {
    ".py": ("#",),
    ".sh": ("#",),
    ".bash": ("#",),
    ".zsh": ("#",),
    ".fish": ("#",),
    ".ps1": ("#",),
    ".r": ("#",),
    ".js": ("//",),
    ".jsx": ("//",),
    ".ts": ("//",),
    ".tsx": ("//",),
    ".java": ("//",),
    ".c": ("//",),
    ".cc": ("//",),
    ".cpp": ("//",),
    ".h": ("//",),
    ".hpp": ("//",),
    ".rs": ("//",),
    ".go": ("//",),
    ".kt": ("//",),
    ".kts": ("//",),
    ".scala": ("//",),
    ".swift": ("//",),
    ".php": ("//", "#"),
}


@dataclass
class LineStats:
    files: int = 0
    total: int = 0
    blank: int = 0
    comment: int = 0
    code: int = 0

    def add(self, other: "LineStats") -> None:
        self.files += other.files
        self.total += other.total
        self.blank += other.blank
        self.comment += other.comment
        self.code += other.code


def parse_extensions(values: list[str]) -> set[str]:
    extensions: set[str] = set()
    for value in values:
        for part in value.split(","):
            ext = part.strip()
            if not ext:
                continue
            extensions.add(ext if ext.startswith(".") else f".{ext}")
    return extensions


def is_comment_line(path: Path, stripped_line: str) -> bool:
    for prefix in COMMENT_PREFIXES.get(path.suffix.lower(), ()):
        if stripped_line.startswith(prefix):
            return True
    return False


def count_file(path: Path) -> LineStats:
    stats = LineStats(files=1)
    in_python_multiline_string = False

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stats.total += 1
            stripped = line.strip()

            if not stripped:
                stats.blank += 1
                continue

            if path.suffix.lower() == ".py":
                triple_count = stripped.count('"""') + stripped.count("'''")
                if in_python_multiline_string:
                    stats.comment += 1
                    if triple_count % 2 == 1:
                        in_python_multiline_string = False
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    stats.comment += 1
                    if triple_count % 2 == 1:
                        in_python_multiline_string = True
                    continue

            if is_comment_line(path, stripped):
                stats.comment += 1
            else:
                stats.code += 1

    return stats


def iter_code_files(
    root: Path,
    extensions: set[str],
    excluded_dirs: set[str],
    max_bytes: int,
) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.relative_to(root).parts[:-1]):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if path.stat().st_size > max_bytes:
            continue
        files.append(path)
    return sorted(files)


def print_table(rows: list[tuple[str, LineStats]]) -> None:
    print(f"{'Type':<12} {'Files':>8} {'Total':>10} {'Blank':>10} {'Comment':>10} {'Code':>10}")
    print("-" * 64)
    for label, stats in rows:
        print(
            f"{label:<12} {stats.files:>8} {stats.total:>10} "
            f"{stats.blank:>10} {stats.comment:>10} {stats.code:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Count source-code lines in a repository.")
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository path to scan. Defaults to the MCA2 repository root.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(DEFAULT_EXTENSIONS)),
        help="Comma-separated file extensions to count, for example: py,sh,js.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory name to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1_000_000,
        help="Skip files larger than this many bytes. Defaults to 1,000,000.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()

    root = args.root.resolve()
    extensions = parse_extensions([args.extensions])
    excluded_dirs = DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)
    by_extension: dict[str, LineStats] = defaultdict(LineStats)
    total = LineStats()

    for path in iter_code_files(root, extensions, excluded_dirs, args.max_bytes):
        stats = count_file(path)
        by_extension[path.suffix.lower()].add(stats)
        total.add(stats)

    rows = sorted(by_extension.items())
    rows.append(("TOTAL", total))

    if args.json:
        payload = {label: asdict(stats) for label, stats in rows}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Scanned root: {root}")
        print_table(rows)


if __name__ == "__main__":
    main()
