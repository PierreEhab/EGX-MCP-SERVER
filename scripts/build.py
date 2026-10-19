"""Build distributions with uv and omit its generated pyproject backup from the sdist."""

from __future__ import annotations

import os
import re
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath


def strip_generated_pyproject_backup(sdist_path: Path) -> None:
    """Remove uv_build's generated ``pyproject.toml.orig`` from an sdist."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sdist_path.name}.",
        suffix=".tmp",
        dir=sdist_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with tarfile.open(sdist_path, "r:gz") as source:
            with tarfile.open(temporary_path, "w:gz") as destination:
                for member in source:
                    if PurePosixPath(member.name).name == "pyproject.toml.orig":
                        continue
                    source_file = source.extractfile(member) if member.isfile() else None
                    destination.addfile(member, source_file)
        temporary_path.replace(sdist_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Run ``uv build`` and clean the generated source distribution."""
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))[
        "project"
    ]
    normalized_name = re.sub(r"[-_.]+", "_", project["name"]).lower()
    sdist_path = (
        project_root
        / "dist"
        / f"{normalized_name}-{project['version']}.tar.gz"
    )

    subprocess.run(["uv", "build"], cwd=project_root, check=True)
    if not sdist_path.is_file():
        raise FileNotFoundError(f"uv build did not create {sdist_path}")
    strip_generated_pyproject_backup(sdist_path)


if __name__ == "__main__":
    main()
