import io
import runpy
import tarfile
from pathlib import Path

strip_generated_pyproject_backup = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build.py")
)["strip_generated_pyproject_backup"]


def test_build_cleanup_removes_only_generated_pyproject_backup(tmp_path: Path) -> None:
    sdist_path = tmp_path / "package-1.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as archive:
        for name, contents in (
            ("package-1.0.0/pyproject.toml", b"[project]\n"),
            ("package-1.0.0/pyproject.toml.orig", b"original"),
            ("package-1.0.0/src/package.py", b"package"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))

    strip_generated_pyproject_backup(sdist_path)

    with tarfile.open(sdist_path, "r:gz") as archive:
        names = archive.getnames()
        assert "package-1.0.0/pyproject.toml.orig" not in names
        assert "package-1.0.0/pyproject.toml" in names
        assert "package-1.0.0/src/package.py" in names
