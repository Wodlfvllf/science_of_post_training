"""Model-free regression tests for the data-only download recovery boundary."""

import io
import stat
import warnings
import zipfile
from pathlib import Path

import pytest
from analyze_downloaded_backups import extract_selected, safe_members


@pytest.mark.parametrize("name", [
    "../escape", "/absolute", "a/../escape", "a/./file", "a//file",
    "C:/drive", "C:drive", "a\\file", "a/../../escape",
])
def test_unsafe_paths_are_rejected(name):
    """ZIP names must be canonical relative POSIX paths, not merely normalized."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, "unsafe")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError):
        safe_members(archive)


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFSOCK])
def test_special_files_are_rejected(kind):
    """Unix special-file metadata must never become an extracted artifact."""
    data = io.BytesIO()
    item = zipfile.ZipInfo("data/posttrain/file.json")
    item.create_system = 3
    item.external_attr = (kind | 0o644) << 16
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(item, "target")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError):
        safe_members(archive)


def test_duplicate_names_are_rejected():
    """Duplicate lookup keys cannot silently replace a validated ZIP member."""
    data = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(data, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        archive.writestr("data/file.json", "one")
        archive.writestr("data/file.json", "two")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError):
        safe_members(archive)


def test_only_allowlisted_files_and_no_overwrite(tmp_path):
    """Do not extract archived source, and never reuse an existing output root."""
    data = io.BytesIO()
    name = "data/posttrain/example.json"
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, "{}")
        archive.writestr("src/interp/evil.py", "raise RuntimeError('not executed')")
    destination = tmp_path / "recovered"
    with zipfile.ZipFile(data) as archive:
        extract_selected(archive, {name}, destination)
        assert (destination / name).read_text() == "{}"
        assert not (destination / "src").exists()
        with pytest.raises(FileExistsError):
            extract_selected(archive, {name}, destination)
    assert (destination / name).read_text() == "{}"


def test_symlink_ancestor_is_rejected(tmp_path):
    """Even an otherwise safe member cannot traverse a linked output ancestor."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("data.json", "{}")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError):
        extract_selected(archive, {"data.json"}, link / "recovered")
    assert not (outside / "recovered/data.json").exists()


def test_recovered_outputs_have_no_code():
    """Check this analysis's actual extraction against the data-only boundary."""
    recovered = Path(__file__).resolve().parents[1] / "runs/download-analysis-20261026/recovered"
    if not recovered.exists():
        pytest.skip("Downloaded-data analysis is not present in this checkout")
    files = [path for path in recovered.rglob("*") if path.is_file()]
    assert len(files) == 2193
    assert all(path.suffix in (".json", ".jsonl", ".csv", ".md") for path in files)
    assert not any(path.is_symlink() for path in recovered.rglob("*"))