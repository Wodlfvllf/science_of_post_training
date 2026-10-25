# Check the real installed HF adapter's API shape with fake responses, never remote credentials.
"""Offline adapter tests for access/privacy, pinned bytes, safe commits, and uncertain cache misses."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interp.hub import HuggingFaceRemote


def test_adapter_authenticates_supported_api_and_refuses_public_destination(monkeypatch):
    """Read authorization must use the supported signature; public results are never accepted."""
    # Import the actual optional client only in this explicit adapter test.
    import huggingface_hub

    # Track deliberate calls without sending anything to the network.
    calls = []

    class FakeApi:
        """Mirror auth_check's actual read-only signature, which has no write argument."""

        def auth_check(self, repo_id, *, repo_type=None, token=None):
            """Record a supported private-dataset authorization check."""
            # Preserve exact non-secret request identity for assertions.
            calls.append(("auth", repo_id, repo_type))

        def repo_info(self, repo_id, *, repo_type, revision):
            """Return a private fixed revision without authenticating externally."""
            # Verify moving main is resolved only once before immutable reads/commit guards.
            calls.append(("info", repo_id, repo_type, revision))
            # Supply the minimum established client response fields used by this adapter.
            return SimpleNamespace(private=True, sha="a" * 40)

    # Replace the optional client's factory, not the workbench's access-control behavior.
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    # Construction itself must not trigger auth, downloads, or remote creation.
    remote = HuggingFaceRemote("offline/results")
    # Assert the adapter constructor remains a side-effect-free configuration boundary.
    assert calls == []
    # Resolve authenticated read access with write intent, without claiming preflight write proof.
    assert remote.head(write=True) == "a" * 40
    # Preserve the exact supported private-dataset client calls.
    assert calls == [("auth", "offline/results", "dataset"), ("info", "offline/results", "dataset", "main")]
    # Make only visibility change, preserving the same legitimate response structure.
    monkeypatch.setattr(FakeApi, "repo_info", lambda *args, **kwargs: SimpleNamespace(private=False, sha="a" * 40))
    # A public results destination cannot bypass the explicit privacy policy.
    with pytest.raises(ValueError, match="private"):
        # Attempt only the access check, never a real upload.
        remote.head()


def test_adapter_reads_exact_revision_and_distinguishes_cache_failure(monkeypatch, tmp_path):
    """A missing cache after network failure cannot masquerade as missing remote history."""
    # Use the actual established client's missing-entry exception classes.
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    # Create one disposable fake cache artifact, never edit the real HF cache.
    cached = tmp_path / "cached.json"
    # Provide exact bytes for the adapter's pinned read result.
    cached.write_bytes(b"verified-test-bytes")
    # Capture non-secret request arguments for revision/path assertions.
    calls = []

    def download(**kwargs):
        """Return local fixture bytes through the installed download API seam."""
        # Record immutable request settings without issuing a network operation.
        calls.append(kwargs)
        # Reuse the disposable test-local file as a fake cached result.
        return str(cached)

    # Replace only remote/cache transfer with the offline fake.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    # Create a configuration-only remote adapter.
    remote = HuggingFaceRemote("offline/results")
    # Read the requested file at one full pinned git SHA.
    assert remote.read("runs/test/manifest.json", revision="b" * 40) == b"verified-test-bytes"
    # Preserve the exact repo type, filename, and immutable revision supplied to the client.
    assert calls == [{"repo_id": "offline/results", "repo_type": "dataset", "filename": "runs/test/manifest.json", "revision": "b" * 40}]

    def missing(**kwargs):
        """Simulate server-confirmed absence of exactly the pinned remote file."""
        # Only this actual remote absence may return None to checkpoint reconciliation.
        raise EntryNotFoundError("Controlled remote absence.")

    # Confirm genuine remote absence is distinguished from transient/cache uncertainty.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    # Return an absent-file sentinel only for the server-confirmed missing entry.
    assert remote.read("missing.json", revision="b" * 40) is None

    def unavailable(**kwargs):
        """Simulate an unavailable remote plus missing local cache."""
        # This class subclasses EntryNotFoundError but does not prove remote absence.
        raise LocalEntryNotFoundError("Controlled unavailable storage.")

    # Model a network/offline cache miss without accessing any real remote service.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    # Preserve uncertainty instead of overwriting unknown remote history as a fresh run.
    with pytest.raises(LocalEntryNotFoundError):
        # Execute the explicit pinned adapter read through the same installed API seam.
        remote.read("manifest.json", revision="b" * 40)


def test_adapter_commits_synchronously_with_parent_guard(monkeypatch):
    """One actual create_commit call must carry all files, parent identity, and no background future."""
    # Use the actual CommitOperationAdd implementation, replacing only the HTTP API factory.
    import huggingface_hub

    # Capture explicit publication arguments, never tokens or arbitrary process variables.
    calls = []

    class FakeApi:
        """Return one acknowledged revision from a deterministic offline commit call."""

        def create_commit(self, **kwargs):
            """Inspect the publication contract without sending files anywhere."""
            # Preserve one call's non-secret metadata and immutable payload operations.
            calls.append(kwargs)
            # Return the minimum established client acknowledgement shape.
            return SimpleNamespace(oid="c" * 40)

    # Replace the explicit remote service boundary while retaining real operation construction.
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    # Prepare a small deterministic one-commit checkpoint payload.
    files = {"runs/test/metadata.json": b"{}", "runs/test/manifest.json": b"{}"}
    # Await exactly one acknowledged publication under a full optimistic parent SHA.
    assert HuggingFaceRemote("offline/results").commit(files, parent="b" * 40) == "c" * 40
    # Never implement an atomic checkpoint as one commit per artifact.
    assert len(calls) == 1
    # Preserve branch guarding and synchronous acknowledgement in actual client arguments.
    assert calls[0]["parent_commit"] == "b" * 40 and calls[0]["run_as_future"] is False
    # Upload only to the explicitly selected private-dataset destination.
    assert (calls[0]["repo_id"], calls[0]["repo_type"]) == ("offline/results", "dataset")
    # Include exactly the declared metadata/pointer operations in the single commit.
    assert {operation.path_in_repo for operation in calls[0]["operations"]} == set(files)