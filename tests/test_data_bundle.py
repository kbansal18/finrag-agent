import io
import tarfile

import pytest

from finrag.data_bundle import BUNDLE_MEMBERS, install, is_ready, package


def make_data(root):
    for m in BUNDLE_MEMBERS:
        (root / m).parent.mkdir(parents=True, exist_ok=True)
        (root / m).write_text(f"content of {m}")


def test_package_and_install_roundtrip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_data(src)
    bundle = tmp_path / "b.tar.gz"
    digest = package(src, bundle)
    install(bundle, dst, digest)
    assert is_ready(dst)
    assert (dst / "financials.db").read_text() == "content of financials.db"


def test_checksum_mismatch_rejected(tmp_path):
    make_data(tmp_path / "src")
    bundle = tmp_path / "b.tar.gz"
    package(tmp_path / "src", bundle)
    with pytest.raises(ValueError, match="checksum"):
        install(bundle, tmp_path / "dst", "0" * 64)


def test_path_traversal_rejected(tmp_path):
    bundle = tmp_path / "evil.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name in [*BUNDLE_MEMBERS, "../escaped.txt"]:
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        install(bundle, tmp_path / "dst", None)
    assert not (tmp_path / "escaped.txt").exists()
