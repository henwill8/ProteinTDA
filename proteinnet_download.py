"""
Download human-readable ProteinNet archives.

URLs follow the official ProteinNet release:
https://github.com/aqlaboratory/proteinnet
"""

import argparse
import ssl
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

AVAILABLE_CASP = (7, 8, 9, 10, 11, 12)

_BASE_URL = (
    "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable"
)
_DEFAULT_DATA_DIR = Path("data") / "proteinnet"
_CHUNK_SIZE = 1 << 20  # 1 MiB


def download_url(casp: int) -> str:
    """Return the download URL for a CASP human-readable tarball."""
    _validate_casp(casp)
    return f"{_BASE_URL}/casp{casp}.tar.gz"


def download_proteinnet_human_readable(
    casp: int,
    output_dir: Path | str | None = None,
    *,
    extract: bool = True,
    force: bool = False,
    insecure: bool = False,
) -> Path:
    """
    Download (and optionally extract) a ProteinNet human-readable CASP archive.

    Parameters
    ----------
    casp
        CASP edition number (7–12).
    output_dir
        Directory for ``casp{N}.tar.gz`` and extracted ``casp{N}/`` folder.
        Defaults to ``data/proteinnet``.
    extract
        If True, extract the tarball after download.
    force
        If True, re-download even when the archive already exists.
    insecure
        If True, disables TLS certificate validation.

    Returns
    -------
    Path
        Path to the extracted ``casp{N}`` directory if ``extract`` is True,
        otherwise path to the ``.tar.gz`` file.
    """
    _validate_casp(casp)
    root = Path(output_dir) if output_dir is not None else _DEFAULT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)

    archive_path = root / f"casp{casp}.tar.gz"
    extract_dir = root / f"casp{casp}"

    if not archive_path.exists() or force:
        _download_file(download_url(casp), archive_path, insecure=insecure)
    elif extract and extract_dir.exists():
        return extract_dir

    if extract:
        _extract_archive(archive_path, root)
        return extract_dir

    return archive_path


def _validate_casp(casp: int) -> None:
    if casp not in AVAILABLE_CASP:
        raise ValueError(
            f"CASP must be one of {AVAILABLE_CASP}, got {casp!r}."
        )


def _download_file(url: str, dest: Path, *, insecure: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ProteinTDA/1.0"})
        context = _build_ssl_context(insecure=insecure)
        with urllib.request.urlopen(request, context=context) as response, tmp.open("wb") as out:
            total = response.headers.get("Content-Length")
            total = int(total) if total is not None else None
            downloaded = 0
            started = time.time()
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total is not None:
                    pct = 100.0 * downloaded / total
                    downloaded_gb = downloaded / (1024 * 1024 * 1024)
                    total_gb = total / (1024 * 1024 * 1024)
                    elapsed = max(time.time() - started, 1e-6)
                    speed_mb_s = (downloaded / (1024 * 1024)) / elapsed
                    remaining_gb = max(total_gb - downloaded_gb, 0.0)
                    speed_gb_s = speed_mb_s / 1024.0
                    eta_s = int(remaining_gb / speed_gb_s) if speed_gb_s > 0 else 0
                    eta_mm = eta_s // 60
                    eta_ss = eta_s % 60
                    print(
                        (
                            f"\r  {downloaded_gb:.2f}/{total_gb:.2f} GB ({pct:.1f}%)"
                            f"  {speed_mb_s:.2f} MB/s  ETA {eta_mm:02d}:{eta_ss:02d}"
                        ),
                        end="",
                        flush=True,
                    )
            if total is not None:
                print()
        tmp.replace(dest)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "Download failed due to TLS certificate verification.\n"
                "Try one of:\n"
                "  1) Install certifi and rerun (recommended).\n"
                "  2) Use --insecure as a last resort.\n"
                f"URL: {url}"
            ) from exc
        raise RuntimeError(f"Download failed: {url}") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def _build_ssl_context(*, insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()  # noqa: SLF001

    # Prefer certifi if installed; fall back to system trust store.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ProteinNet human-readable data for a CASP edition.",
    )
    parser.add_argument(
        "--casp",
        type=int,
        required=True,
        choices=AVAILABLE_CASP,
        help=f"CASP version ({', '.join(map(str, AVAILABLE_CASP))}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"Download directory (default: {_DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep the .tar.gz only; do not extract.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the archive already exists.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (last resort).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    path = download_proteinnet_human_readable(
        args.casp,
        args.output_dir,
        extract=not args.no_extract,
        force=args.force,
        insecure=args.insecure,
    )
    print(f"Done: {path.resolve()}")


if __name__ == "__main__":
    main()
