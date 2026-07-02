#!/usr/bin/env python3
"""Download OpenFold stereo chemical props into the active environment to be used for violation loss."""

import site
import ssl
import urllib.request
from pathlib import Path

URL = (
    "https://git.scicore.unibas.ch/schwede/openstructure/-/raw/"
    "7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt"
)


def main() -> None:
    dest = Path(site.getsitepackages()[0]) / "openfold" / "resources" / "stereo_chemical_props.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(URL, context=ctx) as resp:
        dest.write_bytes(resp.read())

    print(f"Downloaded {dest}")


if __name__ == "__main__":
    main()
