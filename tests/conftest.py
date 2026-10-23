"""Shared pytest fixtures.

``fake_fdmnes`` writes a small stand-in FDMNES executable to a temp
directory, so ``FdmnesRunner`` can be tested against a real subprocess
launch (fdmfile.txt written next to the executable, return codes, timeouts,
fdmnes_error.txt) without needing the real FDMNES binary.
"""

import os
import stat
import sys
from pathlib import Path
from textwrap import dedent

import pytest

FAKE_FDMNES_SOURCE = dedent(
    r'''
    #!/usr/bin/env python3
    """Stand-in FDMNES executable for tests. Controlled by FAKE_FDMNES_MODE:

    success       -> writes _bav.txt, .txt, and (if Convolution present) _conv.txt; exit 0
    bav_warning   -> like success, but bav text contains an "Error"-ish word
    empty_bav     -> writes a zero-byte _bav.txt; exit 0
    missing_conv  -> writes bav/.txt but no _conv.txt even though Convolution was requested
    nan_conv      -> _conv.txt contains a NaN value
    crash         -> writes fdmnes_error.txt next to the executable; exit 1
    hang          -> sleeps well past any reasonable test timeout
    """
    import os
    import sys
    import time
    from pathlib import Path

    mode = os.environ.get("FAKE_FDMNES_MODE", "success")
    exe_dir = Path(__file__).resolve().parent

    if mode == "hang":
        time.sleep(30)
        sys.exit(0)

    fdmfile = exe_dir / "fdmfile.txt"
    lines = [l for l in fdmfile.read_text().splitlines() if l.strip()]
    input_file = Path(lines[1].strip())
    text = input_file.read_text()

    filout = None
    toks = text.splitlines()
    for i, line in enumerate(toks):
        if line.strip() == "Filout":
            filout = Path(toks[i + 1].strip())
            break
    assert filout is not None, "test input file missing Filout keyword"

    if mode == "crash":
        (exe_dir / "fdmnes_error.txt").write_text("Simulated crash: bad input file\n")
        sys.exit(1)

    filout.parent.mkdir(parents=True, exist_ok=True)

    bav_text = "FDMNES bav file\nRadius = 5.000\nNumber of atoms found: 42\n"
    if mode == "bav_warning":
        bav_text += "Error: minor inconsistency noted, continuing anyway\n"
    bav_path = Path(str(filout) + "_bav.txt")
    if mode == "empty_bav":
        bav_path.write_text("")
    else:
        bav_path.write_text(bav_text)

    Path(str(filout) + ".txt").write_text("Energy  Spectrum\n0.0  0.01\n1.0  0.02\n2.0  0.05\n")

    if "Convolution" in text and mode != "missing_conv":
        if mode == "nan_conv":
            conv_body = "Energy  Spectrum\n0.0  0.01\n1.0  nan\n2.0  0.08\n"
        else:
            conv_body = "Energy  Spectrum\n0.0  0.01\n1.0  0.03\n2.0  0.08\n3.0  0.10\n"
        Path(str(filout) + "_conv.txt").write_text(conv_body)

    sys.exit(0)
    '''
).strip("\n")


@pytest.fixture
def fake_fdmnes(tmp_path) -> Path:
    exe_path = tmp_path / "exe" / "fake_fdmnes.py"
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    exe_path.write_text(FAKE_FDMNES_SOURCE, encoding="utf-8")
    exe_path.chmod(exe_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe_path
