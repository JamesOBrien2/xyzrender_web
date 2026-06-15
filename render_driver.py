"""Headless orientation driver for xyzrender.

xyzrender can only apply a user-chosen orientation to multi-frame outputs
(``--gif-ts`` / ``--gif-trj``) through its interactive viewer (``-I``); the
``--ref`` flag is ignored on those code paths. This driver reproduces the
``-I`` behaviour non-interactively by monkeypatching ``xyzrender.api.orient``
to apply a fixed rotation matrix (captured from the web 3D viewer) instead of
opening a GUI, then delegating to xyzrender's normal CLI so every other flag
is handled exactly as usual.

Usage::

    python render_driver.py <rotation.txt> <xyzrender args...> -I ...

``rotation.txt`` holds a 3x3 rotation matrix (whitespace-separated, as written
by ``numpy.savetxt``). Everything after it is passed straight to the xyzrender
CLI; the caller is responsible for including ``-I`` and the input/output args.
"""
import sys

import numpy as np

import xyzrender.api as api


def _install_orient(R: np.ndarray) -> None:
    """Replace xyzrender's interactive ``orient`` with a fixed rotation."""

    def fake_orient(mol, viewer="vmol", also=None):
        def rot(m):
            for nid in m.graph.nodes():
                p = np.array(m.graph.nodes[nid]["position"], dtype=float)
                m.graph.nodes[nid]["position"] = tuple((R @ p).tolist())
            m.oriented = True

        rot(mol)
        for extra in (also or []):
            rot(extra)

    api.orient = fake_orient


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: render_driver.py <rotation.txt> <xyzrender args...>\n")
        sys.exit(2)

    R = np.loadtxt(sys.argv[1]).reshape(3, 3)
    _install_orient(R)

    from xyzrender.cli import main as cli_main

    sys.argv = ["xyzrender", *sys.argv[2:]]
    cli_main()


if __name__ == "__main__":
    main()
