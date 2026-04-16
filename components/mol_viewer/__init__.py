import os
import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend")
_mol_viewer = components.declare_component("mol_viewer", path=_COMPONENT_PATH)


def mol_viewer(xyz_str: str, key: str = None):
    """
    Render an interactive 3Dmol.js viewer inside Streamlit.

    Returns the current view state as a list
    [tx, ty, tz, zoom, qx, qy, qz, qw] on every mouse-up after the
    user drags the molecule, or None until the first interaction.
    The quaternion (indices 4-7) encodes the current rotation and can
    be converted to a rotation matrix to pre-orient coordinates before
    passing them to xyzrender with --no-orient.
    """
    return _mol_viewer(xyz_str=xyz_str, key=key, default=None)
