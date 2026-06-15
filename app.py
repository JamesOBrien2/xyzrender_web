import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from components.mol_viewer import mol_viewer


# --------------------------
# XYZ rotation helpers
# --------------------------
def _quaternion_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Convert a unit quaternion (from 3Dmol getView) to a 3×3 rotation matrix."""
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])


def _apply_rotation_to_xyz(xyz_bytes: bytes, R: np.ndarray) -> bytes:
    """Apply a 3×3 rotation matrix to every atom in an XYZ file."""
    lines = xyz_bytes.decode().splitlines()
    n = int(lines[0].strip())
    comment = lines[1]
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append((parts[0], np.array([float(p) for p in parts[1:4]])))
    out = [str(n), comment]
    for elem, coords in atoms:
        c = R @ coords
        out.append(f"{elem}  {c[0]:.6f}  {c[1]:.6f}  {c[2]:.6f}")
    return "\n".join(out).encode()


@st.cache_data(show_spinner=False)
def _extract_xyz_for_viewer(file_bytes: bytes, file_name: str) -> str | None:
    """Extract a single-frame XYZ for the 3D viewer from a non-XYZ upload
    (e.g. a QM ``.out`` file), using xyzrender's own loader so the atom
    ordering matches what xyzrender renders. Returns None on failure."""
    suffix = Path(file_name).suffix or ".out"
    try:
        from xyzrender.api import load
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / ("input" + suffix)
            src.write_bytes(file_bytes)
            mol = load(str(src))
            dst = Path(td) / "preview.xyz"
            mol.to_xyz(dst, title="xyzrender preview")
            return dst.read_text()
    except Exception:
        return None


st.set_page_config(page_title="xyzrender Web Tool",
                   page_icon="🧪", layout="centered")

st.title("xyzrender Web Tool")
st.write(
    "Upload a `.xyz` or `.out` file (or SMILES) and this app will run `xyzrender file [options]` to "
    "generate an **SVG** or **GIF** preview you can download. Use a multi-frame `.out` "
    "(optimization/trajectory or frequency output) with `--gif-trj`/`--gif-ts` to make a GIF."
)

# --------------------------
# File upload & output name
# --------------------------
uploaded = st.file_uploader("Choose a .xyz or .out file", type=["xyz", "out"])
smiles = st.text_input(
    "Input SMILES string (optional)",
    placeholder="Smiles string here..."
)
out_name = st.text_input(
    "Output file name (extension will be adjusted automatically)",
    value="output.svg"
)

# --------------------------
# Options in two columns (no '-go' checkbox per your request)
# --------------------------
st.subheader("xyzrender Options")

col_left, col_right = st.columns(2)

with col_left:
    opt_bo = st.checkbox("--bo")
    opt_no_bo = st.checkbox("--no-bo")
    opt_vdw = st.checkbox("--vdw")
    opt_ts = st.checkbox("--ts")
    opt_hy = st.checkbox("--hy")
    opt_no_hy = st.checkbox("--no-hy")
    opt_nci = st.checkbox("--nci")
    opt_gif_ts = st.checkbox("--gif-ts")
    opt_gif_trj = st.checkbox("--gif-trj")
    opt_gif_rot = st.checkbox("--gif-rot")
    opt_no_orient = st.checkbox(
        "--no-orient",
        help="Disable PCA auto-orientation. Applied automatically when rotation sliders are used.",
    )

with col_right:
    opt_flat = st.checkbox("--config flat")
    opt_paton = st.checkbox("--config paton")
    opt_pmol = st.checkbox("--config pmol")
    opt_skeletal = st.checkbox("--config skeletal")
    opt_bubble = st.checkbox("--config bubble")
    opt_tube = st.checkbox("--config tube")
    opt_btube = st.checkbox("--config btube")
    opt_mtube = st.checkbox("--config mtube")
    opt_wire = st.checkbox("--config wire")
    opt_graph = st.checkbox("--config graph")


# --------------------------
# Conditional GIF controls
# --------------------------
gif_rot_axis = "y"   # xyzrender default
rot_frames = 120     # xyzrender default
gif_fps = 10         # xyzrender default

if opt_gif_rot:
    gif_rot_axis = st.selectbox(
        "--gif-rot axis",
        options=["y", "x", "z", "-x", "-y", "-z", "xy", "xz", "yz", "yx", "zx", "zy"],
        index=0,
        help="Axis for GIF rotation animation. Default: 'y'.",
    )
    rot_frames = st.number_input(
        "--rot-frames (frames per full rotation)",
        min_value=1, max_value=720, value=120, step=1,
    )

if any([opt_gif_ts, opt_gif_trj, opt_gif_rot]):
    gif_fps = st.number_input(
        "--gif-fps (frames per second)",
        min_value=1, max_value=60, value=10, step=1,
    )


# =========================
# Sidebar controls
# =========================
st.sidebar.image("images/BNNLab_v3.png")
st.sidebar.header("Acknowledgements")
st.sidebar.write("This web tool was built to showcase **xyzrender**, a Python package developed by Alister Goodfellow. The code is available at https://github.com/aligfellow/xyzrender. This is a very powerful command line image generator for molecular modelling outputs. It is best used in that manner and this tool should be used only as a demonstrator. The full instructions for the tool can be found at the Github link.")
st.sidebar.header("Disclaimer")
st.sidebar.write("This software was developed by BNNLab, with all rights reserved, further developments made by James O'Brien. It is offered 'as is', without warranty of any kind, express or implied. The user assumes all risk for any malfunctions, errors, or damages resulting from the use of this software. The creator is not responsible for any direct or indirect loss arising from its use.")

# --------------------------
# Interactive 3D orientation viewer (.xyz and .out uploads)
# --------------------------
_is_xyz_upload = uploaded is not None and uploaded.name.lower().endswith(".xyz")

# Structure shown in the 3D viewer: raw XYZ for .xyz uploads, or a single
# frame extracted from the QM output for .out uploads.
_viewer_xyz = None
if uploaded is not None:
    # Reset stored view when a new file is uploaded
    _file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_file_key") != _file_key:
        st.session_state.pop("mol_view", None)
        st.session_state["_file_key"] = _file_key

    if _is_xyz_upload:
        _viewer_xyz = uploaded.getvalue().decode(errors="replace")
    else:
        with st.spinner("Extracting structure for orientation preview..."):
            _viewer_xyz = _extract_xyz_for_viewer(uploaded.getvalue(), uploaded.name)

if _viewer_xyz:
    st.subheader("Orientation")
    st.caption("Drag to rotate · Scroll to zoom — then click **Render** to capture this view.")

    # Show the interactive viewer; it returns the current view quaternion on mouse-up
    _view_result = mol_viewer(xyz_str=_viewer_xyz, key="mol_viewer")
    if _view_result is not None:
        st.session_state["mol_view"] = _view_result
elif uploaded is not None and not _is_xyz_upload:
    st.caption("Could not extract a structure from this file for the orientation preview; rendering will use auto-orientation.")

_has_view = "mol_view" in st.session_state and _viewer_xyz is not None
if _has_view:
    st.info("Orientation captured — click **Render** to generate the image from this view.")

# Optional: soft warning for contradictory flags
if uploaded is None and not smiles.strip():
    st.warning("No input file uploaded and no SMILES provided. Please upload a .xyz file or enter a SMILES string to proceed.")

if uploaded is not None and smiles.strip():
    st.warning("You provided both a file upload and a SMILES string. The file upload will take precedence over the file upload for rendering.")

if opt_bo and opt_no_bo:
    st.warning(
        "You selected both `--bo` and `--no-bo`. That may be contradictory for xyzrender.")

if opt_hy and opt_no_hy:
    st.warning(
        "You selected both `--hy` and `--no-hy`. That may be contradictory for xyzrender.")

config_options = [opt_flat, opt_paton, opt_pmol, opt_skeletal,
                  opt_bubble, opt_tube, opt_btube, opt_mtube, opt_wire, opt_graph]

if sum(config_options) > 1:
    st.warning(
        "You selected multiple `--config` options. That may be contradictory for xyzrender.")

# Determine mode based on GIF flags (GIF mode if any selected)
gif_flags_selected = any([opt_gif_ts, opt_gif_trj, opt_gif_rot])

expected_ext = ".gif" if gif_flags_selected else ".svg"
expected_mime = "image/gif" if gif_flags_selected else "image/svg+xml"
st.caption(f"Expected output type: **{expected_ext.upper()[1:]}**")

# --------------------------
# Render button
# --------------------------
run_btn = st.button("Render", type="primary", disabled=(
    uploaded is None and not smiles.strip()))

if run_btn:
    if uploaded is None and not smiles.strip():
        st.error("Please upload a .xyz file or enter a SMILES string.")
        st.stop()

    # Normalize output name extension to match expected mode
    desired = Path(out_name)
    if desired.suffix.lower() != expected_ext:
        corrected = desired.with_suffix(expected_ext).name
        st.info(
            f"Adjusted output file extension to `{expected_ext}` → **{corrected}**")
        out_name = corrected

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_path = tmpdir_path / out_name

        # Captured orientation -> rotation matrix (shared by both input types).
        R = None
        if _has_view:
            try:
                view = st.session_state["mol_view"]
                qx, qy, qz, qw = view[4], view[5], view[6], view[7]
                R = _quaternion_to_rotation_matrix(qx, qy, qz, qw)
            except Exception as rot_err:
                st.error(f"Failed to apply viewer orientation: {rot_err}")
                st.stop()

        # .out (or other QM/multi-frame) inputs can't be re-read with rotated
        # coordinates, and xyzrender only honours a custom orientation on those
        # paths via its interactive viewer (-I). render_driver.py reproduces -I
        # non-interactively from the captured rotation matrix.
        use_driver = R is not None and uploaded is not None and not _is_xyz_upload

        # Save uploaded input, preserving its original extension so xyzrender
        # can detect the format (.xyz, .out, …).
        if uploaded is not None:
            in_name = uploaded.name or "input.xyz"
            if not Path(in_name).suffix:
                in_name = Path(in_name).stem + ".xyz"
            in_path = tmpdir_path / in_name
            xyz_data = uploaded.getvalue()
            if R is not None and _is_xyz_upload:
                # XYZ: rotate coordinates in place and disable auto-orientation.
                xyz_data = _apply_rotation_to_xyz(xyz_data, R)
            in_path.write_bytes(xyz_data)

        # Build xyzrender argument list: [input.xyz or smiles] [flags] [output]
        if uploaded:
            cmd = [str(in_path)]
        else:
            cmd = ["--smi", smiles.strip()]

        # Insert flags immediately after input path
        if opt_bo:
            cmd.append("--bo")
        if opt_no_bo:
            cmd.append("--no-bo")
        if opt_vdw:
            cmd.append("--vdw")
        if opt_ts:
            cmd.append("--ts")
        if opt_nci:
            cmd.append("--nci")
        if opt_gif_ts:
            cmd.append("--gif-ts")
        if opt_gif_trj:
            cmd.append("--gif-trj")
        if opt_gif_rot:
            cmd += ["--gif-rot", gif_rot_axis, "--rot-frames", str(rot_frames)]
        if any([opt_gif_ts, opt_gif_trj, opt_gif_rot]):
            cmd += ["--gif-fps", str(gif_fps)]
        if opt_hy:
            cmd.append("--hy")
        if opt_no_hy:
            cmd.append("--no-hy")
        if opt_flat:
            cmd += ["--config", "flat"]
        if opt_paton:
            cmd += ["--config", "paton"]
        if opt_pmol:
            cmd += ["--config", "pmol"]
        if opt_skeletal:
            cmd += ["--config", "skeletal"]
        if opt_bubble:
            cmd += ["--config", "bubble"]
        if opt_tube:
            cmd += ["--config", "tube"]
        if opt_btube:
            cmd += ["--config", "btube"]
        if opt_mtube:
            cmd += ["--config", "mtube"]
        if opt_wire:
            cmd += ["--config", "wire"]
        if opt_graph:
            cmd += ["--config", "graph"]

        # Orientation handling:
        # - .out captured view: run via the driver in -I mode so xyzrender
        #   applies the captured rotation (auto-orient is disabled by -I).
        # - .xyz captured view: coords already rotated, so disable auto-orient.
        # - explicit --no-orient checkbox always honoured.
        if use_driver:
            cmd.append("-I")
        if opt_no_orient or (R is not None and _is_xyz_upload):
            cmd.append("--no-orient")

        # Output handling per your rules:
        # - SVG mode: use "-o <out.svg>"
        # - GIF mode: use "-go <out.gif>" and DO NOT add "-o"
        if gif_flags_selected:
            cmd += ["-go", str(out_path)]
        else:
            cmd += ["-o", str(out_path)]

        # Assemble the executable command. For captured orientations on .out
        # inputs, run through render_driver.py (which injects the rotation into
        # xyzrender's -I path); otherwise invoke the xyzrender CLI directly.
        if use_driver:
            rot_path = tmpdir_path / "rotation.txt"
            np.savetxt(rot_path, R)
            driver = Path(__file__).resolve().parent / "render_driver.py"
            cmd = [sys.executable, str(driver), str(rot_path), *cmd]
        else:
            cmd = ["xyzrender", *cmd]

        # st.write("### Command to be executed")
        # st.code(" ".join(cmd))

        with st.spinner("Running xyzrender..."):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                st.error("Rendering failed.")
                if e.stdout:
                    st.subheader("stdout")
                    st.code(e.stdout)
                if e.stderr:
                    st.subheader("stderr")
                    st.code(e.stderr)
                st.stop()
            except FileNotFoundError:
                st.error(
                    "`xyzrender` not found. Ensure it is installed and on PATH.")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        # Validate output
        if not out_path.exists() or out_path.stat().st_size == 0:
            st.error(
                f"xyzrender did not produce the expected output file: **{out_path.name}**.")
            if proc.stdout:
                st.subheader("stdout")
                st.code(proc.stdout)
            if proc.stderr:
                st.subheader("stderr")
                st.code(proc.stderr)
            st.stop()

        file_bytes = out_path.read_bytes()
        st.success(f"Rendering complete! Generated: **{out_path.name}**")

        # Preview depending on output type
        if expected_ext == ".svg":
            # Display SVG using HTML (avoid PIL)
            try:
                svg_str = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                svg_str = ""
            if svg_str:
                st.write("### Preview (SVG)")
                st.markdown(
                    f'<div style="border:1px solid #ccc;padding:10px;overflow:auto;max-height:75vh;">{svg_str}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info(
                    "SVG preview not available; you can download the file below.")
        else:
            # GIF preview
            st.write("### Preview (GIF)")
            st.image(file_bytes, caption=out_path.name, use_column_width=True)

        # Download button with correct MIME
        st.download_button(
            label=f"Download {expected_ext.upper()[1:]}",
            data=file_bytes,
            file_name=out_path.name,
            mime=expected_mime,
        )

        # Show process outputs
        with st.expander("Process Output"):
            if proc.stdout.strip():
                st.subheader("stdout")
                st.code(proc.stdout)
            if proc.stderr.strip():
                st.subheader("stderr")
                st.code(proc.stderr)
