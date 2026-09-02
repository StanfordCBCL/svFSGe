"""
Neural operator surrogate for WSS and pressure prediction in svFSGe.

Replaces the MESH + FLUID svFSI solvers within each FSI coupling sub-iteration.

Per-sub-iteration inference pipeline:
  1. Compute current solid interface positions = solid_xyz + solid_disp
  2. Write surface as legacy ASCII VTK (triangulated)
  3. Run LDDMM: register cylinder.vtk → current surface
     → reads back 1-shoot-16.vtk (deformed cylinder nodes in physical space)
  4. Compute LDDMM displacement at cylinder nodes:
       disp_cyl = shoot16_pts - cylinder_pts
  5. Project onto SVD basis → geometry branch coefficients
  6. Pull back solid FEM nodes to reference (cylinder) space:
       For each FEM node, find k nearest shoot16 nodes;
       the corresponding cylinder nodes give the reference position
       (cylinder_node[i] ↔ shoot16_node[i] by LDDMM particle correspondence)
  7. NN forward passes (WSS + pressure): trunk = pulled-back FEM positions,
     branch = coefficients
     → WSS (N_solid, 3) and pressure (N_solid,) at solid FEM interface nodes

Required config keys (under "neural_operator" in JSON):
  enabled           – bool
  pt_file           – path to WSS shear_stress_model.pt
  pressure_pt_file  – path to pressure shear_stress_model.pt (required)
  svd_basis_file    – path to *_basis.npz  (Ux, Uy, Uz)
  svd_template_vtk  – path to cylinder.vtk  (672 nodes, LDDMM template)
  work_dir          – writable scratch directory for VTK files and LDDMM output
  branch_dims       – list[int]
  trunk_dims        – list[int]
  final_dim         – int
  mode              – int   (SVD modes; default = branch_dims[0] // 3)
  idw_k             – int   (IDW neighbours; default 8)
  idw_power         – float (IDW exponent;  default 2)
  model_dir         – path to ShapeOperatorLearning (optional)

  lddmm_backend     – "matlab" (default), "matlab_engine", or "python"
                        "matlab"        : spawns matlab -batch per call (~15 s startup each)
                        "matlab_engine" : starts one persistent MATLAB engine at init;
                                          subsequent calls have no restart overhead (~0 s)
                        "python"        : Python/PyTorch port; no MATLAB needed

  -- matlab / matlab_engine backends --
  matlab_exe        – path to MATLAB executable (default ~/matlab/bin/matlab)
  lddmm_script_dir  – directory containing lddmm_register_single.m
  matlab_timeout    – int   MATLAB timeout in seconds (default 600)
  matlab_engine_opts – str  options passed to matlab.engine.start_matlab()
                            (default "-nojvm -nodisplay -nosplash")

  -- python backend only --
  fshapes_tk_dir    – path to fshapesTk repo root (default ~/fshapesTk)
  lddmm_optimizer   – "hanso" (default, matches MATLAB BFGS) or "lbfgsb"
"""

import os
import sys
import importlib
import subprocess
import tempfile

import numpy as np
import torch
from scipy.spatial import cKDTree

import vtk
from vtk.util.numpy_support import vtk_to_numpy


# ---------------------------------------------------------------------------
# VTK I/O helpers
# ---------------------------------------------------------------------------

def _load_vtk_points(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float32)


def _write_surface_vtk(polydata, xyz, out_path):
    """
    Write a vtkPolyData with updated node positions in fshapesTk-compatible
    VTK 2.0 ASCII format (one point per line, POLYGONS with '3 v1 v2 v3' rows).
    Applies vtkTriangleFilter to ensure triangulated cells.
    """
    # Triangulate (interface.vtp cells may be quads)
    poly = vtk.vtkPolyData()
    poly.DeepCopy(polydata)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    tri_pd = tri.GetOutput()

    n_pts   = len(xyz)
    n_cells = tri_pd.GetNumberOfCells()

    with open(out_path, "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("fshape interface\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for x, y, z in xyz:
            f.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        f.write(f"\nPOLYGONS {n_cells} {4 * n_cells}\n")
        id_list = vtk.vtkIdList()
        for ci in range(n_cells):
            tri_pd.GetCellPoints(ci, id_list)
            v0 = id_list.GetId(0)
            v1 = id_list.GetId(1)
            v2 = id_list.GetId(2)
            f.write(f"3 {v0} {v1} {v2}\n")


# ---------------------------------------------------------------------------
# IDW helpers
# ---------------------------------------------------------------------------

def _idw_weights(src_xyz, dst_xyz, k, power):
    """Return (indices, weights) for IDW from src to dst."""
    tree = cKDTree(src_xyz)
    dists, idxs = tree.query(dst_xyz, k=k)
    dists = dists.astype(np.float32)

    exact = dists[:, 0] == 0.0
    dists[exact, :] = 1.0
    w = 1.0 / dists ** power
    w[exact, :] = 0.0
    w[exact, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    return idxs, w


def _apply_idw(values, indices, weights):
    """(N_dst, C) = IDW interpolation of (N_src, C) values."""
    return (values[indices] * weights[:, :, None]).sum(axis=1)


# ---------------------------------------------------------------------------
# NeuralOperator
# ---------------------------------------------------------------------------

class NeuralOperator:
    def __init__(self, cfg):
        model_dir = cfg.get(
            "model_dir",
            os.path.expanduser("~/ShapeOperatorLearning"),
        )
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        model_module = importlib.import_module("models.shearStressNN_coeff")
        ShearStressNN = getattr(model_module, "ShearStressNN")

        branch_dims = list(cfg["branch_dims"])
        self.mode   = cfg.get("mode", branch_dims[0] // 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SVD basis — lives in cylinder template space (N_cyl, mode)
        basis    = np.load(os.path.expanduser(cfg["svd_basis_file"]))
        self.Ux  = basis["Ux"][:, :self.mode].astype(np.float32)
        self.Uy  = basis["Uy"][:, :self.mode].astype(np.float32)
        self.Uz  = basis["Uz"][:, :self.mode].astype(np.float32)

        # Cylinder template node positions (N_cyl, 3) — reference for SVD and pull-back
        self._source_vtk_path = os.path.expanduser(cfg["svd_template_vtk"])
        self.cyl_xyz = _load_vtk_points(self._source_vtk_path)

        # LDDMM backend selection
        self.lddmm_backend = cfg.get("lddmm_backend", "matlab")
        self.work_dir      = os.path.expanduser(cfg.get("work_dir", "~/fsg_no_work"))
        os.makedirs(self.work_dir, exist_ok=True)

        if self.lddmm_backend == "direct":
            # No registration. The svFSGe solid interface nodes coincide with the
            # cylinder template (same cylinder.py mesh), so the displacement is
            # projected straight onto the SVD basis — exactly how the FSG-direct
            # models were trained. No MATLAB / fshapesTk, no per-call LDDMM.
            self._direct_map = None
            self._direct_tol = cfg.get("direct_match_tol", 1e-4)

        elif self.lddmm_backend == "python":
            # Python/PyTorch port — import once, reuse across all calls
            fshapes_dir = os.path.expanduser(
                cfg.get("fshapes_tk_dir", "~/fshapesTk")
            )
            if fshapes_dir not in sys.path:
                sys.path.insert(0, fshapes_dir)
            from python.io_vtk import import_fshape_vtk as _import_vtk
            from python.matching import match_geom, export_matching
            import torch as _torch
            self._py_import_vtk    = _import_vtk
            self._py_match_geom    = match_geom
            self._py_export        = export_matching
            self._py_device        = "cuda" if _torch.cuda.is_available() else "cpu"
            self._py_dtype         = _torch.float64
            self._lddmm_optimizer  = cfg.get("lddmm_optimizer", "hanso")

        elif self.lddmm_backend == "matlab_engine":
            # Persistent MATLAB engine — start once, reuse every call (~0 s overhead)
            import matlab.engine as _me
            self.matlab_exe       = cfg.get("matlab_exe", "/home/shiyi/matlab/bin/matlab")
            self.lddmm_script_dir = os.path.expanduser(
                cfg.get("lddmm_script_dir", "~/TAA_CFD_pipeline")
            )
            self.matlab_timeout   = cfg.get("matlab_timeout", 600)
            eng_opts = cfg.get("matlab_engine_opts", "-nojvm -nodisplay -nosplash")
            print(f"[NeuralOperator] Starting persistent MATLAB engine ({eng_opts}) ...")
            self._matlab_engine = _me.start_matlab(eng_opts)
            self._matlab_engine.addpath(self.lddmm_script_dir, nargout=0)
            print(f"[NeuralOperator] MATLAB engine ready.")

        else:
            # MATLAB subprocess backend (original behaviour)
            self.matlab_exe       = cfg.get("matlab_exe", "/home/shiyi/matlab/bin/matlab")
            self.lddmm_script_dir = os.path.expanduser(
                cfg.get("lddmm_script_dir", "~/TAA_CFD_pipeline")
            )
            self.matlab_timeout   = cfg.get("matlab_timeout", 600)

        # IDW parameters
        self.idw_k     = cfg.get("idw_k", 8)
        self.idw_power = cfg.get("idw_power", 2.0)

        trunk_dims = list(cfg["trunk_dims"])
        final_dim  = cfg.get("final_dim", 64)

        # WSS model. The FSG coupling consumes only |WSS| (svfsi.py reduces the
        # vector via np.linalg.norm), so a dedicated magnitude model (out_dim=1)
        # is both the correct target and more accurate. If wss_magnitude_pt_file
        # is given we use it and deliver the scalar as a vector with the
        # magnitude in the axial component (downstream norm recovers it exactly).
        # A stimulus model (out_dim=1) predicts the G&R growth stimulus
        # s = 1 - |WSS|/tau_o directly; we reconstruct |WSS| = tau_o*(1 - s) and
        # deliver it through the same scalar path as the magnitude model. tau_o
        # (wss_homeo) is the homeostatic |WSS| (prestress reference). This is the
        # quantity the growth law consumes, so training on it focuses capacity
        # where it matters.
        self.wss_stimulus_pt = cfg.get("wss_stimulus_pt_file")
        self.wss_magnitude_pt = cfg.get("wss_magnitude_pt_file")
        self.wss_homeo = cfg.get("wss_homeo", 0.011841)
        self.wss_is_stimulus = False
        self.wss_is_magnitude = False
        if self.wss_stimulus_pt:
            self.model = self._load_model(
                ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=1,
                pt_file=self.wss_stimulus_pt,
            )
            self.wss_is_stimulus = True
        elif self.wss_magnitude_pt:
            self.model = self._load_model(
                ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=1,
                pt_file=self.wss_magnitude_pt,
            )
            self.wss_is_magnitude = True
        else:
            self.model = self._load_model(
                ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=3,
                pt_file=cfg["pt_file"],
            )

        # Pressure model (out_dim=1) — required: solid solver always needs interface pressure
        pressure_pt = cfg["pressure_pt_file"]
        self.pressure_model = self._load_model(
            ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=1,
            pt_file=pressure_pt,
        )
        print(f"[NeuralOperator] pressure model: {pressure_pt}")

        if self.wss_is_stimulus:
            print(f"[NeuralOperator] WSS model (STIMULUS 1-tau/tauo, out_dim=1): {self.wss_stimulus_pt}")
            print(f"[NeuralOperator]   tau_o (wss_homeo) = {self.wss_homeo:.6f}; |WSS|=tau_o*(1-s)")
        elif self.wss_is_magnitude:
            print(f"[NeuralOperator] WSS model (magnitude, out_dim=1): {self.wss_magnitude_pt}")
        else:
            print(f"[NeuralOperator] WSS model (vector, out_dim=3): {cfg['pt_file']}")
        print(f"[NeuralOperator] cylinder template: {len(self.cyl_xyz)} nodes, modes: {self.mode}")
        print(f"[NeuralOperator] LDDMM backend: {self.lddmm_backend}")
        if self.lddmm_backend == "direct":
            print(f"[NeuralOperator] direct (no registration), match tol: {self._direct_tol:.0e}")
        elif self.lddmm_backend == "python":
            print(f"[NeuralOperator] Python LDDMM device: {self._py_device}, optimizer: {self._lddmm_optimizer}")
        elif self.lddmm_backend == "matlab_engine":
            print(f"[NeuralOperator] MATLAB engine: persistent, script_dir={self.lddmm_script_dir}")
        else:
            print(f"[NeuralOperator] MATLAB subprocess: {self.matlab_exe}")
        print(f"[NeuralOperator] work dir: {self.work_dir}")

    def _load_model(self, cls, branch_dims, trunk_dims, final_dim, out_dim, pt_file):
        model = cls(
            branch_dims=branch_dims,
            trunk_dims=trunk_dims,
            final_dim=final_dim,
            out_dim=out_dim,
        )
        ckpt  = torch.load(os.path.expanduser(pt_file), map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # LDDMM
    # ------------------------------------------------------------------

    def _run_lddmm(self, target_vtk, output_dir):
        """Register cylinder → target_vtk; return shoot16 pts (N_cyl, 3).

        Dispatches to the backend selected by cfg["lddmm_backend"]:
          "matlab"        — spawns matlab -batch (~15 s startup per call)
          "matlab_engine" — calls persistent MATLAB engine (no startup overhead)
          "python"        — calls Python/PyTorch port (no MATLAB needed)
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.lddmm_backend == "python":
            return self._run_lddmm_python(target_vtk, output_dir)
        elif self.lddmm_backend == "matlab_engine":
            return self._run_lddmm_engine(target_vtk, output_dir)
        else:
            return self._run_lddmm_matlab(target_vtk, output_dir)

    def _run_lddmm_engine(self, target_vtk, output_dir):
        """Persistent MATLAB engine backend — zero per-call restart cost."""
        print(f"[NeuralOperator] running LDDMM (MATLAB engine) ...")
        eng = self._matlab_engine
        try:
            # lddmm_register_single.m calls restoredefaultpath internally, so we
            # re-add the script dir each call via eval before invoking the function.
            eng.eval(f"addpath('{self.lddmm_script_dir}');", nargout=0)
            eng.lddmm_register_single(
                self._source_vtk_path,
                target_vtk,
                output_dir,
                nargout=0,
            )
        except Exception as e:
            raise RuntimeError(f"MATLAB engine LDDMM failed: {e}") from e
        shoot_vtk = os.path.join(output_dir, "1-shoot-16.vtk")
        if not os.path.exists(shoot_vtk):
            raise FileNotFoundError(f"LDDMM output not found: {shoot_vtk}")
        return _load_vtk_points(shoot_vtk)

    def shutdown(self):
        """Cleanly stop the persistent MATLAB engine if running."""
        if hasattr(self, "_matlab_engine") and self._matlab_engine is not None:
            try:
                self._matlab_engine.quit()
            except Exception:
                pass
            self._matlab_engine = None
            print("[NeuralOperator] MATLAB engine stopped.")

    def _run_lddmm_matlab(self, target_vtk, output_dir):
        """Original MATLAB subprocess backend."""
        source_vtk = self._source_vtk_path
        cmd = [
            self.matlab_exe,
            "-nojvm", "-nodisplay", "-nosplash",
            "-sd", self.lddmm_script_dir,
            "-batch",
            (f"lddmm_register_single("
             f"'{source_vtk}', "
             f"'{target_vtk}', "
             f"'{output_dir}')")
        ]
        print(f"[NeuralOperator] running LDDMM (MATLAB) ...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.matlab_timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB LDDMM failed (exit {result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        shoot_vtk = os.path.join(output_dir, "1-shoot-16.vtk")
        if not os.path.exists(shoot_vtk):
            raise FileNotFoundError(
                f"LDDMM output not found: {shoot_vtk}\n"
                f"MATLAB stdout:\n{result.stdout}"
            )
        return _load_vtk_points(shoot_vtk)

    def _run_lddmm_python(self, target_vtk, output_dir):
        """Python/PyTorch backend — no MATLAB, no per-call startup cost."""
        print(f"[NeuralOperator] running LDDMM (Python/{self._py_device}) ...")
        source = self._py_import_vtk(self._source_vtk_path)
        target = self._py_import_vtk(target_vtk)

        defo_params = {
            'kernel_size_mom': [0.3, 0.2],
            'nb_euler_steps': 15,
        }
        objfun_params = {
            'distance': 'kernel',
            'kernel_distance': {
                'distance': 'var',
                'kernel_size_geom': 0.3,
                'kernel_size_signal': 1.8,
            },
            'weight_coef_dist': 3000,
            'weight_coef_pen_fr': 0.03,
            'weight_coef_pen_f': 0,
            'weight_coef_pen_p': 1,
        }
        optim_params = {
            'bfgs': {'maxit': 30},
            'optimizer': self._lddmm_optimizer,
        }

        momentums, _ = self._py_match_geom(
            source, target, defo_params, objfun_params, optim_params,
            device=self._py_device, dtype=self._py_dtype, verbose=False,
        )
        self._py_export(
            source, momentums, target, {}, output_dir,
            defo_params, device=self._py_device, dtype=self._py_dtype,
        )

        shoot_vtk = os.path.join(output_dir, "1-shoot-16.vtk")
        return _load_vtk_points(shoot_vtk)

    # ------------------------------------------------------------------
    # Pull-back
    # ------------------------------------------------------------------

    def _pull_back(self, solid_xyz, shoot16_pts):
        """
        Pull solid FEM interface nodes from current (physical) space back to
        cylinder reference space.

        LDDMM gives a particle correspondence: cyl_xyz[i] ↔ shoot16_pts[i].
        For each solid FEM node (in physical space), find k nearest shoot16 nodes,
        then IDW-interpolate the corresponding cylinder positions.

        Parameters
        ----------
        solid_xyz   : (N_solid, 3) solid FEM reference positions (physical)
        shoot16_pts : (N_cyl, 3)  deformed cylinder node positions (physical)

        Returns
        -------
        (N_solid, 3) solid FEM nodes pulled back to cylinder reference space
        """
        idxs, weights = _idw_weights(
            shoot16_pts, solid_xyz, self.idw_k, self.idw_power
        )
        return _apply_idw(self.cyl_xyz, idxs, weights)

    # ------------------------------------------------------------------
    # Batched NN forward pass
    # ------------------------------------------------------------------

    def _forward(self, model, coeffs_t, xyz_t, batch=1000):
        """Run model in batches; returns numpy array."""
        preds = []
        with torch.no_grad():
            for start in range(0, len(xyz_t), batch):
                out = model(
                    coeffs_t[start:start+batch],
                    xyz_t[start:start+batch],
                )
                preds.append(out.cpu())
        return torch.cat(preds).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Main inference entry points
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Geometry encoding — maps the current solid interface state to the NN
    # inputs (branch coefficients + trunk positions).  Two interchangeable
    # strategies, selected by cfg["lddmm_backend"]:
    #   "direct"                  -> _encode_direct      (no registration)
    #   "matlab"/"python"/"..."   -> _encode_registration (LDDMM)
    # Both return (coeffs (3*mode,), trunk_input (N_solid, 3)); the shared
    # predict_wss_and_pressure does the NN forward passes.
    # ------------------------------------------------------------------

    def _encode_geometry(self, solid_disp, solid_xyz, solid_mesh, call_id):
        """Dispatch to the configured geometry-encoding strategy."""
        if self.lddmm_backend == "direct":
            return self._encode_direct(solid_disp, solid_xyz)
        return self._encode_registration(solid_disp, solid_xyz, solid_mesh, call_id)

    def _encode_direct(self, solid_disp, solid_xyz):
        """
        Registration-free encoding (lddmm_backend == "direct").

        The solid interface nodes coincide with the cylinder template, so we:
          1. reorder the solid displacement into cylinder-template node order
             (exact nearest-neighbour on the undeformed reference positions),
          2. project onto the SVD basis to get the branch coefficients,
          3. use the solid reference positions directly as the trunk input
             (they ARE the cylinder reference positions).

        This mirrors extract_fsg_displacements.py + the training-time projection,
        so inference is consistent with how the models were trained.
        """
        from scipy.spatial import cKDTree

        # Cache the cylinder-node -> solid-node correspondence (reference is fixed).
        if self._direct_map is None:
            tree = cKDTree(solid_xyz)
            dist, idx = tree.query(self.cyl_xyz, k=1)
            if dist.max() > self._direct_tol:
                raise ValueError(
                    f"[NeuralOperator] direct backend: solid reference nodes do not "
                    f"match the cylinder template (max NN dist {dist.max():.3e} > "
                    f"{self._direct_tol:.0e}). Use an LDDMM backend instead."
                )
            self._direct_map = idx

        disp_cyl = solid_disp[self._direct_map]                 # (N_cyl, 3), cyl order
        coeff_x  = self.Ux.T @ disp_cyl[:, 0]
        coeff_y  = self.Uy.T @ disp_cyl[:, 1]
        coeff_z  = self.Uz.T @ disp_cyl[:, 2]
        coeffs   = np.concatenate([coeff_x, coeff_y, coeff_z]).astype(np.float32)

        # Trunk = cylinder reference positions of the solid nodes = solid_xyz itself.
        return coeffs, solid_xyz

    def _encode_registration(self, solid_disp, solid_xyz, solid_mesh, call_id):
        """
        LDDMM-registration encoding (lddmm_backend matlab/matlab_engine/python).

        Registers the cylinder template to the current interface, derives the SVD
        coefficients from the registered displacement, and pulls the solid FEM
        nodes back to cylinder reference space for the trunk input.
        """
        # 1. Current surface positions
        current_xyz = solid_xyz + solid_disp

        # 2. Write current surface as VTK
        target_vtk = os.path.join(self.work_dir, f"target_{call_id}.vtk")
        output_dir  = os.path.join(self.work_dir, f"lddmm_{call_id}")
        _write_surface_vtk(solid_mesh, current_xyz, target_vtk)

        # 3. Run LDDMM (once per sub-iteration)
        shoot16_pts = self._run_lddmm(target_vtk, output_dir)   # (N_cyl, 3)

        # 4. LDDMM displacement at cylinder nodes → SVD coefficients
        disp_cyl = shoot16_pts - self.cyl_xyz                   # (N_cyl, 3)
        coeff_x  = self.Ux.T @ disp_cyl[:, 0]                  # (mode,)
        coeff_y  = self.Uy.T @ disp_cyl[:, 1]
        coeff_z  = self.Uz.T @ disp_cyl[:, 2]
        coeffs   = np.concatenate([coeff_x, coeff_y, coeff_z]).astype(np.float32)

        # 5. Pull back solid FEM nodes to cylinder reference space
        trunk_input = self._pull_back(solid_xyz, shoot16_pts)   # (N_solid, 3)
        return coeffs, trunk_input

    def predict_wss_and_pressure(self, solid_disp, solid_xyz, solid_mesh, call_id=0):
        """
        Predict WSS and pressure at solid interface nodes.

        Encodes the current geometry into (branch coefficients, trunk positions)
        via the configured strategy (direct or LDDMM), then runs one forward pass
        per model.

        Parameters
        ----------
        solid_disp : (N_solid, 3) displacement of solid interface nodes
        solid_xyz  : (N_solid, 3) reference positions of solid interface nodes
        solid_mesh : vtkPolyData  solid interface mesh (for surface connectivity)
        call_id    : int          iteration counter (used to name temp files)

        Returns
        -------
        wss      : (N_solid, 3)
        pressure : (N_solid,)
        """
        solid_disp = np.asarray(solid_disp, dtype=np.float32)
        solid_xyz  = np.asarray(solid_xyz,  dtype=np.float32)

        # Geometry → NN inputs (strategy-specific)
        coeffs, trunk_input = self._encode_geometry(
            solid_disp, solid_xyz, solid_mesh, call_id)

        # Shared NN forward passes
        n_pts    = len(trunk_input)
        coeffs_t = torch.from_numpy(np.tile(coeffs, (n_pts, 1))).to(self.device)
        xyz_t    = torch.from_numpy(np.asarray(trunk_input, dtype=np.float32)).to(self.device)

        if self.wss_is_stimulus:
            # model predicts the growth stimulus s = 1 - |WSS|/tau_o; reconstruct
            # |WSS| = tau_o*(1 - s) and deliver it as a vector with the magnitude
            # in the axial (z) component (downstream norm recovers it exactly).
            s   = self._forward(self.model, coeffs_t, xyz_t).reshape(-1)     # (N_solid,)
            mag = (self.wss_homeo * (1.0 - s)).astype(np.float32)
            np.clip(mag, 0.0, None, out=mag)   # WSS magnitude is non-negative
            wss = np.zeros((len(mag), 3), dtype=np.float32)
            wss[:, 2] = mag
        elif self.wss_is_magnitude:
            # model predicts the scalar |WSS|; deliver it as a vector with the
            # magnitude in the axial (z) component so the downstream norm in the
            # FSG coupling (svfsi.py) recovers |WSS| exactly. (Direction is
            # discarded there; axial is the physically dominant component.)
            mag = self._forward(self.model, coeffs_t, xyz_t).reshape(-1)     # (N_solid,)
            wss = np.zeros((len(mag), 3), dtype=np.float32)
            wss[:, 2] = mag
        else:
            wss  = self._forward(self.model, coeffs_t, xyz_t)               # (N_solid, 3)
        pressure = self._forward(self.pressure_model, coeffs_t, xyz_t).squeeze(-1)
        return wss, pressure

    def predict_wss(self, solid_disp, solid_xyz, solid_mesh, call_id=0):
        """Backward-compatible wrapper that returns only WSS."""
        wss, _ = self.predict_wss_and_pressure(solid_disp, solid_xyz, solid_mesh, call_id)
        return wss
