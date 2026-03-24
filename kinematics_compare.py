#!/usr/bin/env python3
"""Read all output ROOT files and produce kinematics plots."""
import glob
import math
import os
import sys
from array import array
from collections import defaultdict
import ROOT as root
from libraries import FileNames, VolumeBoundaries, RunParameters

# Parse optional arguments: run=<N>, run=[N,M,...], group=True/False, cutg=True/False
_single_group = True
_run_filter = None   # None means all runs; otherwise a list of ints
_use_cutg = True     # True = apply TCutG filter; False = skip it
for _arg in sys.argv[1:]:
    if _arg.lower() in ("group=false", "group=0"):
        _single_group = False
    elif _arg.lower() in ("group=true", "group=1"):
        _single_group = True
    elif _arg.lower() in ("cutg=false", "cutg=0"):
        _use_cutg = False
    elif _arg.lower() in ("cutg=true", "cutg=1"):
        _use_cutg = True
    elif _arg.lower().startswith("run="):
        _raw = _arg.split("=", 1)[1].strip()
        # Accept both run=88 and run=[88,87]
        _raw = _raw.strip("[]")
        try:
            _run_filter = [int(x.strip()) for x in _raw.split(",") if x.strip()]
        except ValueError:
            print(f"WARNING: could not parse run number(s) from '{_arg}', ignoring")
_run_label = str(_run_filter) if _run_filter is not None else 'all'
print(f"Group filter: single_group={_single_group}, run filter: {_run_label}, use_cutg={_use_cutg}")

root.gROOT.SetBatch(False)
root.TH1.AddDirectory(False)

_NAN = -999.0

# --- Active volume bounds (10 mm border on all sides) ---
_AV_X_MIN, _AV_X_MAX = 3.0, 253.0   # x: 0+3 to 256-3 mm
_AV_Y_MIN, _AV_Y_MAX = 3.0, 253.0   # y: 0+3 to 256-3 mm
_AV_Z_MIN, _AV_Z_MAX = 3.0, 497.0   # z: 0+3 to 500-3 mm
_BEAM_Y_MIN = VolumeBoundaries.BEAM_ZONE_MIN.value - RunParameters.BEAM_ZONE_RELAXATION_MM.value
_BEAM_Y_MAX = VolumeBoundaries.BEAM_ZONE_MAX.value + RunParameters.BEAM_ZONE_RELAXATION_MM.value
_PHI_EXCL_LO, _PHI_EXCL_HI = 70.0, 110.0  # exclude phi in [70,110] and [-110,-70]

def _in_volume(x, y, z):
    return (_AV_X_MIN <= x <= _AV_X_MAX and
            _AV_Y_MIN <= y <= _AV_Y_MAX and
            _AV_Z_MIN <= z <= _AV_Z_MAX)

def _passes_filters(vtx_x, vtx_y, vtx_z, sx, sy, sz, ex, ey, ez, phi):
    if not _in_volume(vtx_x, vtx_y, vtx_z):
        return False
    if not (_BEAM_Y_MIN <= vtx_y <= _BEAM_Y_MAX):
        return False
    if not _in_volume(sx, sy, sz):
        return False
    if not _in_volume(ex, ey, ez):
        return False
    if VolumeBoundaries.BEAM_ZONE_MIN.value <= ey <= VolumeBoundaries.BEAM_ZONE_MAX.value:
        return False
    if phi == _NAN:
        return False
    if (_PHI_EXCL_LO <= phi <= _PHI_EXCL_HI) or (-_PHI_EXCL_HI <= phi <= -_PHI_EXCL_LO):
        return False
    return True

output_dir = FileNames.KINEMATICS_INPUT_DIR.value
images_dir = FileNames.IMAGES_DIR.value

# Load TCutG before opening other files
_prev_level = root.gErrorIgnoreLevel
root.gErrorIgnoreLevel = root.kFatal
_cutg_file = root.TFile(os.path.join(os.path.dirname(__file__), "cutg_ransac.root"), "READ")
root.gErrorIgnoreLevel = _prev_level
_cutg = _cutg_file.Get("cutg_ransac")
if not _cutg:
    print("ERROR: cutg_ransac not found in cutg_ransac.root")
    raise SystemExit(1)

# Collect output files (optionally filtered by run numbers)
if _run_filter is not None:
    files = []
    for _rn in _run_filter:
        _f = sorted(glob.glob(os.path.join(output_dir, f"output_run_{_rn:04d}_*.root")))
        if not _f:
            _f = sorted(glob.glob(os.path.join(output_dir, f"output_run_{_rn}_*.root")))
        files.extend(_f)
    files = sorted(set(files))
else:
    files = sorted(glob.glob(os.path.join(output_dir, "output_run_*.root")))
if not files:
    print(f"No output files found in {output_dir}")
    raise SystemExit(1)

# Chain all trees
chain = root.TChain("track_data")
for fpath in files:
    chain.Add(fpath)
n_entries = chain.GetEntries()
print(f"Loaded {len(files)} file(s), {n_entries} entries")

# Disable branches that are never read in the event loop (speeds up GetEntry significantly)
chain.SetBranchStatus("*", 0)
for _br in (
    "event_status", "event_id", "run_number",
    "ransac_n_vtx_groups", "ransac_vtx_mult",
    "ransac_theta", "ransac_range_mm", "ransac_phi",
    "ransac_vertex_x", "ransac_vertex_y", "ransac_vertex_z",
    "ransac_start_x", "ransac_start_y", "ransac_start_z",
    "ransac_end_x",   "ransac_end_y",   "ransac_end_z",
    "ransac_cp_y",
    "reg_n_vtx_groups", "reg_vtx_mult",
    "reg_theta",  "reg_range_mm",  "reg_phi",
    "reg_vertex_x",  "reg_vertex_y",  "reg_vertex_z",
    "reg_start_x",  "reg_start_y",  "reg_start_z",
    "reg_end_x",    "reg_end_y",    "reg_end_z",
    "reg_cp_y",
    "reg_theta2", "reg_range_comb_mm", "reg_phi2",
    "reg_vertex2_x", "reg_vertex2_y", "reg_vertex2_z",
    "reg_cp_comb_y",
    "ransac_energy_keV",
    "reg_energy_keV",
    "reg_energy_comb_keV",
):
    chain.SetBranchStatus(_br, 1)

# Hoist hasattr checks — do them once before the loop
_has_status   = bool(chain.GetBranch("event_status"))
_has_event_id = bool(chain.GetBranch("event_id"))
_has_run      = bool(chain.GetBranch("run_number"))

# Rejection-reason tracking for GMM (REG scatter) tracks
# Each entry: (event_id, detail_string)
_gmm_rejected = defaultdict(list)  # reason -> [(event_id, detail), ...]
_gmm_low_angle_events = []          # events where GMM passed [25,50]°/[25,50]mm but RANSAC did not
_WIN1_LO, _WIN1_HI = 85, 95        # window 1: near-90° region
_WIN2_LO, _WIN2_HI = 20, 35        # window 2: low-angle region
_WIN3_THETA_LO, _WIN3_THETA_HI = 87, 89   # window 3: theta [87,89]°
_WIN3_RNG_LO,   _WIN3_RNG_HI   = 25, 30   # window 3: range [25,30] mm
_gmm_win1_events = []               # GMM passed [WIN1] window but RANSAC did not
_gmm_win2_events = []               # GMM passed [WIN2] window but RANSAC did not
_gmm_win3_events = []               # GMM passed theta[87,89]+range[25,30] but RANSAC did not
_ransac_90_failures = []            # RANSAC filter-failure reasons for tracks near 90 deg / 20 mm
_RANSAC_90_THETA_LO, _RANSAC_90_THETA_HI = 85.0, 95.0
_RANSAC_90_RNG_LO,   _RANSAC_90_RNG_HI   = 15.0, 25.0

# Target excitation bin: E*=0 MeV, theta_CM=1.5 deg  (bin width 0.75 MeV x 0.3 deg)
_EX_BIN_TCM_LO, _EX_BIN_TCM_HI = 1.35, 1.65    # theta_CM bin edges (deg)
_EX_BIN_EX_LO,  _EX_BIN_EX_HI  = -0.375, 0.375 # E* bin edges (MeV)
_ex_bin_events = []  # events: GMM+resp in bin, RANSAC missing, with per-filter breakdown

# Containers: (theta, range) and (total_charge, track_length)
ransac_range_angle = ([], [])       # (theta, range)
reg_range_angle    = ([], [])
reg_comb_range_angle = ([], [])

ransac_theta_energy   = ([], [])   # (theta_deg, E_MeV) passing same filters
reg_theta_energy      = ([], [])
reg_comb_theta_energy = ([], [])

ransac_charge_length = ([], [], [])   # (track_length, total_charge, theta)
reg_charge_length    = ([], [], [])
reg_comb_charge_length = ([], [], [])

_T_BEAM = 2842.0  # MeV (58Ni beam kinetic energy) — defined here for _lab_to_ex

def _lab_to_ex(theta_lab_deg, T_alpha_MeV):
    """
    Back-calculate (theta_CM_deg, E*_MeV) from measured (theta_lab, T_alpha) of
    the 4He ejectile for 58Ni(4He,4He)58Ni* in inverse kinematics.
    """
    import math as _m
    U       = 931.5
    m_beam  = 57.93 * U
    m_tgt   = 4.002 * U
    m_light = 4.002 * U
    E_beam  = m_beam + _T_BEAM
    p_beam  = _m.sqrt(E_beam**2 - m_beam**2)
    E_alpha = m_light + T_alpha_MeV
    p_alpha = _m.sqrt(max(0.0, E_alpha**2 - m_light**2))
    rad     = _m.radians(theta_lab_deg)
    pz_a    = p_alpha * _m.cos(rad)
    pp_a    = p_alpha * _m.sin(rad)
    E_rec   = (E_beam + m_tgt) - E_alpha
    pz_rec  = p_beam - pz_a
    M2 = E_rec**2 - pz_rec**2 - pp_a**2
    if M2 < 0:
        return None, None
    Ex = _m.sqrt(M2) - 57.93 * U
    sqrt_s   = _m.sqrt(m_beam**2 + m_tgt**2 + 2.0 * E_beam * m_tgt)
    beta_cm  = p_beam / (E_beam + m_tgt)
    gamma_cm = (E_beam + m_tgt) / sqrt_s
    pz_a_cm  = gamma_cm * (pz_a - beta_cm * E_alpha)
    theta_cm = _m.degrees(_m.atan2(pp_a, pz_a_cm))
    if theta_cm < 0:
        theta_cm += 180.0
    theta_cm = 180.0 - theta_cm   # convert to user convention (0° = beam direction)
    return theta_cm, Ex

ransac_excitation   = ([], [])   # (theta_CM_deg, E*_MeV)
reg_excitation      = ([], [])
reg_comb_excitation = ([], [])

_current_run = None
for i in range(n_entries):
    chain.GetEntry(i)
    if _has_run:
        _this_run = int(chain.run_number)
        if _this_run != _current_run:
            _current_run = _this_run
            print(f"Processing run {_current_run}...")
    status = chain.event_status if _has_status else 0
    if status != 0:
        continue

    event_id = int(chain.event_id) if _has_event_id else i

    # per-event tracking for the diagnostic print
    _ransac_passed_88_90 = []   # (theta,) for passing RANSAC tracks in 88-90°
    _reg_passed = False         # any REG scatter track passed all filters+cutg
    _reg_has_any_track = False  # GMM produced at least one scatter track candidate
    _ransac_passed = False      # RANSAC passed at least one track this event
    _evt_gmm_reasons = []       # [(reason, detail_dict), ...] — committed only if RANSAC passed & GMM didn't
    _ransac_passing_tracks = [] # passing RANSAC track info for direction diagnostics
    _gmm_low_angle_passing = [] # GMM tracks passing all filters with theta in [25,50] and range in [25,50]
    _ransac_in_low_angle = []   # RANSAC tracks passing all filters with theta in [25,50] and range in [25,50]
    _ransac_all_tracks = []     # ALL RANSAC tracks in event, regardless of filter outcome
    _gmm_win1_passing = []      # GMM tracks passing all filters with theta in [WIN1_LO, WIN1_HI]
    _ransac_win1_passing = []   # RANSAC tracks passing all filters with theta in [WIN1_LO, WIN1_HI]
    _gmm_win2_passing = []      # GMM tracks passing all filters with theta in [WIN2_LO, WIN2_HI]
    _ransac_win2_passing = []   # RANSAC tracks passing all filters with theta in [WIN2_LO, WIN2_HI]
    _gmm_win3_passing = []      # GMM tracks passing all filters with theta[87,89] AND range[25,30]
    _ransac_win3_passing = []   # RANSAC tracks passing all filters with theta[87,89] AND range[25,30]
    _ransac_90_fail_reasons = []  # filter-failure reasons for RANSAC tracks near 90 deg/20 mm
    _ransac_all_fails     = []  # ALL RANSAC filter failures this event (any theta/range)
    _ransac_ex_bin_tracks = []  # RANSAC tracks that actually land in the target (theta_CM, E*) bin
    _ransac_ex_all_passing = [] # ALL passing RANSAC tracks with their (tcm, ex)
    _gmm_ex_bin_tracks    = []  # GMM+resp tracks that land in the target bin
    _run_this = None
    _ransac_n_grp = 0
    _reg_n_grp = 0

    # --- RANSAC ---
    n_grp = chain.ransac_n_vtx_groups
    _ransac_n_grp = n_grp
    vtx_mult = chain.ransac_vtx_mult
    if _single_group and n_grp != 1:
        pass  # skip RANSAC for this event
    else:
      for g in range(n_grp):
        if vtx_mult[g] != 1:
            # Record tracks in this skipped group so they appear in diagnostics
            for _t_sk in range(int(vtx_mult[g])):
                _th_sk = chain.ransac_theta[g][_t_sk]
                _rn_sk = chain.ransac_range_mm[g][_t_sk]
                _ph_sk = chain.ransac_phi[g][_t_sk]
                if _th_sk == _NAN or _rn_sk == _NAN:
                    continue
                _ransac_all_fails.append({
                    'reason': f"vtx_mult={int(vtx_mult[g])} (multi-track group, entire group skipped)",
                    'theta': _th_sk, 'rng': _rn_sk,
                    'vy': _NAN, 'ey': _NAN, 'phi': _ph_sk,
                })
            continue
        for t in range(vtx_mult[g]):
            theta = chain.ransac_theta[g][t]
            rng   = chain.ransac_range_mm[g][t]
            phi   = chain.ransac_phi[g][t]
            vx    = chain.ransac_vertex_x[g][t]
            vy    = chain.ransac_vertex_y[g][t]
            vz    = chain.ransac_vertex_z[g][t]
            sx = chain.ransac_start_x[g][t]
            sy = chain.ransac_start_y[g][t]
            sz = chain.ransac_start_z[g][t]
            ex = chain.ransac_end_x[g][t]
            ey = chain.ransac_end_y[g][t]
            ez = chain.ransac_end_z[g][t]
            if theta == _NAN or rng == _NAN:
                continue
            # Record ALL reconstructed tracks before any filter (for diagnostics)
            _ransac_all_tracks.append({'theta': theta, 'rng': rng, 'phi': phi})
            if not _passes_filters(vx, vy, vz, sx, sy, sz, ex, ey, ez, phi):
                # Determine which filter killed it
                if not _in_volume(vx, vy, vz):
                    _rfail = f"vtx_outside_volume vtx=({vx:.1f},{vy:.1f},{vz:.1f})"
                elif not (_BEAM_Y_MIN <= vy <= _BEAM_Y_MAX):
                    _rfail = f"vtx_not_in_beam_zone vy={vy:.1f} beam=[{_BEAM_Y_MIN:.1f},{_BEAM_Y_MAX:.1f}]"
                elif not _in_volume(sx, sy, sz):
                    _rfail = f"start_outside_volume start=({sx:.1f},{sy:.1f},{sz:.1f})"
                elif not _in_volume(ex, ey, ez):
                    _rfail = f"end_outside_volume end=({ex:.1f},{ey:.1f},{ez:.1f})"
                elif VolumeBoundaries.BEAM_ZONE_MIN.value <= ey <= VolumeBoundaries.BEAM_ZONE_MAX.value:
                    _rfail = f"end_in_beam_zone ey={ey:.2f} zone=[{VolumeBoundaries.BEAM_ZONE_MIN.value},{VolumeBoundaries.BEAM_ZONE_MAX.value}]"
                elif phi == _NAN:
                    _rfail = "phi_is_NAN"
                elif (_PHI_EXCL_LO <= phi <= _PHI_EXCL_HI) or (-_PHI_EXCL_HI <= phi <= -_PHI_EXCL_LO):
                    _rfail = f"phi_exclusion_zone phi={phi:.1f}"
                else:
                    _rfail = "unknown"
                _fail_rec = {'reason': _rfail, 'theta': theta, 'rng': rng,
                             'vy': vy, 'ey': ey, 'phi': phi}
                _ransac_all_fails.append(_fail_rec)
                # keep the old 90-deg/20mm summary as well
                if (_RANSAC_90_THETA_LO <= theta <= _RANSAC_90_THETA_HI and
                        _RANSAC_90_RNG_LO <= rng <= _RANSAC_90_RNG_HI):
                    _ransac_90_fail_reasons.append(_fail_rec)
                continue

            cp_y = list(chain.ransac_cp_y[g][t])
            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0 and (_use_cutg is False or _cutg.IsInside(track_len, total_charge)):
                ransac_range_angle[0].append(theta)
                ransac_range_angle[1].append(rng)
                ransac_charge_length[0].append(track_len)
                ransac_charge_length[1].append(total_charge)
                ransac_charge_length[2].append(theta)
                _e_kr = chain.ransac_energy_keV[g][t]
                if _e_kr != _NAN and _e_kr > 0:
                    ransac_theta_energy[0].append(theta)
                    ransac_theta_energy[1].append(_e_kr / 1000.0)
                    _tcm_r, _ex_r = _lab_to_ex(theta, _e_kr / 1000.0)
                    if _tcm_r is not None:
                        ransac_excitation[0].append(_tcm_r)
                        ransac_excitation[1].append(_ex_r)
                        _ransac_ex_all_passing.append({'theta': theta, 'rng': rng,
                                                       'tcm': _tcm_r, 'ex': _ex_r})
                        if (_EX_BIN_TCM_LO <= _tcm_r <= _EX_BIN_TCM_HI and
                                _EX_BIN_EX_LO <= _ex_r <= _EX_BIN_EX_HI):
                            _ransac_ex_bin_tracks.append({'theta': theta, 'rng': rng,
                                                          'tcm': _tcm_r, 'ex': _ex_r})
                    else:
                        _ransac_all_fails.append({
                            'reason': f"energy_unphysical E={_e_kr/1000.0:.3f}MeV theta={theta:.1f}",
                            'theta': theta, 'rng': rng,
                            'vy': vy, 'ey': ey, 'phi': phi,
                        })
                else:
                    _ransac_all_fails.append({
                        'reason': f"no_energy (energy_keV={_e_kr})",
                        'theta': theta, 'rng': rng,
                        'vy': vy, 'ey': ey, 'phi': phi,
                    })
            else:
                if total_charge <= 0:
                    _ransac_all_fails.append({
                        'reason': "zero_charge",
                        'theta': theta, 'rng': rng,
                        'vy': vy, 'ey': ey, 'phi': phi,
                    })
                else:
                    _ransac_all_fails.append({
                        'reason': f"failed_cutg len={track_len:.1f} q={total_charge:.1f}",
                        'theta': theta, 'rng': rng,
                        'vy': vy, 'ey': ey, 'phi': phi,
                    })
                if 88.0 <= theta <= 90.0:
                    _ransac_passed_88_90.append(theta)
                if 25.0 <= theta <= 50.0 and 25.0 <= rng <= 50.0:
                    _ransac_in_low_angle.append({'theta': theta, 'rng': rng, 'phi': phi,
                                                  'vtx': (vx, vy, vz)})
                _ransac_passed = True
                if _WIN1_LO <= theta <= _WIN1_HI:
                    _ransac_win1_passing.append(theta)
                if _WIN2_LO <= theta <= _WIN2_HI:
                    _ransac_win2_passing.append(theta)
                if _WIN3_THETA_LO <= theta <= _WIN3_THETA_HI and _WIN3_RNG_LO <= rng <= _WIN3_RNG_HI:
                    _ransac_win3_passing.append({'theta': theta, 'rng': rng})
                _dx = ex - sx; _dy = ey - sy; _dz = ez - sz
                _dn = math.sqrt(_dx*_dx + _dy*_dy + _dz*_dz)
                _dunit = (_dx/_dn, _dy/_dn, _dz/_dn) if _dn > 1e-9 else (0.0, 0.0, 0.0)
                _ransac_passing_tracks.append({
                    'theta': theta, 'phi': phi,
                    'vtx': (vx, vy, vz),
                    'dir': _dunit,
                    'start': (sx, sy, sz), 'end': (ex, ey, ez),
                })

    # --- REG scatter ---
    n_grp = chain.reg_n_vtx_groups
    _reg_n_grp = n_grp
    vtx_mult = chain.reg_vtx_mult
    if _single_group and n_grp != 1:
        if _ransac_passed:
            _ransac_str_mv = "  |  ".join(
                f"theta={r['theta']:.1f} phi={r['phi']:.1f} "
                f"vtx=({r['vtx'][0]:.1f},{r['vtx'][1]:.1f},{r['vtx'][2]:.1f})"
                for r in _ransac_passing_tracks
            ) if _ransac_passing_tracks else "(none)"
            _gmm_rejected["multi_vtx_group"].append(
                (event_id, f"reg_n_vtx_groups={n_grp}", _ransac_str_mv))
        pass  # skip REG for this event
    else:
      for g in range(n_grp):
        if vtx_mult[g] != 1:
            continue
        for t in range(vtx_mult[g]):
            theta = chain.reg_theta[g][t]
            rng   = chain.reg_range_mm[g][t]
            phi   = chain.reg_phi[g][t]
            vx    = chain.reg_vertex_x[g][t]
            vy    = chain.reg_vertex_y[g][t]
            vz    = chain.reg_vertex_z[g][t]
            sx = chain.reg_start_x[g][t]
            sy = chain.reg_start_y[g][t]
            sz = chain.reg_start_z[g][t]
            ex = chain.reg_end_x[g][t]
            ey = chain.reg_end_y[g][t]
            ez = chain.reg_end_z[g][t]
            _reg_has_any_track = True
            if theta == _NAN or rng == _NAN:
                _evt_gmm_reasons.append(
                    ("no_theta_or_range", f"theta={theta} range={rng}"))
                continue
            # Check each filter individually and record the first failure reason
            _dx_g = ex - sx; _dy_g = ey - sy; _dz_g = ez - sz
            _dn_g = math.sqrt(_dx_g*_dx_g + _dy_g*_dy_g + _dz_g*_dz_g)
            _gdir = (_dx_g/_dn_g, _dy_g/_dn_g, _dz_g/_dn_g) if _dn_g > 1e-9 else (0.0, 0.0, 0.0)
            if not _in_volume(vx, vy, vz):
                _evt_gmm_reasons.append(
                    ("vertex_outside_volume", {
                        'gmm_vtx': (vx, vy, vz), 'gmm_dir': _gdir,
                        'gmm_theta': theta,
                    }))
                continue
            if not (_BEAM_Y_MIN <= vy <= _BEAM_Y_MAX):
                _evt_gmm_reasons.append(
                    ("vertex_not_in_beam_zone", {
                        'gmm_vtx': (vx, vy, vz), 'gmm_dir': _gdir,
                        'gmm_theta': theta,
                    }))
                continue
            if not _in_volume(sx, sy, sz):
                _evt_gmm_reasons.append(
                    ("start_outside_volume", {'start': (sx, sy, sz), 'gmm_theta': theta}))
                continue
            if not _in_volume(ex, ey, ez):
                _evt_gmm_reasons.append(
                    ("end_outside_volume", {'end': (ex, ey, ez), 'gmm_theta': theta}))
                continue
            if VolumeBoundaries.BEAM_ZONE_MIN.value <= ey <= VolumeBoundaries.BEAM_ZONE_MAX.value:
                _evt_gmm_reasons.append(
                    ("end_in_beam_zone", {'ey': ey, 'gmm_theta': theta}))
                continue
            if phi == _NAN:
                _evt_gmm_reasons.append(("no_phi", {}))
                continue
            if (_PHI_EXCL_LO <= phi <= _PHI_EXCL_HI) or (-_PHI_EXCL_HI <= phi <= -_PHI_EXCL_LO):
                _evt_gmm_reasons.append(
                    ("phi_exclusion_zone", {'phi': phi, 'gmm_theta': theta}))
                continue

            cp_y = list(chain.reg_cp_y[g][t])
            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0 and (_use_cutg is False or _cutg.IsInside(track_len, total_charge)):
                reg_range_angle[0].append(theta)
                reg_range_angle[1].append(rng)
                reg_charge_length[0].append(track_len)
                reg_charge_length[1].append(total_charge)
                reg_charge_length[2].append(theta)
                _e_kg = chain.reg_energy_keV[g][t]
                if _e_kg != _NAN and _e_kg > 0:
                    reg_theta_energy[0].append(theta)
                    reg_theta_energy[1].append(_e_kg / 1000.0)
                    _tcm_g, _ex_g = _lab_to_ex(theta, _e_kg / 1000.0)
                    if _tcm_g is not None:
                        reg_excitation[0].append(_tcm_g)
                        reg_excitation[1].append(_ex_g)
                _reg_passed = True
                if _WIN1_LO <= theta <= _WIN1_HI:
                    _gmm_win1_passing.append({'theta': theta, 'rng': rng})
                if _WIN2_LO <= theta <= _WIN2_HI:
                    _gmm_win2_passing.append({'theta': theta, 'rng': rng})
                if _WIN3_THETA_LO <= theta <= _WIN3_THETA_HI and _WIN3_RNG_LO <= rng <= _WIN3_RNG_HI:
                    _gmm_win3_passing.append({'theta': theta, 'rng': rng})
                if 25.0 <= theta <= 50.0 and 25.0 <= rng <= 50.0:
                    _gmm_low_angle_passing.append({'theta': theta, 'rng': rng, 'phi': phi,
                                                    'vtx': (vx, vy, vz)})
            else:
                if total_charge <= 0:
                    _evt_gmm_reasons.append(
                        ("zero_charge", {'gmm_theta': theta, 'rng': rng}))
                else:
                    _evt_gmm_reasons.append(
                        ("failed_cutg", {'gmm_theta': theta, 'rng': rng,
                                         'len': track_len, 'q': total_charge}))

            # --- REG combined fit ---
            theta2 = chain.reg_theta2[g][t]
            rng_c  = chain.reg_range_comb_mm[g][t]
            phi2   = chain.reg_phi2[g][t]
            vx2    = chain.reg_vertex2_x[g][t]
            vy2    = chain.reg_vertex2_y[g][t]
            vz2    = chain.reg_vertex2_z[g][t]
            if theta2 == _NAN or rng_c == _NAN:
                continue
            if not _passes_filters(vx2, vy2, vz2, sx, sy, sz, ex, ey, ez, phi2):
                continue

            cp_y_c = list(chain.reg_cp_comb_y[g][t])
            total_charge_c = sum(cp_y_c) if len(cp_y_c) > 0 else 0.0
            if total_charge_c > 0 and (_use_cutg is False or _cutg.IsInside(track_len, total_charge_c)):
                reg_comb_range_angle[0].append(theta2)
                reg_comb_range_angle[1].append(rng_c)
                reg_comb_charge_length[0].append(track_len)
                reg_comb_charge_length[1].append(total_charge_c)
                reg_comb_charge_length[2].append(theta2)
                _e_kc = chain.reg_energy_comb_keV[g][t]
                if _e_kc != _NAN and _e_kc > 0:
                    reg_comb_theta_energy[0].append(theta2)
                    reg_comb_theta_energy[1].append(_e_kc / 1000.0)
                    _tcm_c, _ex_c = _lab_to_ex(theta2, _e_kc / 1000.0)
                    if _tcm_c is not None:
                        reg_comb_excitation[0].append(_tcm_c)
                        reg_comb_excitation[1].append(_ex_c)
                        if (_EX_BIN_TCM_LO <= _tcm_c <= _EX_BIN_TCM_HI and
                                _EX_BIN_EX_LO <= _ex_c <= _EX_BIN_EX_HI):
                            _gmm_ex_bin_tracks.append({'theta': theta2, 'rng': rng_c,
                                                       'tcm': _tcm_c, 'ex': _ex_c})
                if theta2 < 40 and rng_c < 40:
                    _run = int(chain.run_number) if _has_run else -1
                    # print(f"[COMB-LOW] run={_run} event={event_id} theta2={theta2:.2f} range_comb={rng_c:.2f}")

    # --- Accumulate RANSAC 90-deg/20mm filter failures ---
    if _ransac_90_fail_reasons:
        _run_90 = int(chain.run_number) if _has_run else -1
        for _fr in _ransac_90_fail_reasons:
            _ransac_90_failures.append({'run': _run_90, 'event': event_id,
                                        'reason': _fr['reason'], 'theta': _fr['theta'],
                                        'rng': _fr['rng'], 'vy': _fr['vy'],
                                        'ey': _fr['ey'], 'phi': _fr['phi']})

    # --- Accumulate excitation-bin (E*=0, theta_CM=1.5 deg): GMM in bin, RANSAC not ---
    if _gmm_ex_bin_tracks and not _ransac_ex_bin_tracks:
        _run_eb = int(chain.run_number) if _has_run else -1
        _ex_bin_events.append({
            'run': _run_eb, 'event': event_id,
            'gmm_tracks':      list(_gmm_ex_bin_tracks),
            'ransac_all_fails': list(_ransac_all_fails),
            'ransac_ex_passing': list(_ransac_ex_all_passing),
            'n_grp_ransac':    _ransac_n_grp,
        })

    # --- Accumulate GMM-LOW events (GMM passed [25,50] deg / [25,50]mm, RANSAC did not) ---
    if _gmm_low_angle_passing and not _ransac_in_low_angle:
        _run_this = int(chain.run_number) if _has_run else -1
        _gmm_low_angle_events.append({
            'run': _run_this,
            'event': event_id,
            'gmm_tracks': list(_gmm_low_angle_passing),
            'ransac_tracks': list(_ransac_passing_tracks),
            'ransac_all_tracks': list(_ransac_all_tracks),
        })

    # --- Accumulate window-specific: GMM passed window but RANSAC did not ---
    if _gmm_win1_passing and not _ransac_win1_passing:
        _run_this = int(chain.run_number) if _has_run else -1
        _gmm_win1_events.append({
            'run': _run_this, 'event': event_id,
            'gmm_tracks': list(_gmm_win1_passing),
            'ransac_passing': list(_ransac_passing_tracks),
            'ransac_all_in_win': [r for r in _ransac_all_tracks if _WIN1_LO <= r['theta'] <= _WIN1_HI],
        })
    if _gmm_win2_passing and not _ransac_win2_passing:
        _run_this = int(chain.run_number) if _has_run else -1
        _gmm_win2_events.append({
            'run': _run_this, 'event': event_id,
            'gmm_tracks': list(_gmm_win2_passing),
            'ransac_passing': list(_ransac_passing_tracks),
            'ransac_all_in_win': [r for r in _ransac_all_tracks if _WIN2_LO <= r['theta'] <= _WIN2_HI],
        })
    if _gmm_win3_passing and not _ransac_win3_passing:
        _run_this = int(chain.run_number) if _has_run else -1
        _gmm_win3_events.append({
            'run': _run_this, 'event': event_id,
            'gmm_tracks': list(_gmm_win3_passing),
            'ransac_passing': list(_ransac_passing_tracks),
            # pre-filter RANSAC candidates inside the theta+range box
            'ransac_in_box': [r for r in _ransac_all_tracks
                              if _WIN3_THETA_LO <= r['theta'] <= _WIN3_THETA_HI
                              and _WIN3_RNG_LO   <= r['rng']   <= _WIN3_RNG_HI],
            # pre-filter RANSAC candidates that are only in the theta window (any range)
            'ransac_theta_only': [r for r in _ransac_all_tracks
                                  if _WIN3_THETA_LO <= r['theta'] <= _WIN3_THETA_HI],
        })

    # --- Diagnostic: RANSAC passed (88-90°) but GMM did not ---
    # if _ransac_passed_88_90 and not _reg_passed:
    #     for th in _ransac_passed_88_90:
    #         print(f"[DIAG] event_id={event_id}  RANSAC passed (theta={th:.2f}°)  GMM did not pass")
    # Commit per-event GMM rejection reasons only for events where RANSAC passed but GMM did not
    if _ransac_passed and not _reg_passed:
        # ── build compact RANSAC passing-track string (reused in every entry) ──
        _ransac_str = "  |  ".join(
            f"theta={r['theta']:.1f} phi={r['phi']:.1f} "
            f"vtx=({r['vtx'][0]:.1f},{r['vtx'][1]:.1f},{r['vtx'][2]:.1f}) "
            f"dir=({r['dir'][0]:.3f},{r['dir'][1]:.3f},{r['dir'][2]:.3f})"
            for r in _ransac_passing_tracks
        ) if _ransac_passing_tracks else "(none)"

        if not _reg_has_any_track:
            _gmm_rejected["no_gmm_scatter_track"].append(
                (event_id,
                 f"reg_groups={_reg_n_grp} ransac_groups={_ransac_n_grp}",
                 _ransac_str))
        for _reason, _detail in _evt_gmm_reasons:
            if _reason in ("vertex_outside_volume", "vertex_not_in_beam_zone"):
                _gv = _detail['gmm_vtx']; _gd = _detail['gmm_dir']
                _gt = _detail['gmm_theta']
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_gt:.1f} vtx=({_gv[0]:.1f},{_gv[1]:.1f},{_gv[2]:.1f}) "
                    f"dir=({_gd[0]:.3f},{_gd[1]:.3f},{_gd[2]:.3f})",
                    _ransac_str))
            elif _reason == "failed_cutg":
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f} rng={_detail['rng']:.1f} "
                    f"len={_detail.get('len',0):.1f} q={_detail.get('q',0):.1f}",
                    _ransac_str))
            elif _reason == "start_outside_volume":
                _sv = _detail['start']
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f} start=({_sv[0]:.1f},{_sv[1]:.1f},{_sv[2]:.1f})",
                    _ransac_str))
            elif _reason == "end_outside_volume":
                _ev2 = _detail['end']
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f} end=({_ev2[0]:.1f},{_ev2[1]:.1f},{_ev2[2]:.1f})",
                    _ransac_str))
            elif _reason == "end_in_beam_zone":
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f} ey={_detail['ey']:.1f}",
                    _ransac_str))
            elif _reason == "phi_exclusion_zone":
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f} phi={_detail['phi']:.1f}",
                    _ransac_str))
            elif _reason == "zero_charge":
                _gmm_rejected[_reason].append((
                    event_id,
                    f"GMM: theta={_detail['gmm_theta']:.1f}",
                    _ransac_str))
            else:
                _gmm_rejected[_reason].append((event_id, str(_detail), _ransac_str))


# ── GMM Low-Angle Exclusive Summary ─────────────────────────────────────────
def _classify_ransac_theta(gmm_theta, ransac_theta, ransac_rng,
                            window_lo=25.0, window_hi=50.0, near_deg=10.0, far_deg=70.0):
    """Classify why a RANSAC track missed the GMM low-angle window."""
    if window_lo <= ransac_theta <= window_hi:
        if window_lo <= ransac_rng <= window_hi:
            return 'IN_WINDOW'          # both pass — shouldn't appear here
        delta_rng = ransac_rng - window_hi if ransac_rng > window_hi else window_lo - ransac_rng
        return f'RIGHT_ANGLE_WRONG_RANGE(rng={ransac_rng:.1f},Δ={delta_rng:.1f}mm)'
    delta_lo = window_lo - ransac_theta  # >0 means theta too small
    delta_hi = ransac_theta - window_hi  # >0 means theta too large
    outside_by = max(delta_lo, delta_hi, 0)
    if outside_by <= near_deg:
        side = 'low' if delta_lo > 0 else 'high'
        return f'NEAR_MISS_{side.upper()}(theta={ransac_theta:.1f},Δ={outside_by:.1f}°)'
    if ransac_theta > far_deg:
        return f'LARGE_ANGLE(theta={ransac_theta:.1f})'
    return f'WRONG_ANGLE(theta={ransac_theta:.1f},Δ={outside_by:.1f}°)'

if _gmm_low_angle_events:
    print("\n" + "="*80)
    print("GMM LOW-ANGLE EXCLUSIVE EVENTS  (theta 25-50°, range 25-50 mm)")
    print("GMM passed all filters; RANSAC had no track in that window")
    print("Category key:")
    print("  NEAR_MISS_LOW/HIGH  : RANSAC theta within 10° of window boundary")
    print("  RIGHT_ANGLE_WRONG_RANGE : RANSAC theta OK but range outside [25,50] mm")
    print("  LARGE_ANGLE         : RANSAC theta > 70° (reconstructed different track)")
    print("  WRONG_ANGLE         : RANSAC theta >10° outside window")
    print("  NO_RANSAC_TRACK     : no RANSAC track reconstructed in this event")
    print("="*80)
    _cat_counts = defaultdict(int)
    for _rec in _gmm_low_angle_events:
        for _gt in _rec['gmm_tracks']:
            _passing_str = "  |  ".join(
                f"theta={_rt['theta']:.1f} phi={_rt['phi']:.1f}"
                for _rt in _rec['ransac_tracks']
            ) if _rec['ransac_tracks'] else ""
            _all = _rec.get('ransac_all_tracks', [])
            if not _all:
                _cat = 'NO_RANSAC_TRACK'
                _detail = '(none reconstructed)'
            else:
                # Pick the RANSAC track whose theta is closest to GMM theta
                _best = min(_all, key=lambda r: abs(r['theta'] - _gt['theta']))
                _cat = _classify_ransac_theta(_gt['theta'], _best['theta'], _best['rng'])
                _detail = '  |  '.join(
                    f"theta={r['theta']:.1f} rng={r['rng']:.1f}"
                    for r in _all
                )
            _cat_base = _cat.split('(')[0]
            _cat_counts[_cat_base] += 1
            print(f"  run={_rec['run']}  event={_rec['event']}")
            print(f"    GMM   : theta={_gt['theta']:.1f}°  range={_gt['rng']:.1f}mm  phi={_gt['phi']:.1f}°")
            print(f"    RANSAC: {_cat}")
            if _detail:
                print(f"      all RANSAC tracks: {_detail}")
            if _passing_str:
                print(f"      passing filter  : {_passing_str}")
    print()
    print(f"  Total: {len(_gmm_low_angle_events)} events")
    print("  Category summary:")
    for _cat, _n in sorted(_cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {_cat:<35} {_n}")
    print("="*80 + "\n")

# ── GMM Rejection Summary ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("GMM REJECTION SUMMARY")
print("="*70)
_reason_labels = {
    "no_gmm_scatter_track":    "RANSAC passed but GMM found no scatter track",
    "multi_vtx_group":         "Event skipped: >1 vertex group (single_group filter)",
    "no_theta_or_range":       "theta or range is NaN (-999)",
    "vertex_outside_volume":   "Vertex outside active volume",
    "vertex_not_in_beam_zone": "Vertex Y not in beam zone",
    "start_outside_volume":    "Track start outside active volume",
    "end_outside_volume":      "Track end outside active volume",
    "end_in_beam_zone":        "Track end Y inside beam zone",
    "no_phi":                  "phi is NaN (-999)",
    "phi_exclusion_zone":      "phi in exclusion zone [70,110] or [-110,-70] deg",
    "zero_charge":             "Zero charge after beam-zone exclusion",
    "failed_cutg":             "Track outside TCutG (charge/length gate)",
}
for _reason, _entries in sorted(_gmm_rejected.items(), key=lambda x: -len(x[1])):
    _label = _reason_labels.get(_reason, _reason)
    print(f"\n[{_label}]  ({len(_entries)} occurrences)")
    for _eid, _gmm_det, _rans_det in _entries[:20]:
        print(f"  event={_eid}")
        print(f"    GMM   : {_gmm_det}")
        print(f"    RANSAC: {_rans_det}")
    if len(_entries) > 20:
        print(f"  ... and {len(_entries)-20} more")
print("="*70 + "\n")


# ── Window Summary: GMM reconstructed but RANSAC failed ─────────────────────
def _print_window_summary(events, win_lo, win_hi, label):
    print(f"\n{'='*80}")
    print(f"GMM RECONSTRUCTED, RANSAC FAILED  —  theta [{win_lo}°, {win_hi}°]  ({label})")
    print(f"{'='*80}")
    print(f"  Total events: {len(events)}")
    for rec in events[:50]:
        ransac_str = '  |  '.join(
            f"theta={r['theta']:.1f} rng={r.get('rng', 0):.1f}"
            for r in rec.get('ransac_all_in_win', [])
        ) or '(none in window)'
        passing_str = '  |  '.join(
            f"theta={r['theta']:.1f} phi={r['phi']:.1f}"
            for r in rec.get('ransac_passing', [])
        ) or '(none passed)'
        print(f"  run={rec['run']}  event={rec['event']}")
        for gt in rec['gmm_tracks']:
            print(f"    GMM  : theta={gt['theta']:.1f}°  range={gt['rng']:.1f}mm")
        print(f"    RANSAC in window  : {ransac_str}")
        print(f"    RANSAC passing    : {passing_str}")
    if len(events) > 50:
        print(f"  ... and {len(events)-50} more")
    print(f"{'='*80}\n")

_print_window_summary(_gmm_win1_events, _WIN1_LO, _WIN1_HI, "near 90°")
_print_window_summary(_gmm_win2_events, _WIN2_LO, _WIN2_HI, "low angle")

# ── RANSAC 90-deg/20mm filter-failure breakdown ──────────────────────────────
print(f"\n{'='*80}")
print(f"RANSAC FILTER FAILURES  —  theta [{_RANSAC_90_THETA_LO},{_RANSAC_90_THETA_HI}] deg  AND  range [{_RANSAC_90_RNG_LO},{_RANSAC_90_RNG_HI}] mm")
print(f"  (tracks reconstructed by RANSAC in this window that were rejected by a filter)")
print(f"{'='*80}")
print(f"  Total rejected tracks in window: {len(_ransac_90_failures)}")
if _ransac_90_failures:
    from collections import Counter
    _reason_counts = Counter(r['reason'].split()[0] for r in _ransac_90_failures)
    print("  Failure reason counts (first keyword):")
    for _rk, _rc in _reason_counts.most_common():
        print(f"    {_rc:5d}  {_rk}")
    print()
    print("  First 40 individual failed tracks:")
    for _rf in _ransac_90_failures[:40]:
        print(f"    run={_rf['run']}  evt={_rf['event']}  "
              f"theta={_rf['theta']:.1f}  rng={_rf['rng']:.1f}  "
              f"vy={_rf['vy']:.2f}  ey={_rf['ey']:.2f}  phi={_rf['phi']:.1f}"
              f"  =>  {_rf['reason']}")
    if len(_ransac_90_failures) > 40:
        print(f"  ... and {len(_ransac_90_failures)-40} more")
print(f"{'='*80}\n")

# ── Excitation-bin diagnostic: E*=0 MeV, theta_CM=1.5 deg ───────────────────
print(f"\n{'='*80}")
print(f"EXCITATION BIN DIAGNOSTIC  —  E* [{_EX_BIN_EX_LO},{_EX_BIN_EX_HI}] MeV  x  "
      f"theta_CM [{_EX_BIN_TCM_LO},{_EX_BIN_TCM_HI}] deg")
print(f"  Events where GMM+resp filled this bin and RANSAC did not")
print(f"{'='*80}")
print(f"  Total such events: {len(_ex_bin_events)}")
if _ex_bin_events:
    from collections import Counter as _Counter
    _reason_ctr   = _Counter()
    _elsewhere_ctr = _Counter()
    _no_cand_ctr  = 0
    for _eb in _ex_bin_events:
        if _eb['ransac_ex_passing']:
            # Case B: RANSAC passed but landed elsewhere
            for _rp in _eb['ransac_ex_passing']:
                _tcm_b = round(_rp['tcm'] / 0.3) * 0.3
                _ex_b  = round(_rp['ex']  / 0.75) * 0.75
                _elsewhere_ctr[f"tcm~{_tcm_b:.1f} ex~{_ex_b:.2f}"] += 1
        elif _eb['ransac_all_fails']:
            # Case A: RANSAC had tracks but all failed a filter
            for _rf in _eb['ransac_all_fails']:
                _reason_ctr[_rf['reason'].split()[0]] += 1
        else:
            _no_cand_ctr += 1
    print(f"\n  Case A — RANSAC had tracks but ALL failed a filter ({sum(_reason_ctr.values())} tracks):")
    for _rk, _rc in _reason_ctr.most_common():
        print(f"    {_rc:5d}  {_rk}")
    print(f"\n  Case B — RANSAC passed filters but landed in a DIFFERENT bin:")
    for _bk, _bc in _elsewhere_ctr.most_common(15):
        print(f"    {_bc:5d}  {_bk}")
    print(f"\n  Case C — RANSAC produced no tracks at all: {_no_cand_ctr}")
    print()
    print(f"  First 40 events detail:")
    for _eb in _ex_bin_events[:40]:
        _gmm_s = '  |  '.join(
            f"tcm={g['tcm']:.2f} ex={g['ex']:.3f} (lab theta={g['theta']:.1f} rng={g['rng']:.1f})"
            for g in _eb['gmm_tracks'])
        print(f"  run={_eb['run']}  evt={_eb['event']}  n_grp_ransac={_eb['n_grp_ransac']}")
        print(f"    GMM+resp : {_gmm_s}")
        if _eb['ransac_ex_passing']:
            print(f"    RANSAC [B] passed filters, landed in:")
            for _rp in _eb['ransac_ex_passing']:
                print(f"      tcm={_rp['tcm']:.2f} ex={_rp['ex']:.3f}"
                      f" (lab theta={_rp['theta']:.1f} rng={_rp['rng']:.1f})")
        elif _eb['ransac_all_fails']:
            print(f"    RANSAC [A] all tracks rejected:")
            for _rf in _eb['ransac_all_fails']:
                print(f"      theta={_rf['theta']:.1f} rng={_rf['rng']:.1f}"
                      f" ey={_rf['ey']:.3f} phi={_rf['phi']:.1f}  =>  {_rf['reason']}")
        else:
            print(f"    RANSAC [C] no tracks produced")
    if len(_ex_bin_events) > 40:
        print(f"  ... and {len(_ex_bin_events)-40} more")
print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"GMM RECONSTRUCTED, RANSAC FAILED  —  theta [{_WIN3_THETA_LO}°,{_WIN3_THETA_HI}°]  AND  range [{_WIN3_RNG_LO},{_WIN3_RNG_HI}] mm")
print(f"  Condition: GMM passed all filters+TCutG in box; RANSAC had no passing track in box")
print(f"{'='*80}")
print(f"  Total events: {len(_gmm_win3_events)}")
for _rec in _gmm_win3_events:
    _in_box   = _rec.get('ransac_in_box', [])
    _th_only  = _rec.get('ransac_theta_only', [])
    _rpassing = _rec.get('ransac_passing', [])
    print(f"  run={_rec['run']}  event={_rec['event']}")
    for _gt in _rec['gmm_tracks']:
        print(f"    GMM  : theta={_gt['theta']:.2f}°  range={_gt['rng']:.2f}mm")
    if _in_box:
        # RANSAC had a candidate in the box but it failed one or more filters
        _box_str = '  |  '.join(f"theta={r['theta']:.2f} rng={r['rng']:.2f}" for r in _in_box)
        print(f"    RANSAC in box (pre-filter, FAILED filters) : {_box_str}")
    elif _th_only:
        # RANSAC had a track in the theta window but wrong range
        _th_str = '  |  '.join(f"theta={r['theta']:.2f} rng={r['rng']:.2f}" for r in _th_only)
        print(f"    RANSAC in theta window, wrong range (pre-filter) : {_th_str}")
    else:
        print(f"    RANSAC: no candidate reconstructed in theta[{_WIN3_THETA_LO},{_WIN3_THETA_HI}]")
    if _rpassing:
        _rp_str = '  |  '.join(f"theta={r['theta']:.2f} phi={r['phi']:.2f}" for r in _rpassing)
        print(f"    RANSAC passing (other tracks)              : {_rp_str}")
if len(_gmm_win3_events) > 50:
    print(f"  ... (showing first 50 of {len(_gmm_win3_events)})")
print(f"{'='*80}\n")


_THETA_LO, _THETA_HI = 85, 95   # projection slice in degrees

# --- Canvas: 4 rows x 4 cols ---
c = root.TCanvas("c_kin", "Kinematics", 2400, 2000)
c.Divide(4, 4)

datasets = [
    ("RANSAC",        ransac_range_angle, ransac_charge_length),
    ("GMM_REG",       reg_range_angle,    reg_charge_length),
    ("GMM_REG comb",  reg_comb_range_angle, reg_comb_charge_length),
]

keep = []  # prevent ROOT garbage collection
for row, (label, ra_data, cl_data) in enumerate(datasets):
    # Col 1: Range vs Lab Angle
    c.cd(4 * row + 1)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    h_ra = root.TH2F(
        f"h_ra_{row}", f"{label}: Range vs Lab Angle;#theta (deg);Range (mm)",
        90, 0, 180, 80, 0, 400,
    )
    h_ra.SetDirectory(0)
    for th, rng in zip(ra_data[0], ra_data[1]):
        h_ra.Fill(th, rng)
    h_ra.Draw("COLZ")
    keep.append(h_ra)

    # Col 2: Total Charge vs Track Length
    c.cd(4 * row + 2)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    h_cl = root.TH2F(
        f"h_cl_{row}", f"{label}: Total Charge vs Track Length;Track Length (mm);Total Charge (a.u.)",
        200, 0, 400, 200, 0, 50000,
    )
    h_cl.SetDirectory(0)
    for tl, tc in zip(cl_data[0], cl_data[1]):
        h_cl.Fill(tl, tc)
    h_cl.Draw("COLZ")
    _cutg.Draw("same")
    keep.append(h_cl)

    # Col 3: Theta distribution restricted to [THETA_LO, THETA_HI]
    c.cd(4 * row + 3)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    h_proj_ra = root.TH1F(
        f"h_proj_ra_{row}",
        f"{label}: Counts ({_THETA_LO}#minus{_THETA_HI}#circ);#theta (deg);Counts",
        20, _THETA_LO, _THETA_HI,
    )
    h_proj_ra.SetDirectory(0)
    for th in ra_data[0]:
        if _THETA_LO <= th <= _THETA_HI:
            h_proj_ra.Fill(th)
    h_proj_ra.Draw()
    keep.append(h_proj_ra)

    # Col 4: Total Charge projection for theta slice
    c.cd(4 * row + 4)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    h_proj_cl = root.TH1F(
        f"h_proj_cl_{row}",
        f"{label}: Charge ({_THETA_LO}#minus{_THETA_HI}#circ);Total Charge (a.u.);Counts",
        500, 0, 50000,
    )
    h_proj_cl.SetDirectory(0)
    for tl, tc, th in zip(cl_data[0], cl_data[1], cl_data[2]):
        if _THETA_LO <= th <= _THETA_HI:
            h_proj_cl.Fill(tc)
    h_proj_cl.Draw()
    keep.append(h_proj_cl)

# --- Row 4: GMM_REG - RANSAC difference ---
# keep indices per row: h_ra=4*r, h_cl=4*r+1, h_proj_ra=4*r+2, h_proj_cl=4*r+3
# RANSAC: 0,1,2,3   REG: 4,5,6,7

# Col 1: Range vs Lab Angle diff
c.cd(13)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_ra = keep[4].Clone("h_diff_ra")
h_diff_ra.SetDirectory(0)
h_diff_ra.SetTitle("GMM_REG #minus RANSAC: Range vs Lab Angle;#theta (deg);Range (mm)")
h_diff_ra.Add(keep[0], -1)
h_diff_ra.Draw("COLZ")
keep.append(h_diff_ra)

# Col 2: Total Charge vs Track Length diff
c.cd(14)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_cl = keep[5].Clone("h_diff_cl")
h_diff_cl.SetDirectory(0)
h_diff_cl.SetTitle("GMM_REG #minus RANSAC: Total Charge vs Track Length;Track Length (mm);Total Charge (a.u.)")
h_diff_cl.Add(keep[1], -1)
h_diff_cl.Draw("COLZ")
keep.append(h_diff_cl)

# Col 3: Range projection diff
c.cd(15)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_proj_ra = keep[6].Clone("h_diff_proj_ra")
h_diff_proj_ra.SetDirectory(0)
h_diff_proj_ra.SetTitle("GMM_REG #minus RANSAC: Range ({0}#minus{1}#circ);Range (mm);Counts".format(_THETA_LO, _THETA_HI))
h_diff_proj_ra.Add(keep[2], -1)
h_diff_proj_ra.Draw()
keep.append(h_diff_proj_ra)

# Col 4: Charge projection diff
c.cd(16)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_proj_cl = keep[7].Clone("h_diff_proj_cl")
h_diff_proj_cl.SetDirectory(0)
h_diff_proj_cl.SetTitle("GMM_REG #minus RANSAC: Charge ({0}#minus{1}#circ);Total Charge (a.u.);Counts".format(_THETA_LO, _THETA_HI))
h_diff_proj_cl.Add(keep[3], -1)
h_diff_proj_cl.Draw()
keep.append(h_diff_proj_cl)

c.Update()
c.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
c.SaveAs(os.path.join(images_dir, "kinematics.png"))

# --- Overlay: theta projections for all three methods ---
c_ov = root.TCanvas("c_overlay", "Theta Projection Comparison", 900, 700)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)

# keep indices: h_proj_ra is at 4*row+2 → RANSAC=2, GMM_REG=6, GMM_REG_comb=10
_ov_colors = [root.kRed, root.kBlue, root.kGreen + 2]
_ov_labels = ["RANSAC", "GMM_REG", "GMM_REG comb"]
_ov_idx    = [2, 6, 10]
_ov_histos = []

for i, (idx, color, label) in enumerate(zip(_ov_idx, _ov_colors, _ov_labels)):
    h = keep[idx].Clone(f"h_overlay_{i}")
    h.SetDirectory(0)
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetTitle(f"Theta Projection ({_THETA_LO}#minus{_THETA_HI}#circ);#theta (deg);Counts")
    h.GetXaxis().SetRangeUser(_THETA_LO, _THETA_HI)
    if i == 0:
        h.Draw("HIST")
    else:
        h.Draw("HIST SAME")
    _ov_histos.append(h)

leg = root.TLegend(0.50, 0.55, 0.88, 0.88)
for h, label in zip(_ov_histos, _ov_labels):
    n_slice = int(h.Integral(h.GetXaxis().FindBin(_THETA_LO), h.GetXaxis().FindBin(_THETA_HI)))
    leg.AddEntry(h, f"{label}  N={n_slice}  #mu={h.GetMean():.1f}  #sigma={h.GetStdDev():.1f}", "l")
leg.SetTextSize(0.03)
leg.Draw()

c_ov.Update()
c_ov.WaitPrimitive()
c_ov.SaveAs(os.path.join(images_dir, "theta_projection_overlay.png"))
c_ov.SaveAs(os.path.join(images_dir, "theta_projection_overlay.root"))

# Save all histograms to ROOT file
root_out_path = os.path.join(images_dir, "kinematics.root")
fout = root.TFile(root_out_path, "RECREATE")
for h in keep:
    h.Write()
for h in _ov_histos:
    h.Write()
fout.Close()

print(f"Points — RANSAC: {len(ransac_range_angle[0])}, REG: {len(reg_range_angle[0])}, REG_comb: {len(reg_comb_range_angle[0])}")
print(f"Saved to {os.path.join(images_dir, 'kinematics.png')} and {root_out_path}")

# Explicit cleanup to avoid ROOT shutdown errors
for h in keep + _ov_histos:
    h.SetDirectory(0)
del keep[:]
del _ov_histos[:]

# ─────────────────────────────────────────────────────────────────────────────
# 4-pad theta-lab vs energy canvas with kinematic lines
# ─────────────────────────────────────────────────────────────────────────────

def _kine_line(T_beam_MeV, Ex_MeV, n_pts=1800):
    """
    58Ni(4He,4He)58Ni* — inverse kinematics.
    Returns (theta_lab_deg[], T_alpha_MeV[]) for the full CM range 0-180°.
    Beam = 58Ni at T_beam_MeV, target = 4He at rest, ejectile = 4He.
    """
    import math as _math
    U        = 931.5
    m_beam   = 57.93  * U        # 58Ni
    m_tgt    = 4.002  * U        # 4He target
    m_light  = 4.002  * U        # 4He ejectile
    m_heavy  = 57.93  * U + Ex_MeV  # 58Ni recoil + excitation

    E_beam  = m_beam + T_beam_MeV
    p_beam  = _math.sqrt(E_beam**2 - m_beam**2)
    sqrt_s  = _math.sqrt(m_beam**2 + m_tgt**2 + 2.0 * E_beam * m_tgt)
    s       = sqrt_s ** 2

    if sqrt_s < m_light + m_heavy:
        return [], []   # not kinematically accessible

    beta_cm  = p_beam / (E_beam + m_tgt)
    gamma_cm = (E_beam + m_tgt) / sqrt_s

    E_light_cm = (s + m_light**2 - m_heavy**2) / (2.0 * sqrt_s)
    p_light_cm = _math.sqrt(max(0.0, E_light_cm**2 - m_light**2))

    thetas, energies = [], []
    for i in range(n_pts + 1):
        theta_cm = _math.pi * i / n_pts
        cos_cm   = _math.cos(theta_cm)
        sin_cm   = _math.sin(theta_cm)
        p_par_lab = gamma_cm * (p_light_cm * cos_cm + beta_cm * E_light_cm)
        p_perp    = p_light_cm * sin_cm
        E_lab     = gamma_cm * (E_light_cm + beta_cm * p_light_cm * cos_cm)
        T_lab     = E_lab - m_light
        if T_lab < 0:
            continue
        theta_lab = _math.degrees(_math.atan2(p_perp, p_par_lab))
        if theta_lab < 0:
            theta_lab += 180.0
        thetas.append(theta_lab)
        energies.append(T_lab)
    return thetas, energies


_EX_LIST  = [0, 5, 10, 15, 20, 25, 30]  # excitation energies in MeV
_KINE_COLORS = [
    root.kBlack, root.kRed, root.kBlue, root.kGreen + 2,
    root.kMagenta, root.kCyan + 1, root.kOrange + 1,
]

# Pre-compute kinematic lines once
_kine_graphs = []
_kine_keep   = []  # prevent garbage collection
for _ex, _col in zip(_EX_LIST, _KINE_COLORS):
    _kx, _ky = _kine_line(_T_BEAM, _ex)
    if not _kx:
        continue
    _kg = root.TGraph(len(_kx), array('d', _kx), array('d', _ky))
    _kg.SetLineColor(_col)
    _kg.SetLineWidth(2)
    _kg.SetLineStyle(1 if _ex == 0 else 2)
    _kine_graphs.append((_ex, _kg))
    _kine_keep.append(_kg)


def _draw_kine_legend(pad):
    """Draw a small legend for the kinematic lines on the current pad."""
    leg_k = root.TLegend(0.52, 0.55, 0.88, 0.88)
    leg_k.SetTextSize(0.03)
    for _ex, _kg in _kine_graphs:
        leg_k.AddEntry(_kg, f"Ex={_ex} MeV", "l")
    leg_k.Draw()
    _kine_keep.append(leg_k)


# Horizontal reference lines at each known excitation energy (for E* vs theta_CM canvas)
_hline_keep = []
_hlines = []
for _ex_val, _col in zip(_EX_LIST, _KINE_COLORS):
    _hl = root.TLine(0.0, float(_ex_val), 6.0, float(_ex_val))
    _hl.SetLineColor(_col)
    _hl.SetLineWidth(2)
    _hl.SetLineStyle(1 if _ex_val == 0 else 2)
    _hlines.append((_ex_val, _hl))
    _hline_keep.append(_hl)

def _draw_ex_legend():
    leg_ex = root.TLegend(0.52, 0.55, 0.88, 0.88)
    leg_ex.SetTextSize(0.03)
    for _ev, _hl in _hlines:
        leg_ex.AddEntry(_hl, f"E*={_ev} MeV", "l")
    leg_ex.Draw()
    _hline_keep.append(leg_ex)

_EX_RANGE_LO, _EX_RANGE_HI = -3.0, 33.0   # MeV range for y-axis (48 bins × 0.75 MeV)

c_te = root.TCanvas("c_te", "Excitation Energy vs CM Angle", 2000, 1600)
c_te.Divide(2, 2)

_ex_datasets = [
    ("RANSAC",       ransac_excitation),
    ("GMM (scat)",   reg_excitation),
    ("GMM+resp",     reg_comb_excitation),
]
_te_keep = []

for _pad, (_lbl, _ex_data) in enumerate(_ex_datasets, start=1):
    c_te.cd(_pad)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _h = root.TH2F(
        f"h_ex_{_pad}",
        f"{_lbl}: E* vs #theta_{{CM}};#theta_{{CM}} (deg);E* (MeV)",
        20, 0, 6, 48, _EX_RANGE_LO, _EX_RANGE_HI,
    )
    _h.SetDirectory(0)
    for _tc, _ev in zip(_ex_data[0], _ex_data[1]):
        if _EX_RANGE_LO < _ev < _EX_RANGE_HI:
            _h.Fill(_tc, _ev)
    _h.Draw("COLZ")
    _te_keep.append(_h)
    for _ev, _hl in _hlines:
        _hl.Draw("SAME")
    _draw_ex_legend()

# Pad 4: GMM+resp minus RANSAC
c_te.cd(4)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
_h_diff_te = _te_keep[2].Clone("h_ex_diff")
_h_diff_te.SetDirectory(0)
_h_diff_te.SetTitle("GMM+resp #minus RANSAC: E* vs #theta_{CM};#theta_{CM} (deg);E* (MeV)")
_h_diff_te.Add(_te_keep[0], -1)
_h_diff_te.Draw("COLZ")
_te_keep.append(_h_diff_te)
for _ev, _hl in _hlines:
    _hl.Draw("SAME")
_draw_ex_legend()

c_te.Update()
c_te.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
c_te.SaveAs(os.path.join(images_dir, "kinematics_compare_ex_cm.png"))
c_te.SaveAs(os.path.join(images_dir, "kinematics_compare_ex_cm.root"))

print(f"excitation-energy points — RANSAC: {len(ransac_excitation[0])},"
      f" GMM: {len(reg_excitation[0])}, GMM+resp: {len(reg_comb_excitation[0])}")

for _h in _te_keep:
    _h.SetDirectory(0)
del _te_keep[:]
del _kine_keep[:]
del _hline_keep[:]
c.Close()
c_ov.Close()
root.gROOT.GetListOfFiles().Clear()
root.gROOT.GetListOfCanvases().Clear()
