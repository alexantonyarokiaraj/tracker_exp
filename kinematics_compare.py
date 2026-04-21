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
_save = False        # True = batch mode: save files without opening canvases
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
    elif _arg.lower() in ("save=true", "save=1"):
        _save = True
    elif _arg.lower() in ("save=false", "save=0"):
        _save = False
_run_label = str(_run_filter) if _run_filter is not None else 'all'
print(f"Group filter: single_group={_single_group}, run filter: {_run_label}, use_cutg={_use_cutg}, save={_save}")

root.gROOT.SetBatch(_save)
root.TH1.AddDirectory(False)

_NAN = -999.0

# --- Active volume bounds (10 mm border on all sides) ---
_AV_X_MIN, _AV_X_MAX = 10.0, 246.0  # x: 0+10 to 256-10 mm
_AV_Y_MIN, _AV_Y_MAX = 10.0, 246.0  # y: 0+10 to 256-10 mm
_AV_Z_MIN, _AV_Z_MAX = 10.0, 490.0  # z: 0+10 to 500-10 mm
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
_tag = ("_group" if _single_group else "_nogroup") + ("_cutg" if _use_cutg else "_nocutg")

# Load TCutG before opening other files
_prev_level = root.gErrorIgnoreLevel
root.gErrorIgnoreLevel = root.kFatal
_cutg_ransac_file = root.TFile(os.path.join(os.path.dirname(__file__), "cutg_ransac_final.root"), "READ")
root.gErrorIgnoreLevel = _prev_level
_cutg_ransac = _cutg_ransac_file.Get("cutg_ransac_final")
if not _cutg_ransac:
    print("ERROR: cutg_ransac_final not found in cutg_ransac_final.root")
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
_WIN4_THETA_LO, _WIN4_THETA_HI = 20, 25   # window 4: theta [20,25]°
_WIN4_RNG_LO,   _WIN4_RNG_HI   = 10, 30   # window 4: range [10,30] mm
_gmm_win1_events = []               # GMM passed [WIN1] window but RANSAC did not
_gmm_win2_events = []               # GMM passed [WIN2] window but RANSAC did not
_gmm_win3_events = []               # GMM passed theta[87,89]+range[25,30] but RANSAC did not
_gmm_win4_events = []               # GMM passed theta[20,25]+range[10,30] but RANSAC did not
_ransac_90_failures = []            # RANSAC filter-failure reasons for tracks near 90 deg / 20 mm
_RANSAC_90_THETA_LO, _RANSAC_90_THETA_HI = 85.0, 95.0
_RANSAC_90_RNG_LO,   _RANSAC_90_RNG_HI   = 15.0, 25.0

# Target excitation bin: E*=0 MeV, theta_CM=1.5 deg  (bin width 0.75 MeV x 0.3 deg)
_EX_BIN_TCM_LO, _EX_BIN_TCM_HI = 1.35, 1.65    # theta_CM bin edges (deg)
_EX_BIN_EX_LO,  _EX_BIN_EX_HI  = -0.375, 0.375 # E* bin edges (MeV)
_ex_bin_events = []  # events: GMM+resp in bin, RANSAC missing, with per-filter breakdown
_ransac_ex_bin_events_rev = []  # events: RANSAC in bin, GMM not, with per-filter breakdown
_both_in_bin_count = 0  # events: both GMM and RANSAC filled the elastic bin

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

ransac_excitation      = ([], [])   # (theta_CM_deg, E*_MeV)
ransac_excitation_evts = []          # parallel (run, event_id) for each entry
reg_excitation         = ([], [])
reg_comb_excitation    = ([], [])
reg_comb_excitation_evts = []        # parallel (run, event_id) for each entry

# Filter diagnostic containers — spatial coords of all tracks that pass all
# filters AND the cutg gate. Used for the filter diagnostics canvas.
_filt_ransac = dict(vtx_x=[], vtx_y=[], vtx_z=[], sx=[], sy=[], sz=[],
                    ex=[], ey=[], ez=[], phi=[], track_len=[], charge=[])
_filt_gmm    = dict(vtx_x=[], vtx_y=[], vtx_z=[], sx=[], sy=[], sz=[],
                    ex=[], ey=[], ez=[], phi=[], track_len=[], charge=[])

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
    _gmm_win4_passing = []      # GMM tracks passing all filters with theta[20,25] AND range[10,30]
    _ransac_win4_passing = []   # RANSAC tracks passing all filters with theta[20,25] AND range[10,30]
    _ransac_90_fail_reasons = []  # filter-failure reasons for RANSAC tracks near 90 deg/20 mm
    _ransac_all_fails     = []  # ALL RANSAC filter failures this event (any theta/range)
    _ransac_ex_bin_tracks = []  # RANSAC tracks that actually land in the target (theta_CM, E*) bin
    _ransac_ex_all_passing = [] # ALL passing RANSAC tracks with their (tcm, ex)
    _gmm_ex_bin_tracks    = []  # GMM+resp tracks that land in the target bin
    _gmm_ex_all_passing   = []  # ALL passing GMM comb tracks with their (tcm, ex)
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
            if total_charge > 0 and (_use_cutg is False or _cutg_ransac.IsInside(track_len, total_charge)):
                ransac_range_angle[0].append(theta)
                ransac_range_angle[1].append(rng)
                ransac_charge_length[0].append(track_len)
                ransac_charge_length[1].append(total_charge)
                ransac_charge_length[2].append(theta)
                _filt_ransac['vtx_x'].append(vx); _filt_ransac['vtx_y'].append(vy); _filt_ransac['vtx_z'].append(vz)
                _filt_ransac['sx'].append(sx);    _filt_ransac['sy'].append(sy);    _filt_ransac['sz'].append(sz)
                _filt_ransac['ex'].append(ex);    _filt_ransac['ey'].append(ey);    _filt_ransac['ez'].append(ez)
                _filt_ransac['phi'].append(phi);  _filt_ransac['track_len'].append(track_len); _filt_ransac['charge'].append(total_charge)
                _e_kr = chain.ransac_energy_keV[g][t]
                if _e_kr != _NAN and _e_kr > 0:
                    ransac_theta_energy[0].append(theta)
                    ransac_theta_energy[1].append(_e_kr / 1000.0)
                    _tcm_r, _ex_r = _lab_to_ex(theta, _e_kr / 1000.0)
                    if _tcm_r is not None:
                        ransac_excitation[0].append(_tcm_r)
                        ransac_excitation[1].append(_ex_r)
                        ransac_excitation_evts.append(
                            (int(chain.run_number) if _has_run else -1, event_id))
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
                if _WIN4_THETA_LO <= theta <= _WIN4_THETA_HI and _WIN4_RNG_LO <= rng <= _WIN4_RNG_HI:
                    _ransac_win4_passing.append({'theta': theta, 'rng': rng})
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
            if total_charge > 0 and (_use_cutg is False or _cutg_ransac.IsInside(track_len, total_charge)):
                reg_range_angle[0].append(theta)
                reg_range_angle[1].append(rng)
                reg_charge_length[0].append(track_len)
                reg_charge_length[1].append(total_charge)
                reg_charge_length[2].append(theta)
                _filt_gmm['vtx_x'].append(vx); _filt_gmm['vtx_y'].append(vy); _filt_gmm['vtx_z'].append(vz)
                _filt_gmm['sx'].append(sx);    _filt_gmm['sy'].append(sy);    _filt_gmm['sz'].append(sz)
                _filt_gmm['ex'].append(ex);    _filt_gmm['ey'].append(ey);    _filt_gmm['ez'].append(ez)
                _filt_gmm['phi'].append(phi);  _filt_gmm['track_len'].append(track_len); _filt_gmm['charge'].append(total_charge)
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
                if _WIN4_THETA_LO <= theta <= _WIN4_THETA_HI and _WIN4_RNG_LO <= rng <= _WIN4_RNG_HI:
                    _gmm_win4_passing.append({'theta': theta, 'rng': rng})
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
            # Use scatter-fit geometry (track_len, total_charge) for the cutg decision —
            # the combined fit shares the same track extent; total_charge_c includes beam
            # contributions which would shift the point incorrectly on the cutg plane.
            if total_charge > 0 and (_use_cutg is False or _cutg_ransac.IsInside(track_len, total_charge)):
                reg_comb_range_angle[0].append(theta2)
                reg_comb_range_angle[1].append(rng_c)
                reg_comb_charge_length[0].append(track_len)
                reg_comb_charge_length[1].append(total_charge)  # scatter charge, consistent with cutg
                reg_comb_charge_length[2].append(theta2)
                _e_kc = chain.reg_energy_comb_keV[g][t]
                if _e_kc != _NAN and _e_kc > 0:
                    reg_comb_theta_energy[0].append(theta2)
                    reg_comb_theta_energy[1].append(_e_kc / 1000.0)
                    _tcm_c, _ex_c = _lab_to_ex(theta2, _e_kc / 1000.0)
                    if _tcm_c is not None:
                        reg_comb_excitation[0].append(_tcm_c)
                        reg_comb_excitation[1].append(_ex_c)
                        reg_comb_excitation_evts.append(
                            (int(chain.run_number) if _has_run else -1, event_id))
                        _gmm_ex_all_passing.append({'theta': theta2, 'rng': rng_c,
                                                    'tcm': _tcm_c, 'ex': _ex_c})
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

    # --- Accumulate excitation-bin (E*=0, theta_CM=1.5 deg): RANSAC in bin, GMM not ---
    if _ransac_ex_bin_tracks and not _gmm_ex_bin_tracks:
        _run_eb_rev = int(chain.run_number) if _has_run else -1
        _ransac_ex_bin_events_rev.append({
            'run': _run_eb_rev, 'event': event_id,
            'ransac_tracks':   list(_ransac_ex_bin_tracks),
            'gmm_reasons':     list(_evt_gmm_reasons),
            'gmm_has_track':   _reg_has_any_track,
            'gmm_passed':      _reg_passed,
            'gmm_ex_all':      list(_gmm_ex_all_passing),
        })

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

    # --- Accumulate excitation-bin (E*=0, theta_CM=1.5 deg): BOTH in bin ---
    if _gmm_ex_bin_tracks and _ransac_ex_bin_tracks:
        _both_in_bin_count += 1

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
            'ransac_in_box': [r for r in _ransac_all_tracks
                              if _WIN3_THETA_LO <= r['theta'] <= _WIN3_THETA_HI
                              and _WIN3_RNG_LO   <= r['rng']   <= _WIN3_RNG_HI],
            # pre-filter RANSAC candidates that are only in the theta window (any range)
            'ransac_theta_only': [r for r in _ransac_all_tracks
                                  if _WIN3_THETA_LO <= r['theta'] <= _WIN3_THETA_HI],
        })
    if _gmm_win4_passing and not _ransac_win4_passing:
        _run_this = int(chain.run_number) if _has_run else -1
        _gmm_win4_events.append({
            'run': _run_this, 'event': event_id,
            'gmm_tracks': list(_gmm_win4_passing),
            'ransac_passing': list(_ransac_passing_tracks),
            'ransac_in_box': [r for r in _ransac_all_tracks
                              if _WIN4_THETA_LO <= r['theta'] <= _WIN4_THETA_HI
                              and _WIN4_RNG_LO   <= r['rng']   <= _WIN4_RNG_HI],
            'ransac_theta_only': [r for r in _ransac_all_tracks
                                  if _WIN4_THETA_LO <= r['theta'] <= _WIN4_THETA_HI],
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
    # Sort all events by case (A/B/C), then run, then event number
    def _case_key(eb):
        if eb['ransac_ex_passing']:
            return (1, eb['run'], eb['event'])   # Case B
        elif eb['ransac_all_fails']:
            return (0, eb['run'], eb['event'])   # Case A
        else:
            return (2, eb['run'], eb['event'])   # Case C

    _sorted_events = sorted(_ex_bin_events, key=_case_key)
    _current_case = None
    for _eb in _sorted_events:
        if _eb['ransac_ex_passing']:
            _case = 'B'
        elif _eb['ransac_all_fails']:
            _case = 'A'
        else:
            _case = 'C'
        if _case != _current_case:
            _current_case = _case
            print(f"\n  {'─'*76}")
            print(f"  Case {_case}")
            print(f"  {'─'*76}")
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
print(f"{'='*80}\n")

# ── Parallel commands for TrackReader_image.py (Cases A, B, C) ──────────────
from collections import defaultdict as _ddict
_case_a_by_run = _ddict(list)
_case_b_by_run = _ddict(list)
_case_c_by_run = _ddict(list)
for _eb in _ex_bin_events:
    _r = _eb['run']
    _e = _eb['event']
    if _eb['ransac_ex_passing']:
        _case_b_by_run[_r].append(_e)
    elif _eb['ransac_all_fails']:
        _case_a_by_run[_r].append(_e)
    else:
        _case_c_by_run[_r].append(_e)

def _parallel_cmd(case_label, by_run):
    print(f"\n# Case {case_label}")
    for _r, _evts in sorted(by_run.items()):
        _evts_str = ' '.join(str(e) for e in _evts)
        print(f"parallel -j 18 python3 TrackReader_image.py {_r}@{{}}_{{}} ::: {_evts_str}")

print("\n" + "="*80)
print("PARALLEL COMMANDS  —  TrackReader_image.py per case")
print("="*80)
_parallel_cmd('A', _case_a_by_run)
_parallel_cmd('B', _case_b_by_run)
_parallel_cmd('C', _case_c_by_run)
print("="*80 + "\n")

# ── Excitation-bin diagnostic (REVERSE): RANSAC in bin, GMM not ──────────────
print(f"\n{'='*80}")
print(f"EXCITATION BIN DIAGNOSTIC (REVERSE)  —  E* [{_EX_BIN_EX_LO},{_EX_BIN_EX_HI}] MeV  x  "
      f"theta_CM [{_EX_BIN_TCM_LO},{_EX_BIN_TCM_HI}] deg")
print(f"  Events where RANSAC filled this bin and GMM did not")
print(f"{'='*80}")
print(f"  Total such events: {len(_ransac_ex_bin_events_rev)}")
if _ransac_ex_bin_events_rev:
    from collections import Counter as _Counter2
    _rev_reason_ctr    = _Counter2()
    _rev_elsewhere_ctr = _Counter2()
    _rev_no_cand_ctr   = 0
    for _rb in _ransac_ex_bin_events_rev:
        if _rb['gmm_passed'] and _rb['gmm_ex_all']:
            # Case B: GMM passed filters but landed elsewhere
            for _gp in _rb['gmm_ex_all']:
                _tcm_b = round(_gp['tcm'] / 0.3) * 0.3
                _ex_b  = round(_gp['ex']  / 0.75) * 0.75
                _rev_elsewhere_ctr[f"tcm~{_tcm_b:.1f} ex~{_ex_b:.2f}"] += 1
        elif _rb['gmm_has_track']:
            # Case A: GMM had tracks but all failed a filter
            for _reason, _detail in _rb['gmm_reasons']:
                _rev_reason_ctr[_reason.split()[0]] += 1
        else:
            # Case C: GMM produced no scatter tracks
            _rev_no_cand_ctr += 1
    print(f"\n  Case A — GMM had tracks but ALL failed a filter ({sum(_rev_reason_ctr.values())} tracks):")
    for _rk, _rc in _rev_reason_ctr.most_common():
        print(f"    {_rc:5d}  {_rk}")
    print(f"\n  Case B — GMM passed filters but landed in a DIFFERENT bin:")
    for _bk, _bc in _rev_elsewhere_ctr.most_common(15):
        print(f"    {_bc:5d}  {_bk}")
    print(f"\n  Case C — GMM produced no scatter tracks at all: {_rev_no_cand_ctr}")
print(f"{'='*80}\n")

# ── Elastic-bin Venn diagram ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches
import matplotlib.patheffects as _pe
from collections import Counter as _VCounter

# ── Rebuild per-case counts from stored lists ────────────────────────────────
def _classify_forward(ev):
    if ev['ransac_ex_passing']:   return 'B'
    elif ev['ransac_all_fails']:  return 'A'
    else:                         return 'C'

def _classify_reverse(ev):
    if ev['gmm_passed'] and ev['gmm_ex_all']:  return 'B'
    elif ev['gmm_has_track']:                  return 'A'
    else:                                      return 'C'

_fwd_cases = [_classify_forward(e) for e in _ex_bin_events]
_rev_cases = [_classify_reverse(e) for e in _ransac_ex_bin_events_rev]

_fwd_total = len(_ex_bin_events)
_rev_total = len(_ransac_ex_bin_events_rev)
_both_total = _both_in_bin_count

_fwd_A = _fwd_cases.count('A')
_fwd_B = _fwd_cases.count('B')
_fwd_C = _fwd_cases.count('C')
_rev_A = _rev_cases.count('A')
_rev_B = _rev_cases.count('B')
_rev_C = _rev_cases.count('C')

# ── Sub-filter breakdowns ──────────────────────────────────────────────────
# Forward Case A: filter reasons
_fwd_A_ctr = _VCounter()
for ev in _ex_bin_events:
    if _classify_forward(ev) == 'A':
        for rf in ev['ransac_all_fails']:
            _fwd_A_ctr[rf['reason'].split()[0]] += 1

# Forward Case B: destination bins
_fwd_B_ctr = _VCounter()
for ev in _ex_bin_events:
    if _classify_forward(ev) == 'B':
        for rp in ev['ransac_ex_passing']:
            _tcm_b = round(rp['tcm'] / 0.3) * 0.3
            _ex_b  = round(rp['ex']  / 0.75) * 0.75
            _fwd_B_ctr[f"tcm~{_tcm_b:.1f} ex~{_ex_b:.2f}"] += 1

# Reverse Case A: filter reasons
_rev_A_ctr = _VCounter()
for ev in _ransac_ex_bin_events_rev:
    if _classify_reverse(ev) == 'A':
        for reason, _ in ev['gmm_reasons']:
            _rev_A_ctr[reason.split()[0]] += 1

# Reverse Case B: destination bins
_rev_B_ctr = _VCounter()
for ev in _ransac_ex_bin_events_rev:
    if _classify_reverse(ev) == 'B':
        for gp in ev['gmm_ex_all']:
            _tcm_b = round(gp['tcm'] / 0.3) * 0.3
            _ex_b  = round(gp['ex']  / 0.75) * 0.75
            _rev_B_ctr[f"tcm~{_tcm_b:.1f} ex~{_ex_b:.2f}"] += 1

# ── Layout: Venn on top, two breakdown bars below ──────────────────────────
_fig = _plt.figure(figsize=(18, 13))
_fig.patch.set_facecolor('#1a1a2e')

_ax_venn  = _fig.add_axes([0.03, 0.48, 0.94, 0.50])   # top: Venn
_ax_fwd   = _fig.add_axes([0.05, 0.05, 0.42, 0.38])   # bottom-left: forward bars
_ax_rev   = _fig.add_axes([0.53, 0.05, 0.42, 0.38])   # bottom-right: reverse bars

_C_BG    = '#1a1a2e'
_C_GMM   = '#3a86ff'   # blue  — GMM only
_C_BOTH  = '#8338ec'   # purple — both
_C_RAC   = '#ff6b6b'   # red   — RANSAC only
_C_TEXT  = '#e0e0e0'
_C_A     = '#ffd166'   # yellow
_C_B     = '#06d6a0'   # teal
_C_CC    = '#ef476f'   # pink

for _ax in [_ax_venn, _ax_fwd, _ax_rev]:
    _ax.set_facecolor(_C_BG)
    for sp in _ax.spines.values():
        sp.set_visible(False)

# ── Venn circles ──────────────────────────────────────────────────────────────
_ax_venn.set_xlim(0, 10)
_ax_venn.set_ylim(0, 5)
_ax_venn.set_aspect('equal')
_ax_venn.axis('off')

_r = 1.9
_cx_gmm  = 3.4
_cx_rac  = 6.6
_cy      = 2.4

_circ_gmm  = _patches.Circle((_cx_gmm, _cy), _r, color=_C_GMM,  alpha=0.35, zorder=2)
_circ_rac  = _patches.Circle((_cx_rac, _cy), _r, color=_C_RAC,  alpha=0.35, zorder=2)
_ax_venn.add_patch(_circ_gmm)
_ax_venn.add_patch(_circ_rac)

# overlap wedge approximated with a third circle fill in purple
_circ_ov = _patches.Circle(((_cx_gmm+_cx_rac)/2, _cy), _r*0.38, color=_C_BOTH, alpha=0.9, zorder=3)
_ax_venn.add_patch(_circ_ov)

# circle outlines
for _cx, _col in [(_cx_gmm, _C_GMM), (_cx_rac, _C_RAC)]:
    _out = _patches.Circle((_cx, _cy), _r, fill=False, edgecolor=_col, linewidth=2.5, zorder=4)
    _ax_venn.add_patch(_out)

# labels
_pefx = [_pe.withStroke(linewidth=3, foreground='black')]
_ax_venn.text(_cx_gmm - _r*0.55, _cy + _r*1.05, "GMM only", color=_C_GMM,  fontsize=13,
              fontweight='bold', ha='center', zorder=5)
_ax_venn.text(_cx_rac + _r*0.55, _cy + _r*1.05, "RANSAC only", color=_C_RAC, fontsize=13,
              fontweight='bold', ha='center', zorder=5)
_ax_venn.text(5.0, _cy + _r*1.05, "Both", color=_C_BOTH, fontsize=13,
              fontweight='bold', ha='center', zorder=5)

# counts inside circles
_ax_venn.text(_cx_gmm - 0.7, _cy, str(_fwd_total), color=_C_TEXT, fontsize=22,
              fontweight='bold', ha='center', va='center', zorder=6,
              path_effects=_pefx)
_ax_venn.text(5.0, _cy, str(_both_total), color='white', fontsize=22,
              fontweight='bold', ha='center', va='center', zorder=6,
              path_effects=_pefx)
_ax_venn.text(_cx_rac + 0.7, _cy, str(_rev_total), color=_C_TEXT, fontsize=22,
              fontweight='bold', ha='center', va='center', zorder=6,
              path_effects=_pefx)

# sub-counts A/B/C inside each circle
_ax_venn.text(_cx_gmm - 0.7, _cy - 0.65,
              f"A:{_fwd_A}  B:{_fwd_B}  C:{_fwd_C}",
              color=_C_TEXT, fontsize=10, ha='center', va='center', zorder=6)
_ax_venn.text(_cx_rac + 0.7, _cy - 0.65,
              f"A:{_rev_A}  B:{_rev_B}  C:{_rev_C}",
              color=_C_TEXT, fontsize=10, ha='center', va='center', zorder=6)

_ax_venn.set_title("Elastic bin  E* ∈ [−0.375, 0.375] MeV  ×  θ_CM ∈ [1.35, 1.65]°",
                   color=_C_TEXT, fontsize=13, pad=8)

# ── Helper: horizontal stacked bar per case ──────────────────────────────────
def _draw_breakdown(_ax, total, case_a, case_b, case_c,
                    a_ctr, b_ctr, label, text_col):
    _ax.set_facecolor(_C_BG)
    _ax.set_xlim(0, 1)
    _ax.axis('off')

    # title
    _ax.text(0.5, 0.97, f"{label}  (total = {total})",
             color=text_col, fontsize=12, fontweight='bold',
             ha='center', va='top', transform=_ax.transAxes)

    _bar_h   = 0.13
    _bar_top = 0.82
    _gap     = 0.04

    # top bar: total split A/B/C
    _segs = [(case_a, _C_A, 'A'), (case_b, _C_B, 'B'), (case_c, _C_CC, 'C')]
    _x = 0.0
    for _n, _col, _lbl in _segs:
        _w = _n / total if total else 0
        _rect = _patches.FancyBboxPatch((_x, _bar_top), _w, _bar_h,
                                        boxstyle="square,pad=0",
                                        facecolor=_col, edgecolor='none',
                                        transform=_ax.transAxes, clip_on=True)
        _ax.add_patch(_rect)
        if _w > 0.04:
            _ax.text(_x + _w/2, _bar_top + _bar_h/2,
                     f"{_lbl}\n{_n}", color='black', fontsize=9,
                     fontweight='bold', ha='center', va='center',
                     transform=_ax.transAxes)
        _x += _w

    # Case A sub-bar
    _ay = _bar_top - _bar_h - _gap
    _ax.text(0.0, _ay + _bar_h + 0.005, f"Case A — filter failures  ({case_a} events / {sum(a_ctr.values())} tracks)",
             color=_C_A, fontsize=8.5, va='bottom', transform=_ax.transAxes)
    _a_total = sum(a_ctr.values()) or 1
    _a_top5 = a_ctr.most_common(5)
    _x = 0.0
    _cols_a = ['#ffd166','#f4a261','#e76f51','#e9c46a','#2a9d8f']
    for _i, (_k, _v) in enumerate(_a_top5):
        _w = _v / _a_total
        _rect = _patches.FancyBboxPatch((_x, _ay), _w, _bar_h,
                                        boxstyle="square,pad=0",
                                        facecolor=_cols_a[_i % len(_cols_a)], edgecolor='none',
                                        transform=_ax.transAxes, clip_on=True)
        _ax.add_patch(_rect)
        if _w > 0.05:
            _short = _k[:12]
            _ax.text(_x + _w/2, _ay + _bar_h/2,
                     f"{_short}\n{_v}", color='black', fontsize=7,
                     ha='center', va='center', transform=_ax.transAxes)
        _x += _w

    # Case B sub-bar
    _by = _ay - _bar_h - _gap
    _ax.text(0.0, _by + _bar_h + 0.005, f"Case B — landed in different bin  ({case_b} events / {sum(b_ctr.values())} tracks)",
             color=_C_B, fontsize=8.5, va='bottom', transform=_ax.transAxes)
    _b_total = sum(b_ctr.values()) or 1
    _b_top5 = b_ctr.most_common(6)
    _x = 0.0
    _cols_b = ['#06d6a0','#118ab2','#073b4c','#0096c7','#48cae4','#90e0ef']
    for _i, (_k, _v) in enumerate(_b_top5):
        _w = _v / _b_total
        _rect = _patches.FancyBboxPatch((_x, _by), _w, _bar_h,
                                        boxstyle="square,pad=0",
                                        facecolor=_cols_b[_i % len(_cols_b)], edgecolor='none',
                                        transform=_ax.transAxes, clip_on=True)
        _ax.add_patch(_rect)
        if _w > 0.04:
            _ax.text(_x + _w/2, _by + _bar_h/2,
                     f"{_k}\n{_v}", color='white', fontsize=6.5,
                     ha='center', va='center', transform=_ax.transAxes)
        _x += _w

    # Case C label
    _cy2 = _by - _bar_h*0.6 - _gap
    _ax.text(0.0, _cy2, f"Case C — no tracks produced at all:  {case_c} events",
             color=_C_CC, fontsize=8.5, va='top', transform=_ax.transAxes)

_draw_breakdown(_ax_fwd, _fwd_total, _fwd_A, _fwd_B, _fwd_C,
                _fwd_A_ctr, _fwd_B_ctr,
                "FORWARD  (GMM in bin, RANSAC not)", _C_GMM)
_draw_breakdown(_ax_rev, _rev_total, _rev_A, _rev_B, _rev_C,
                _rev_A_ctr, _rev_B_ctr,
                "REVERSE  (RANSAC in bin, GMM not)", _C_RAC)

_venn_path = os.path.join(images_dir, f"elastic_bin_venn{_tag}.png")
_fig.savefig(_venn_path, dpi=150, bbox_inches='tight', facecolor=_C_BG)
_plt.close(_fig)
print(f"[Venn diagram saved] {_venn_path}")


# ── WIN4 diagnostic: GMM passed theta[20,25]+range[10,30], RANSAC did not ────
print(f"\n{'='*80}")
print(f"WIN4 DIAGNOSTIC  —  theta [{_WIN4_THETA_LO},{_WIN4_THETA_HI}] deg  x  range [{_WIN4_RNG_LO},{_WIN4_RNG_HI}] mm")
print(f"  Events where GMM scatter passed this box and RANSAC did not")
print(f"{'='*80}")
print(f"  Total such events: {len(_gmm_win4_events)}")
for _w4 in _gmm_win4_events:
    _gmm_str = "  |  ".join(f"theta={t['theta']:.1f} rng={t['rng']:.1f}" for t in _w4['gmm_tracks'])
    print(f"  run={_w4['run']}  event={_w4['event']}")
    print(f"    GMM  : {_gmm_str}")
    if _w4['ransac_in_box']:
        _rib = "  |  ".join(f"theta={r['theta']:.1f} rng={r['rng']:.1f}" for r in _w4['ransac_in_box'])
        print(f"    RANSAC in box (pre-filter): {_rib}")
    elif _w4['ransac_theta_only']:
        _rto = "  |  ".join(f"theta={r['theta']:.1f} rng={r['rng']:.1f}" for r in _w4['ransac_theta_only'])
        print(f"    RANSAC theta-only (pre-filter): {_rto}")
    else:
        print(f"    RANSAC: no candidates in theta window")
print(f"{'='*80}\n")


_THETA_LO, _THETA_HI = 70, 110   # projection slice in degrees

# --- Canvas: 4 rows x 4 cols ---
c = root.TCanvas("c_kin", "Kinematics", 2400, 2000)
c.Divide(4, 4)

datasets = [
    ("RANSAC",        ransac_range_angle,    ransac_charge_length,    _cutg_ransac),
    ("GMM_REG",       reg_range_angle,       reg_charge_length,       _cutg_ransac),
    ("GMM_REG comb",  reg_comb_range_angle,  reg_comb_charge_length,  _cutg_ransac),
]

keep = []  # prevent ROOT garbage collection
for row, (label, ra_data, cl_data, _row_cutg) in enumerate(datasets):
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
    root.gPad.SetLogz(1)
    keep.append(h_ra)

    # Col 2: Total Charge vs Track Length
    c.cd(4 * row + 2)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    h_cl = root.TH2F(
        f"h_cl_{row}", f"{label}: Total Charge vs Track Length;Track Length (mm);Total Charge (a.u.)",
        200, 0, 400, 200, 0, 100000,
    )
    h_cl.SetDirectory(0)
    for tl, tc in zip(cl_data[0], cl_data[1]):
        h_cl.Fill(tl, tc)
    h_cl.Draw("COLZ")
    root.gPad.SetLogz(1)
    _row_cutg.Draw("same")
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
        500, 0, 100000,
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
root.gPad.SetLogz(1)
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
root.gPad.SetLogz(1)
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
if not _save:
    c.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
c.SaveAs(os.path.join(images_dir, f"kinematics{_tag}.png"))

# --- Overlay: theta projections for all three methods ---
c_ov = root.TCanvas("c_overlay", "Theta Projection Comparison", 900, 700)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
root.gPad.SetLogy(1)

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
if not _save:
    c_ov.WaitPrimitive()
c_ov.SaveAs(os.path.join(images_dir, f"theta_projection_overlay{_tag}.png"))
c_ov.SaveAs(os.path.join(images_dir, f"theta_projection_overlay{_tag}.root"))

# Save all histograms to ROOT file
root_out_path = os.path.join(images_dir, f"kinematics{_tag}.root")
fout = root.TFile(root_out_path, "RECREATE")
for h in keep:
    h.Write()
for h in _ov_histos:
    h.Write()
fout.Close()

print(f"Points — RANSAC: {len(ransac_range_angle[0])}, REG: {len(reg_range_angle[0])}, REG_comb: {len(reg_comb_range_angle[0])}")
print(f"Saved to {os.path.join(images_dir, f'kinematics{_tag}.png')} and {root_out_path}")

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
    root.gPad.SetLogz(1)
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
root.gPad.SetLogz(1)
_te_keep.append(_h_diff_te)
for _ev, _hl in _hlines:
    _hl.Draw("SAME")
_draw_ex_legend()

c_te.Update()
if not _save:
    c_te.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
c_te.SaveAs(os.path.join(images_dir, f"kinematics_compare_ex_cm{_tag}.png"))
c_te.SaveAs(os.path.join(images_dir, f"kinematics_compare_ex_cm{_tag}.root"))

print(f"excitation-energy points — RANSAC: {len(ransac_excitation[0])},"
      f" GMM: {len(reg_excitation[0])}, GMM+resp: {len(reg_comb_excitation[0])}")

# ─────────────────────────────────────────────────────────────────────────────
# Performance-gain canvas: GMM+resp vs RANSAC
# Pad 1: relative gain (GMM-RANSAC)/RANSAC per 0.3°-θ_CM bin, E* ∈ [-1, 1] MeV
# Pad 2: same,                                                  E* ∈ [9, 11]  MeV
# Pad 3: overlay E* histograms [16.5, 18.5] MeV, θ_CM ∈ [1°, 5°]
# ─────────────────────────────────────────────────────────────────────────────
_pg_tcm_lo  = 1.0
_pg_tcm_hi  = 5.0
_pg_bin_w   = 0.3                              # deg per bin
_pg_n_bins  = int(round((_pg_tcm_hi - _pg_tcm_lo) / _pg_bin_w))   # 13
_pg_tcm_hi_adj = _pg_tcm_lo + _pg_n_bins * _pg_bin_w              # 4.9

def _pg_bin_counts(ex_data, ex_lo, ex_hi):
    """Count entries per θ_CM bin (0.3°) for a given E* slice."""
    counts = [0] * _pg_n_bins
    for _tc, _ev in zip(ex_data[0], ex_data[1]):
        if ex_lo <= _ev < ex_hi and _pg_tcm_lo <= _tc < _pg_tcm_hi_adj:
            _bi = int((_tc - _pg_tcm_lo) / _pg_bin_w)
            if 0 <= _bi < _pg_n_bins:
                counts[_bi] += 1
    return counts

_pg_slices = [
    ("E* #in [#minus1, 1] MeV", -1.0,  1.0),
    ("E* #in [9, 11] MeV",       9.0, 11.0),
]

c_pg = root.TCanvas("c_pg", "GMM+resp vs RANSAC performance gain", 4800, 700)
c_pg.Divide(6, 1)
_pg_keep = []

for _pi, (_pg_title, _pg_elo, _pg_ehi) in enumerate(_pg_slices, start=1):
    c_pg.cd(_pi)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)

    _cnts_r = _pg_bin_counts(ransac_excitation,   _pg_elo, _pg_ehi)
    _cnts_g = _pg_bin_counts(reg_comb_excitation, _pg_elo, _pg_ehi)

    _h_gain = root.TH1F(
        f"h_gain_{_pi}",
        f"{_pg_title};#theta_{{CM}} (deg);(GMM+resp #minus RANSAC) / RANSAC",
        _pg_n_bins, _pg_tcm_lo, _pg_tcm_hi_adj,
    )
    _h_gain.SetDirectory(0)
    _ylo_gain, _yhi_gain = float("inf"), float("-inf")
    for _bi in range(_pg_n_bins):
        _r = _cnts_r[_bi]
        _g = _cnts_g[_bi]
        if _r > 0:
            _val = (_g - _r) / float(_r)
            _h_gain.SetBinContent(_bi + 1, _val)
            # Poisson error on ratio: sqrt(g/r²  + g²/r³) ≈ sqrt((g+r)/r²) for large counts
            _err = math.sqrt(_g / _r**2 + _g**2 / _r**3) if _r > 0 and _g > 0 else (1.0 / math.sqrt(_r) if _r > 0 else 0)
            _h_gain.SetBinError(_bi + 1, _err)
            _ylo_gain = min(_ylo_gain, _val - _err)
            _yhi_gain = max(_yhi_gain, _val + _err)
    if _ylo_gain == float("inf"):
        _ylo_gain, _yhi_gain = -1.0, 2.0
    _margin = max(0.5, (_yhi_gain - _ylo_gain) * 0.2)
    _h_gain.GetYaxis().SetRangeUser(_ylo_gain - _margin, _yhi_gain + _margin)
    _h_gain.SetLineColor(root.kBlue + 1)
    _h_gain.SetMarkerColor(root.kBlue + 1)
    _h_gain.SetMarkerStyle(20)
    _h_gain.SetMarkerSize(0.8)
    _h_gain.Draw("E1")
    _pg_line0 = root.TLine(_pg_tcm_lo, 0.0, _pg_tcm_hi_adj, 0.0)
    _pg_line0.SetLineColor(root.kRed)
    _pg_line0.SetLineStyle(2)
    _pg_line0.SetLineWidth(2)
    _pg_line0.Draw()
    _pg_keep += [_h_gain, _pg_line0]
    # Print counts per θ_CM bin
    print(f"\nPerformance gain counts: {_pg_title}")
    print(f"  {'theta_CM_center':>16}  {'RANSAC':>8}  {'GMM+resp':>10}  {'(G-R)/R':>8}")
    for _bi in range(_pg_n_bins):
        _tc_lo_b = _pg_tcm_lo + _bi * _pg_bin_w
        _tc_hi_b = _tc_lo_b + _pg_bin_w
        _tc_ctr  = _tc_lo_b + 0.5 * _pg_bin_w
        _rc, _gc = _cnts_r[_bi], _cnts_g[_bi]
        _gs = f"{(_gc - _rc) / _rc:.3f}" if _rc > 0 else "N/A"
        print(f"  {_tc_ctr:>16.2f}  {_rc:>8d}  {_gc:>10d}  {_gs:>8}")
        # Print per-event breakdown when counts differ, showing both sides so net == G - R
        if _gc != _rc:
            _r_cnt_evt = {}
            for _idx, (_tc, _ev) in enumerate(zip(ransac_excitation[0], ransac_excitation[1])):
                if _pg_elo <= _ev < _pg_ehi and _tc_lo_b <= _tc < _tc_hi_b:
                    _key = ransac_excitation_evts[_idx]
                    _r_cnt_evt[_key] = _r_cnt_evt.get(_key, 0) + 1
            _g_cnt_evt = {}
            for _idx, (_tc, _ev) in enumerate(zip(reg_comb_excitation[0], reg_comb_excitation[1])):
                if _pg_elo <= _ev < _pg_ehi and _tc_lo_b <= _tc < _tc_hi_b:
                    _key = reg_comb_excitation_evts[_idx]
                    _g_cnt_evt[_key] = _g_cnt_evt.get(_key, 0) + 1
            _all_keys = set(_r_cnt_evt) | set(_g_cnt_evt)
            _gmm_surplus  = sorted(
                (_re, _ee, _g_cnt_evt.get((_re,_ee),0) - _r_cnt_evt.get((_re,_ee),0))
                for (_re, _ee) in _all_keys
                if _g_cnt_evt.get((_re,_ee),0) > _r_cnt_evt.get((_re,_ee),0)
            )
            _rans_surplus = sorted(
                (_re, _ee, _r_cnt_evt.get((_re,_ee),0) - _g_cnt_evt.get((_re,_ee),0))
                for (_re, _ee) in _all_keys
                if _r_cnt_evt.get((_re,_ee),0) > _g_cnt_evt.get((_re,_ee),0)
            )
            _tot_g = sum(_d for _,_,_d in _gmm_surplus)
            _tot_r = sum(_d for _,_,_d in _rans_surplus)
            print(f"    net = GMM_surplus({_tot_g}) - RANSAC_surplus({_tot_r}) = {_tot_g - _tot_r}  (== G-R = {_gc-_rc})")
            if _gmm_surplus:
                print(f"    GMM-surplus events (run, event, +delta):")
                for _re, _ee, _d in _gmm_surplus:
                    print(f"      run={_re}  event={_ee}  +{_d}")
            if _rans_surplus:
                print(f"    RANSAC-surplus events (run, event, +delta)  [cancel above]:")
                for _re, _ee, _d in _rans_surplus:
                    print(f"      run={_re}  event={_ee}  +{_d}")

# Pad 3: overlay E* histogram, E* ∈ [16.5, 18.5] MeV, θ_CM ∈ [1°, 5°]
c_pg.cd(3)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
_pg_elo3, _pg_ehi3 = 16.5, 18.5
_pg_nb3 = 16   # 16 bins × 0.125 MeV/bin
_h_ov_r = root.TH1F(
    "h_ov_ransac",
    f"E* #in [{_pg_elo3}, {_pg_ehi3}] MeV, #theta_{{CM}} #in [1, 5]#circ;"
    "E* (MeV);Counts",
    _pg_nb3, _pg_elo3, _pg_ehi3,
)
_h_ov_g = root.TH1F("h_ov_gmm", "", _pg_nb3, _pg_elo3, _pg_ehi3)
_h_ov_r.SetDirectory(0)
_h_ov_g.SetDirectory(0)
for _tc, _ev in zip(ransac_excitation[0], ransac_excitation[1]):
    if _pg_tcm_lo <= _tc <= _pg_tcm_hi and _pg_elo3 <= _ev <= _pg_ehi3:
        _h_ov_r.Fill(_ev)
for _tc, _ev in zip(reg_comb_excitation[0], reg_comb_excitation[1]):
    if _pg_tcm_lo <= _tc <= _pg_tcm_hi and _pg_elo3 <= _ev <= _pg_ehi3:
        _h_ov_g.Fill(_ev)
_pg_ymax3 = max(_h_ov_r.GetMaximum(), _h_ov_g.GetMaximum()) * 1.3
if _pg_ymax3 <= 0:
    _pg_ymax3 = 10.0
_h_ov_r.GetYaxis().SetRangeUser(0, _pg_ymax3)
_h_ov_r.SetLineColor(root.kRed + 1)
_h_ov_r.SetLineWidth(2)
_h_ov_g.SetLineColor(root.kBlue + 1)
_h_ov_g.SetLineWidth(2)
_h_ov_r.Draw("HIST")
_h_ov_g.Draw("HIST SAME")
_pg_leg3 = root.TLegend(0.62, 0.72, 0.92, 0.88)
_pg_leg3.SetBorderSize(0)
_pg_leg3.AddEntry(_h_ov_r, "RANSAC",   "l")
_pg_leg3.AddEntry(_h_ov_g, "GMM+resp", "l")
_pg_leg3.Draw()
_pg_keep += [_h_ov_r, _h_ov_g, _pg_leg3]

# Pad 4: θ_CM histogram, counts summed over E* ∈ [16.5, 18.5] MeV
c_pg.cd(4)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
_pg_nb4  = 13   # 13 bins × 0.3°/bin matching pads 1&2
_h_tcm_r = root.TH1F(
    "h_tcm_ransac",
    "E* #in [16.5, 18.5] MeV;#theta_{CM} (deg);Counts",
    _pg_nb4, _pg_tcm_lo, _pg_tcm_lo + _pg_nb4 * _pg_bin_w,
)
_h_tcm_g = root.TH1F(
    "h_tcm_gmm", "",
    _pg_nb4, _pg_tcm_lo, _pg_tcm_lo + _pg_nb4 * _pg_bin_w,
)
_h_tcm_r.SetDirectory(0)
_h_tcm_g.SetDirectory(0)
for _tc, _ev in zip(ransac_excitation[0], ransac_excitation[1]):
    if 16.5 <= _ev <= 18.5:
        _h_tcm_r.Fill(_tc)
for _tc, _ev in zip(reg_comb_excitation[0], reg_comb_excitation[1]):
    if 16.5 <= _ev <= 18.5:
        _h_tcm_g.Fill(_tc)
_pg_ymax4 = max(_h_tcm_r.GetMaximum(), _h_tcm_g.GetMaximum()) * 1.3
if _pg_ymax4 <= 0:
    _pg_ymax4 = 10.0
_h_tcm_r.GetYaxis().SetRangeUser(0, _pg_ymax4)
_h_tcm_r.SetLineColor(root.kRed + 1)
_h_tcm_r.SetLineWidth(2)
_h_tcm_g.SetLineColor(root.kBlue + 1)
_h_tcm_g.SetLineWidth(2)
_h_tcm_r.Draw("HIST")
_h_tcm_g.Draw("HIST SAME")
_pg_leg4 = root.TLegend(0.62, 0.72, 0.92, 0.88)
_pg_leg4.SetBorderSize(0)
_pg_leg4.AddEntry(_h_tcm_r, "RANSAC",   "l")
_pg_leg4.AddEntry(_h_tcm_g, "GMM+resp", "l")
_pg_leg4.Draw()
_pg_keep += [_h_tcm_r, _h_tcm_g, _pg_leg4]

# Pads 5 & 6: θ_CM histograms (RANSAC vs GMM+resp) gated on E* ∈ [-1,1] and [9,11] MeV
for _oi, (_ot, _oelo, _oehi) in enumerate([
    ("E* #in [#minus1, 1] MeV", -1.0,  1.0),
    ("E* #in [9, 11] MeV",       9.0, 11.0),
], start=5):
    c_pg.cd(_oi)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _h_eov_r = root.TH1F(
        f"h_eov_ransac_{_oi}",
        f"{_ot};#theta_{{CM}} (deg);Counts",
        _pg_n_bins, _pg_tcm_lo, _pg_tcm_hi_adj,
    )
    _h_eov_g = root.TH1F(
        f"h_eov_gmm_{_oi}", "",
        _pg_n_bins, _pg_tcm_lo, _pg_tcm_hi_adj,
    )
    _h_eov_r.SetDirectory(0)
    _h_eov_g.SetDirectory(0)
    for _tc, _ev in zip(ransac_excitation[0], ransac_excitation[1]):
        if _oelo <= _ev < _oehi and _pg_tcm_lo <= _tc < _pg_tcm_hi_adj:
            _h_eov_r.Fill(_tc)
    for _tc, _ev in zip(reg_comb_excitation[0], reg_comb_excitation[1]):
        if _oelo <= _ev < _oehi and _pg_tcm_lo <= _tc < _pg_tcm_hi_adj:
            _h_eov_g.Fill(_tc)
    _pg_ymax_ov = max(_h_eov_r.GetMaximum(), _h_eov_g.GetMaximum()) * 1.3
    if _pg_ymax_ov <= 0:
        _pg_ymax_ov = 10.0
    _h_eov_r.GetYaxis().SetRangeUser(0, _pg_ymax_ov)
    _h_eov_r.SetLineColor(root.kRed + 1)
    _h_eov_r.SetLineWidth(2)
    _h_eov_g.SetLineColor(root.kBlue + 1)
    _h_eov_g.SetLineWidth(2)
    _h_eov_r.Draw("HIST")
    _h_eov_g.Draw("HIST SAME")
    _pg_leg_ov = root.TLegend(0.62, 0.72, 0.92, 0.88)
    _pg_leg_ov.SetBorderSize(0)
    _pg_leg_ov.AddEntry(_h_eov_r, "RANSAC",   "l")
    _pg_leg_ov.AddEntry(_h_eov_g, "GMM+resp", "l")
    _pg_leg_ov.Draw()
    _pg_keep += [_h_eov_r, _h_eov_g, _pg_leg_ov]

c_pg.Update()
if not _save:
    c_pg.WaitPrimitive()
c_pg.SaveAs(os.path.join(images_dir, f"performance_gain{_tag}.png"))
c_pg.SaveAs(os.path.join(images_dir, f"performance_gain{_tag}.root"))

# Save a separate ROOT file with only the two gain-ratio histograms (pads 1 & 2)
_c_gain2 = root.TCanvas("c_gain2", "GMM+resp vs RANSAC gain ratio", 1600, 700)
_c_gain2.Divide(2, 1)
for _pi2, _hg in enumerate(
    [_o for _o in _pg_keep if isinstance(_o, root.TH1F) and _o.GetName().startswith("h_gain_")],
    start=1,
):
    _c_gain2.cd(_pi2)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _hg.Draw("E1")
    # redraw the zero line
    _zl = root.TLine(_pg_tcm_lo, 0.0, _pg_tcm_hi_adj, 0.0)
    _zl.SetLineColor(root.kRed); _zl.SetLineStyle(2); _zl.SetLineWidth(2)
    _zl.Draw()
    _pg_keep.append(_zl)
_c_gain2.Update()
_c_gain2.SaveAs(os.path.join(images_dir, f"performance_gain_ratio_only{_tag}.root"))
root.gROOT.GetListOfCanvases().Remove(_c_gain2)
del _c_gain2

# ─────────────────────────────────────────────────────────────────────────────
# Filter diagnostics — two 3×3 canvases (RANSAC and GMM+resp separately)
# Pad 1: Vertex X vs Y        Pad 2: Vertex Z vs End Z   Pad 3: End X vs Y
# Pad 4: Start X vs Y         Pad 5: Phi (1D)             Pad 6: End Y (1D)
# Pad 7: Vertex Y (1D)        Pad 8: Charge vs Len        Pad 9: Track Len (1D)
# Cyan   = active volume cut [3,253]/[3,497]
# Orange = beam zone (relaxed [118,136] vertex, strict [122,132] end)
# Red    = phi exclusion zones [±70°, ±110°]
# ─────────────────────────────────────────────────────────────────────────────
_AV_LC  = root.kCyan + 1
_BZ_LC  = root.kOrange + 1
_PHI_LC = root.kRed
_AV_BOX = [(_AV_X_MIN,_AV_Y_MIN,_AV_X_MAX,_AV_Y_MIN), (_AV_X_MAX,_AV_Y_MIN,_AV_X_MAX,_AV_Y_MAX),
           (_AV_X_MAX,_AV_Y_MAX,_AV_X_MIN,_AV_Y_MAX), (_AV_X_MIN,_AV_Y_MAX,_AV_X_MIN,_AV_Y_MIN)]

def _fline(keep, x1, y1, x2, y2, col, sty=2, wid=2):
    _l = root.TLine(x1, y1, x2, y2)
    _l.SetLineColor(col); _l.SetLineStyle(sty); _l.SetLineWidth(wid)
    _l.Draw(); keep.append(_l)

for _fd_name, _fd_label, _fd_col, _fd, _fd_cutg in [
        ("ransac", "RANSAC",   root.kBlue, _filt_ransac, _cutg_ransac),
        ("gmm",    "GMM+resp", root.kRed,  _filt_gmm,    _cutg_ransac)]:
    _c_filt = root.TCanvas(f"c_filt_{_fd_name}", f"Filter Diagnostics — {_fd_label}", 2700, 2700)
    _c_filt.Divide(3, 3)
    _fk = []

    # ── Pad 1: Vertex X vs Y ─────────────────────────────────────────────────
    _c_filt.cd(1); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_vxy = root.TH2F(f"h_filt_vxy_{_fd_name}",
                        f"Vertex X vs Y ({_fd_label});Vtx X (mm);Vtx Y (mm)",
                        64, 0.0, 256.0, 64, 0.0, 256.0)
    _h_vxy.SetDirectory(0)
    for _fx, _fy in zip(_fd['vtx_x'], _fd['vtx_y']): _h_vxy.Fill(_fx, _fy)
    _h_vxy.Draw("COLZ"); root.gPad.SetLogz(1); _fk.append(_h_vxy)
    for _p in _AV_BOX: _fline(_fk, *_p, _AV_LC)
    _fline(_fk, 0.0, _BEAM_Y_MIN, 256.0, _BEAM_Y_MIN, _BZ_LC)
    _fline(_fk, 0.0, _BEAM_Y_MAX, 256.0, _BEAM_Y_MAX, _BZ_LC)

    # ── Pad 2: Vertex Z vs End Z ─────────────────────────────────────────────
    _c_filt.cd(2); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_vzez = root.TH2F(f"h_filt_vzez_{_fd_name}",
                         f"Vertex Z vs End Z ({_fd_label});Vtx Z (mm);End Z (mm)",
                         100, 0.0, 500.0, 100, 0.0, 500.0)
    _h_vzez.SetDirectory(0)
    for _fvz, _fez in zip(_fd['vtx_z'], _fd['ez']): _h_vzez.Fill(_fvz, _fez)
    _h_vzez.Draw("COLZ"); root.gPad.SetLogz(1); _fk.append(_h_vzez)
    _fline(_fk, _AV_Z_MIN, 0.0, _AV_Z_MIN, 500.0, _AV_LC)
    _fline(_fk, _AV_Z_MAX, 0.0, _AV_Z_MAX, 500.0, _AV_LC)
    _fline(_fk, 0.0, _AV_Z_MIN, 500.0, _AV_Z_MIN, _AV_LC)
    _fline(_fk, 0.0, _AV_Z_MAX, 500.0, _AV_Z_MAX, _AV_LC)

    # ── Pad 3: End X vs Y ────────────────────────────────────────────────────
    _c_filt.cd(3); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_exy = root.TH2F(f"h_filt_exy_{_fd_name}",
                        f"End X vs Y ({_fd_label});End X (mm);End Y (mm)",
                        64, 0.0, 256.0, 64, 0.0, 256.0)
    _h_exy.SetDirectory(0)
    for _fx, _fy in zip(_fd['ex'], _fd['ey']): _h_exy.Fill(_fx, _fy)
    _h_exy.Draw("COLZ"); root.gPad.SetLogz(1); _fk.append(_h_exy)
    for _p in _AV_BOX: _fline(_fk, *_p, _AV_LC)
    _fline(_fk, 0.0, VolumeBoundaries.BEAM_ZONE_MIN.value,
           256.0, VolumeBoundaries.BEAM_ZONE_MIN.value, _BZ_LC, sty=1)
    _fline(_fk, 0.0, VolumeBoundaries.BEAM_ZONE_MAX.value,
           256.0, VolumeBoundaries.BEAM_ZONE_MAX.value, _BZ_LC, sty=1)

    # ── Pad 4: Start X vs Y ──────────────────────────────────────────────────
    _c_filt.cd(4); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_sxy = root.TH2F(f"h_filt_sxy_{_fd_name}",
                        f"Start X vs Y ({_fd_label});Start X (mm);Start Y (mm)",
                        64, 0.0, 256.0, 64, 0.0, 256.0)
    _h_sxy.SetDirectory(0)
    for _fx, _fy in zip(_fd['sx'], _fd['sy']): _h_sxy.Fill(_fx, _fy)
    _h_sxy.Draw("COLZ"); root.gPad.SetLogz(1); _fk.append(_h_sxy)
    for _p in _AV_BOX: _fline(_fk, *_p, _AV_LC)

    # ── Pad 5: Phi (1D) ──────────────────────────────────────────────────────
    _c_filt.cd(5); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_phi = root.TH1F(f"h_filt_phi_{_fd_name}",
                        f"Phi ({_fd_label});#phi (deg);Counts", 72, -180.0, 180.0)
    _h_phi.SetDirectory(0)
    for _p in _fd['phi']: _h_phi.Fill(_p)
    _h_phi.SetLineColor(_fd_col); _h_phi.SetLineWidth(2)
    _h_phi.Draw("HIST"); _fk.append(_h_phi)
    _h_phi.GetXaxis().SetRangeUser(-180.0, 180.0)
    _ymax_phi = (_h_phi.GetMaximum() * 1.15) or 1.0
    _h_phi.SetMaximum(_ymax_phi)
    for _xv in [-_PHI_EXCL_HI, -_PHI_EXCL_LO, _PHI_EXCL_LO, _PHI_EXCL_HI]:
        _fline(_fk, _xv, 0.0, _xv, _ymax_phi, _PHI_LC)

    # ── Pad 6: End Y (1D) ────────────────────────────────────────────────────
    _c_filt.cd(6); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_ey = root.TH1F(f"h_filt_ey_{_fd_name}",
                       f"End Y ({_fd_label}) [orange=strict beam excl 122-132];End Y (mm);Counts",
                       64, 0.0, 256.0)
    _h_ey.SetDirectory(0)
    for _y in _fd['ey']: _h_ey.Fill(_y)
    _h_ey.SetLineColor(_fd_col); _h_ey.SetLineWidth(2)
    _h_ey.Draw("HIST"); _fk.append(_h_ey)
    _ymax_ey = (_h_ey.GetMaximum() * 1.15) or 1.0
    _h_ey.SetMaximum(_ymax_ey)
    _fline(_fk, float(VolumeBoundaries.BEAM_ZONE_MIN.value), 0.0,
           float(VolumeBoundaries.BEAM_ZONE_MIN.value), _ymax_ey, _BZ_LC, sty=1)
    _fline(_fk, float(VolumeBoundaries.BEAM_ZONE_MAX.value), 0.0,
           float(VolumeBoundaries.BEAM_ZONE_MAX.value), _ymax_ey, _BZ_LC, sty=1)

    # ── Pad 7: Vertex Y (1D) ─────────────────────────────────────────────────
    _c_filt.cd(7); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_vy = root.TH1F(f"h_filt_vy_{_fd_name}",
                       f"Vertex Y ({_fd_label}) [orange=relaxed beam zone 118-136];Vtx Y (mm);Counts",
                       64, 0.0, 256.0)
    _h_vy.SetDirectory(0)
    for _y in _fd['vtx_y']: _h_vy.Fill(_y)
    _h_vy.SetLineColor(_fd_col); _h_vy.SetLineWidth(2)
    _h_vy.Draw("HIST"); _fk.append(_h_vy)
    _ymax_vy = (_h_vy.GetMaximum() * 1.15) or 1.0
    _h_vy.SetMaximum(_ymax_vy)
    _fline(_fk, _BEAM_Y_MIN, 0.0, _BEAM_Y_MIN, _ymax_vy, _BZ_LC)
    _fline(_fk, _BEAM_Y_MAX, 0.0, _BEAM_Y_MAX, _ymax_vy, _BZ_LC)

    # ── Pad 8: Charge vs Track Length (with cutg) ────────────────────────────
    _c_filt.cd(8); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_cl2 = root.TH2F(f"h_filt_cl_{_fd_name}",
                        f"{_fd_label}: Charge vs Track Length;Track Length (mm);Total Charge (a.u.)",
                        200, 0.0, 400.0, 200, 0.0, 100000.0)
    _h_cl2.SetDirectory(0)
    for _tl, _tc in zip(_fd['track_len'], _fd['charge']): _h_cl2.Fill(_tl, _tc)
    _h_cl2.Draw("COLZ"); root.gPad.SetLogz(1); _fk.append(_h_cl2)
    _fd_cutg.Draw("same")

    # ── Pad 9: Track Length (1D) ─────────────────────────────────────────────
    _c_filt.cd(9); root.gPad.SetGridx(1); root.gPad.SetGridy(1)
    _h_tl = root.TH1F(f"h_filt_tl_{_fd_name}",
                       f"Track Length ({_fd_label});Track Length (mm);Counts",
                       100, 0.0, 400.0)
    _h_tl.SetDirectory(0)
    for _tl in _fd['track_len']: _h_tl.Fill(_tl)
    _h_tl.SetLineColor(_fd_col); _h_tl.SetLineWidth(2)
    _h_tl.Draw("HIST"); _fk.append(_h_tl)

    _c_filt.Update()
    if not _save:
        _c_filt.WaitPrimitive()
    os.makedirs(images_dir, exist_ok=True)
    _c_filt_png  = os.path.join(images_dir, f"filter_diagnostics_{_fd_name}{_tag}.png")
    _c_filt_root = os.path.join(images_dir, f"filter_diagnostics_{_fd_name}{_tag}.root")
    _c_filt.SaveAs(_c_filt_png)
    _fout_filt = root.TFile(_c_filt_root, "RECREATE")
    _c_filt.Write()
    _fout_filt.Close()
    print(f"[Filter diagnostics saved] {_c_filt_png}")
    print(f"[Filter diagnostics ROOT]  {_c_filt_root}")
    for _hh in _fk:
        if hasattr(_hh, 'SetDirectory'):
            _hh.SetDirectory(0)
    del _fk[:]
    _c_filt.Close()

# ─────────────────────────────────────────────────────────────────────────────
# 2-pad canvas: E* vs theta_CM — RANSAC (left) and GMM+resp (right)
# ─────────────────────────────────────────────────────────────────────────────
_c2 = root.TCanvas("c_ex_2pad", "Excitation Energy vs CM Angle (2-pad)", 2000, 800)
_c2.Divide(2, 1)
_c2_keep = []

_ex2_datasets = [
    ("RANSAC",    ransac_excitation),
    ("GMM+resp",  reg_comb_excitation),
]
for _pad2, (_lbl2, _ex2_data) in enumerate(_ex2_datasets, start=1):
    _c2.cd(_pad2)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _h2 = root.TH2F(
        f"h_ex2_{_pad2}",
        f"{_lbl2}: E* vs #theta_{{CM}};#theta_{{CM}} (deg);E* (MeV)",
        20, 0, 6, 48, _EX_RANGE_LO, _EX_RANGE_HI,
    )
    _h2.SetDirectory(0)
    for _tc2, _ev2 in zip(_ex2_data[0], _ex2_data[1]):
        if _EX_RANGE_LO < _ev2 < _EX_RANGE_HI:
            _h2.Fill(_tc2, _ev2)
    _h2.Draw("COLZ")
    root.gPad.SetLogz(1)
    _c2_keep.append(_h2)
    for _ev, _hl in _hlines:
        _hl.Draw("SAME")
    _draw_ex_legend()

_c2.Update()
if not _save:
    _c2.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
_c2_png  = os.path.join(images_dir, f"kinematics_ex_2pad{_tag}.png")
_c2_root = os.path.join(images_dir, f"kinematics_ex_2pad{_tag}.root")
_c2.SaveAs(_c2_png)
_fout2 = root.TFile(_c2_root, "RECREATE")
_c2.Write()   # saves the full 2-pad canvas — open in ROOT browser to see both pads together
_fout2.Close()
print(f"[2-pad canvas saved] {_c2_png}")
print(f"[2-pad ROOT saved]   {_c2_root}")

# ─────────────────────────────────────────────────────────────────────────────
# 4-pad combined canvas: (a,b) E* vs θ_CM 2D histos  +  (c,d) gain ratios
# ─────────────────────────────────────────────────────────────────────────────
_c4 = root.TCanvas("c_combined_4pad", "E* vs CM + Gain Ratio (4-pad)", 2400, 1600)
_c4.Divide(2, 2)
_c4_keep = []

# Pads (a) and (b): E* vs θ_CM — RANSAC (top-left) and GMM+resp (top-right)
for _p4, (_l4, _d4) in enumerate([
    ("RANSAC",   ransac_excitation),
    ("GMM+resp", reg_comb_excitation),
], start=1):
    _c4.cd(_p4)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _h4 = root.TH2F(
        f"h_c4_ex_{_p4}",
        f"{_l4}: E* vs #theta_{{CM}};#theta_{{CM}} (deg);E* (MeV)",
        20, 0, 6, 48, _EX_RANGE_LO, _EX_RANGE_HI,
    )
    _h4.SetDirectory(0)
    for _tc4, _ev4 in zip(_d4[0], _d4[1]):
        if _EX_RANGE_LO < _ev4 < _EX_RANGE_HI:
            _h4.Fill(_tc4, _ev4)
    _h4.Draw("COLZ")
    root.gPad.SetLogz(1)
    _c4_keep.append(_h4)
    for _ev, _hl in _hlines:
        _hl.Draw("SAME")
    _draw_ex_legend()

# Pads (c) and (d): gain ratio histograms from _pg_keep
_gain_hists = [_o for _o in _pg_keep if isinstance(_o, root.TH1F) and _o.GetName().startswith("h_gain_")]
for _p4g, _hg4 in enumerate(_gain_hists, start=3):
    _c4.cd(_p4g)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    _hg4.SetStats(0)
    _hg4.Draw("E1")
    _zl4 = root.TLine(_pg_tcm_lo, 0.0, _pg_tcm_hi_adj, 0.0)
    _zl4.SetLineColor(root.kRed); _zl4.SetLineStyle(2); _zl4.SetLineWidth(2)
    _zl4.Draw()
    _c4_keep.append(_zl4)

_c4.Update()
_c4_root = os.path.join(images_dir, f"combined_ex_gain{_tag}.root")
_c4_png  = os.path.join(images_dir, f"combined_ex_gain{_tag}.png")
_c4.SaveAs(_c4_png)
_fout4 = root.TFile(_c4_root, "RECREATE")
_c4.Write()
_fout4.Close()
print(f"[4-pad combined saved] {_c4_png}")
print(f"[4-pad combined ROOT]  {_c4_root}")
for _hh4 in _c4_keep:
    if hasattr(_hh4, 'SetDirectory'):
        _hh4.SetDirectory(0)
del _c4_keep[:]
_c4.Close()

for _hh in _c2_keep:
    _hh.SetDirectory(0)
del _c2_keep[:]
_c2.Close()

for _h in _te_keep:
    _h.SetDirectory(0)
del _te_keep[:]
del _kine_keep[:]
del _hline_keep[:]
c.Close()
c_ov.Close()
root.gROOT.GetListOfFiles().Clear()
root.gROOT.GetListOfCanvases().Clear()
