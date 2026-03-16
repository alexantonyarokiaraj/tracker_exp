#!/usr/bin/env python3
"""Read all output ROOT files and produce kinematics plots."""
import glob
import math
import os
import sys
import ROOT as root
from libraries import FileNames, VolumeBoundaries, RunParameters

# Parse optional argument: group=True (default) or group=False
_single_group = True
for _arg in sys.argv[1:]:
    if _arg.lower() in ("group=false", "group=0"):
        _single_group = False
    elif _arg.lower() in ("group=true", "group=1"):
        _single_group = True
print(f"Group filter: single_group={_single_group}")

root.gROOT.SetBatch(False)
root.TH1.AddDirectory(False)

_NAN = -999.0

# --- Active volume bounds (10 mm border on all sides) ---
_AV_X_MIN, _AV_X_MAX = 10.0, 246.0   # x: 0+10 to 256-10 mm
_AV_Y_MIN, _AV_Y_MAX = 10.0, 246.0   # y: 0+10 to 256-10 mm
_AV_Z_MIN, _AV_Z_MAX = 10.0, 490.0   # z: 0+10 to 500-10 mm
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

output_dir = FileNames.OUTPUT_DIR.value
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

# Collect all output files
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

# Containers: (theta, range) and (total_charge, track_length)
ransac_range_angle = ([], [])       # (theta, range)
reg_range_angle    = ([], [])
reg_comb_range_angle = ([], [])

ransac_charge_length = ([], [], [])   # (track_length, total_charge, theta)
reg_charge_length    = ([], [], [])
reg_comb_charge_length = ([], [], [])

for i in range(n_entries):
    chain.GetEntry(i)
    status = chain.event_status if hasattr(chain, 'event_status') else 0
    if status != 0:
        continue

    event_id = int(chain.event_id) if hasattr(chain, 'event_id') else i

    # per-event tracking for the diagnostic print
    _ransac_passed_88_90 = []   # (theta,) for passing RANSAC tracks in 88-90°
    _reg_passed = False         # any REG scatter track passed all filters+cutg

    # --- RANSAC ---
    n_grp = chain.ransac_n_vtx_groups
    vtx_mult = chain.ransac_vtx_mult
    if _single_group and n_grp != 1:
        pass  # skip RANSAC for this event
    else:
      for g in range(n_grp):
        if vtx_mult[g] != 1:
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
            cp_y = list(chain.ransac_cp_y[g][t])

            if theta == _NAN or rng == _NAN:
                continue
            if not _passes_filters(vx, vy, vz, sx, sy, sz, ex, ey, ez, phi):
                continue

            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0 and _cutg.IsInside(track_len, total_charge):
                ransac_range_angle[0].append(theta)
                ransac_range_angle[1].append(rng)
                ransac_charge_length[0].append(track_len)
                ransac_charge_length[1].append(total_charge)
                ransac_charge_length[2].append(theta)
                if 88.0 <= theta <= 90.0:
                    _ransac_passed_88_90.append(theta)

    # --- REG scatter ---
    n_grp = chain.reg_n_vtx_groups
    vtx_mult = chain.reg_vtx_mult
    if _single_group and n_grp != 1:
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
            cp_y = list(chain.reg_cp_y[g][t])

            if theta == _NAN or rng == _NAN:
                continue
            if not _passes_filters(vx, vy, vz, sx, sy, sz, ex, ey, ez, phi):
                continue

            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0 and _cutg.IsInside(track_len, total_charge):
                reg_range_angle[0].append(theta)
                reg_range_angle[1].append(rng)
                reg_charge_length[0].append(track_len)
                reg_charge_length[1].append(total_charge)
                reg_charge_length[2].append(theta)
                _reg_passed = True

            # --- REG combined fit ---
            theta2 = chain.reg_theta2[g][t]
            rng_c  = chain.reg_range_comb_mm[g][t]
            phi2   = chain.reg_phi2[g][t]
            vx2    = chain.reg_vertex2_x[g][t]
            vy2    = chain.reg_vertex2_y[g][t]
            vz2    = chain.reg_vertex2_z[g][t]
            cp_y_c = list(chain.reg_cp_comb_y[g][t])

            if theta2 == _NAN or rng_c == _NAN:
                continue
            if not _passes_filters(vx2, vy2, vz2, sx, sy, sz, ex, ey, ez, phi2):
                continue

            total_charge_c = sum(cp_y_c) if len(cp_y_c) > 0 else 0.0
            if total_charge_c > 0 and _cutg.IsInside(track_len, total_charge_c):
                reg_comb_range_angle[0].append(theta2)
                reg_comb_range_angle[1].append(rng_c)
                reg_comb_charge_length[0].append(track_len)
                reg_comb_charge_length[1].append(total_charge_c)
                reg_comb_charge_length[2].append(theta2)
                if theta2 < 40 and rng_c < 40:
                    _run = int(chain.run_number) if hasattr(chain, 'run_number') else -1
                    print(f"[COMB-LOW] run={_run} event={event_id} theta2={theta2:.2f} range_comb={rng_c:.2f}")

    # --- Diagnostic: RANSAC passed (88-90°) but GMM did not ---
    if _ransac_passed_88_90 and not _reg_passed:
        for th in _ransac_passed_88_90:
            print(f"[DIAG] event_id={event_id}  RANSAC passed (theta={th:.2f}°)  GMM did not pass")


_THETA_LO, _THETA_HI = 80, 100  # projection slice in degrees

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
c.Close()
c_ov.Close()
root.gROOT.GetListOfFiles().Clear()
root.gROOT.GetListOfCanvases().Clear()
