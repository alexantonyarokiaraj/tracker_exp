#!/usr/bin/env python3
"""Read all output ROOT files and produce kinematics plots."""
import glob
import math
import os
import ROOT as root
from libraries import FileNames

root.gROOT.SetBatch(False)

_NAN = -999.0
output_dir = FileNames.OUTPUT_DIR.value
images_dir = FileNames.IMAGES_DIR.value

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

    # --- RANSAC ---
    n_grp = chain.ransac_n_vtx_groups
    vtx_mult = chain.ransac_vtx_mult
    if n_grp != 1 or vtx_mult[0] != 1:
        pass  # skip RANSAC for this event
    else:
      for g in range(n_grp):
        for t in range(vtx_mult[g]):
            theta = chain.ransac_theta[g][t]
            rng   = chain.ransac_range_mm[g][t]
            sx = chain.ransac_start_x[g][t]
            sy = chain.ransac_start_y[g][t]
            sz = chain.ransac_start_z[g][t]
            ex = chain.ransac_end_x[g][t]
            ey = chain.ransac_end_y[g][t]
            ez = chain.ransac_end_z[g][t]
            cp_y = list(chain.ransac_cp_y[g][t])

            if theta == _NAN or rng == _NAN:
                continue
            ransac_range_angle[0].append(theta)
            ransac_range_angle[1].append(rng)

            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0:
                ransac_charge_length[0].append(track_len)
                ransac_charge_length[1].append(total_charge)
                ransac_charge_length[2].append(theta)

    # --- REG scatter ---
    n_grp = chain.reg_n_vtx_groups
    vtx_mult = chain.reg_vtx_mult
    if n_grp != 1 or vtx_mult[0] != 1:
        pass  # skip REG for this event
    else:
      for g in range(n_grp):
        for t in range(vtx_mult[g]):
            theta = chain.reg_theta[g][t]
            rng   = chain.reg_range_mm[g][t]
            sx = chain.reg_start_x[g][t]
            sy = chain.reg_start_y[g][t]
            sz = chain.reg_start_z[g][t]
            ex = chain.reg_end_x[g][t]
            ey = chain.reg_end_y[g][t]
            ez = chain.reg_end_z[g][t]
            cp_y = list(chain.reg_cp_y[g][t])

            if theta == _NAN or rng == _NAN:
                continue
            reg_range_angle[0].append(theta)
            reg_range_angle[1].append(rng)

            track_len = math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)
            total_charge = sum(cp_y) if len(cp_y) > 0 else 0.0
            if total_charge > 0:
                reg_charge_length[0].append(track_len)
                reg_charge_length[1].append(total_charge)
                reg_charge_length[2].append(theta)

            # --- REG combined fit ---
            theta2 = chain.reg_theta2[g][t]
            rng_c  = chain.reg_range_comb_mm[g][t]
            cp_y_c = list(chain.reg_cp_comb_y[g][t])

            if theta2 == _NAN or rng_c == _NAN:
                continue
            reg_comb_range_angle[0].append(theta2)
            reg_comb_range_angle[1].append(rng_c)

            total_charge_c = sum(cp_y_c) if len(cp_y_c) > 0 else 0.0
            if total_charge_c > 0:
                reg_comb_charge_length[0].append(track_len)
                reg_comb_charge_length[1].append(total_charge_c)
                reg_comb_charge_length[2].append(theta2)


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
        900, 0, 180, 125, 0, 400,
    )
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
    for tl, tc in zip(cl_data[0], cl_data[1]):
        h_cl.Fill(tl, tc)
    h_cl.Draw("COLZ")
    keep.append(h_cl)

    # Col 3: Range projection for theta slice
    c.cd(4 * row + 3)
    root.gPad.SetGridx(1)
    root.gPad.SetGridy(1)
    bin_lo = h_ra.GetXaxis().FindBin(_THETA_LO)
    bin_hi = h_ra.GetXaxis().FindBin(_THETA_HI)
    h_proj_ra = h_ra.ProjectionX(f"h_proj_ra_{row}", 1, h_ra.GetNbinsY())
    h_proj_ra.GetXaxis().SetRangeUser(_THETA_LO, _THETA_HI)
    h_proj_ra.SetTitle(f"{label}: Counts ({_THETA_LO}#minus{_THETA_HI}#circ);#theta (deg);Counts")
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
h_diff_ra.SetTitle("GMM_REG #minus RANSAC: Range vs Lab Angle;#theta (deg);Range (mm)")
h_diff_ra.Add(keep[0], -1)
h_diff_ra.Draw("COLZ")
keep.append(h_diff_ra)

# Col 2: Total Charge vs Track Length diff
c.cd(14)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_cl = keep[5].Clone("h_diff_cl")
h_diff_cl.SetTitle("GMM_REG #minus RANSAC: Total Charge vs Track Length;Track Length (mm);Total Charge (a.u.)")
h_diff_cl.Add(keep[1], -1)
h_diff_cl.Draw("COLZ")
keep.append(h_diff_cl)

# Col 3: Range projection diff
c.cd(15)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_proj_ra = keep[6].Clone("h_diff_proj_ra")
h_diff_proj_ra.SetTitle("GMM_REG #minus RANSAC: Range ({0}#minus{1}#circ);Range (mm);Counts".format(_THETA_LO, _THETA_HI))
h_diff_proj_ra.Add(keep[2], -1)
h_diff_proj_ra.Draw()
keep.append(h_diff_proj_ra)

# Col 4: Charge projection diff
c.cd(16)
root.gPad.SetGridx(1)
root.gPad.SetGridy(1)
h_diff_proj_cl = keep[7].Clone("h_diff_proj_cl")
h_diff_proj_cl.SetTitle("GMM_REG #minus RANSAC: Charge ({0}#minus{1}#circ);Total Charge (a.u.);Counts".format(_THETA_LO, _THETA_HI))
h_diff_proj_cl.Add(keep[3], -1)
h_diff_proj_cl.Draw()
keep.append(h_diff_proj_cl)

c.Update()
c.WaitPrimitive()
os.makedirs(images_dir, exist_ok=True)
c.SaveAs(os.path.join(images_dir, "kinematics.png"))

print(f"Points — RANSAC: {len(ransac_range_angle[0])}, REG: {len(reg_range_angle[0])}, REG_comb: {len(reg_comb_range_angle[0])}")
print(f"Saved to {os.path.join(images_dir, 'kinematics.png')}")
