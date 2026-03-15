#!/usr/bin/env python3
"""Read output ROOT file and print the nested tree structure."""
import argparse
import os
import ROOT as root
from libraries import FileNames, Optimize

_ALPHA = Optimize.ALPHA.value

parser = argparse.ArgumentParser(description="Print track analysis results in tree format.")
parser.add_argument("run_info", help="Format: runnumber@start_event_end_event (e.g. 53@0_9999)")
args = parser.parse_args()

run_number, event_range = args.run_info.split("@")
start_event, end_event = event_range.split("_")
filename = os.path.join(
    FileNames.OUTPUT_DIR.value,
    f"output_run_{run_number}_{start_event}_{end_event}.root",
)

f = root.TFile.Open(filename, "READ")
if not f or f.IsZombie():
    print(f"ERROR: cannot open {filename}")
    raise SystemExit(1)

tree = f.Get("track_data")
if not tree:
    print("ERROR: TTree 'track_data' not found")
    raise SystemExit(1)

_NAN = -999.0

def _fmt(val, unit="", prec=1):
    if val is None or val == _NAN:
        return "---"
    return f"{val:.{prec}f}{unit}"

def _print_track(prefix, grp_i, trk_j, entry, indent, is_reg=False):
    """Print one track's parameters."""
    tid = getattr(entry, f"{prefix}_track_id")[grp_i][trk_j]
    bid = getattr(entry, f"{prefix}_closest_beam_id")[grp_i][trk_j]
    bdist = getattr(entry, f"{prefix}_closest_beam_dist")[grp_i][trk_j]
    theta = getattr(entry, f"{prefix}_theta")[grp_i][trk_j]
    phi = getattr(entry, f"{prefix}_phi")[grp_i][trk_j]
    rng = getattr(entry, f"{prefix}_range_mm")[grp_i][trk_j]
    eng = getattr(entry, f"{prefix}_energy_keV")[grp_i][trk_j]
    r2d = getattr(entry, f"{prefix}_r2d")[grp_i][trk_j]
    dz = getattr(entry, f"{prefix}_delta_z")[grp_i][trk_j]
    vx = getattr(entry, f"{prefix}_vertex_x")[grp_i][trk_j]
    vy = getattr(entry, f"{prefix}_vertex_y")[grp_i][trk_j]
    vz = getattr(entry, f"{prefix}_vertex_z")[grp_i][trk_j]
    tsx = getattr(entry, f"{prefix}_trunc_start_x")[grp_i][trk_j]
    tsy = getattr(entry, f"{prefix}_trunc_start_y")[grp_i][trk_j]
    tsz = getattr(entry, f"{prefix}_trunc_start_z")[grp_i][trk_j]
    tex = getattr(entry, f"{prefix}_trunc_end_x")[grp_i][trk_j]
    tey = getattr(entry, f"{prefix}_trunc_end_y")[grp_i][trk_j]
    tez = getattr(entry, f"{prefix}_trunc_end_z")[grp_i][trk_j]
    sx = getattr(entry, f"{prefix}_start_x")[grp_i][trk_j]
    sy = getattr(entry, f"{prefix}_start_y")[grp_i][trk_j]
    sz = getattr(entry, f"{prefix}_start_z")[grp_i][trk_j]
    ex = getattr(entry, f"{prefix}_end_x")[grp_i][trk_j]
    ey = getattr(entry, f"{prefix}_end_y")[grp_i][trk_j]
    ez = getattr(entry, f"{prefix}_end_z")[grp_i][trk_j]
    cp_ys = list(getattr(entry, f"{prefix}_cp_ys")[grp_i][trk_j])
    cp_len = len(cp_ys)
    if cp_len > 0:
        max_charge = max(cp_ys)
        alpha_charge = _ALPHA * max_charge
        cp_info = f"bins={cp_len}  peak_charge={max_charge:.1f}  alpha_charge({_ALPHA:.2f})={alpha_charge:.1f}"
    else:
        cp_info = "bins=0"

    p = indent
    print(f"{p}Track {trk_j}  (cluster_id={tid}, beam_id={bid}, beam_dist={_fmt(bdist, ' mm')})")
    print(f"{p}  Scatter fit:  theta={_fmt(theta, chr(176))}  phi={_fmt(phi, chr(176))}  range={_fmt(rng, ' mm')}  energy={_fmt(eng, ' keV')}")
    print(f"{p}  r2d={_fmt(r2d, ' mm')}  delta_z={_fmt(dz, ' mm')}  CP: {cp_info}")
    print(f"{p}  Vertex:      ({_fmt(vx)}, {_fmt(vy)}, {_fmt(vz)})")
    print(f"{p}  Trunc start: ({_fmt(tsx)}, {_fmt(tsy)}, {_fmt(tsz)})")
    print(f"{p}  Trunc end:   ({_fmt(tex)}, {_fmt(tey)}, {_fmt(tez)})")
    print(f"{p}  Full start:  ({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})")
    print(f"{p}  Full end:    ({_fmt(ex)}, {_fmt(ey)}, {_fmt(ez)})")

    if is_reg:
        t2 = getattr(entry, "reg_theta2")[grp_i][trk_j]
        p2 = getattr(entry, "reg_phi2")[grp_i][trk_j]
        rc = getattr(entry, "reg_range_comb_mm")[grp_i][trk_j]
        ec = getattr(entry, "reg_energy_comb_keV")[grp_i][trk_j]
        r2c = getattr(entry, "reg_r2d_comb")[grp_i][trk_j]
        dzc = getattr(entry, "reg_delta_z_comb")[grp_i][trk_j]
        v2x = getattr(entry, "reg_vertex2_x")[grp_i][trk_j]
        v2y = getattr(entry, "reg_vertex2_y")[grp_i][trk_j]
        v2z = getattr(entry, "reg_vertex2_z")[grp_i][trk_j]
        cpc_ys = list(getattr(entry, "reg_cp_comb_ys")[grp_i][trk_j])
        cpc_len = len(cpc_ys)
        if cpc_len > 0:
            max_charge_c = max(cpc_ys)
            alpha_charge_c = _ALPHA * max_charge_c
            cpc_info = f"bins={cpc_len}  peak_charge={max_charge_c:.1f}  alpha_charge({_ALPHA:.2f})={alpha_charge_c:.1f}"
        else:
            cpc_info = "bins=0"
        print(f"{p}  Combined fit: theta2={_fmt(t2, chr(176))}  phi2={_fmt(p2, chr(176))}  range={_fmt(rc, ' mm')}  energy={_fmt(ec, ' keV')}")
        print(f"{p}  Vertex2:     ({_fmt(v2x)}, {_fmt(v2y)}, {_fmt(v2z)})")
        print(f"{p}  r2d_comb={_fmt(r2c, ' mm')}  delta_z_comb={_fmt(dzc, ' mm')}  CP_comb: {cpc_info}")


n_entries = tree.GetEntries()
print(f"File: {filename}")
print(f"Total entries (events): {n_entries}")
print("=" * 80)

for i_entry in range(n_entries):
    tree.GetEntry(i_entry)
    run = tree.run_number
    evid = tree.event_id
    status = tree.event_status if hasattr(tree, 'event_status') else 0
    if status != 0:
        print(f"\nEvent {evid}  (run {run})  *** FAILED")
        continue
    print(f"\nEvent {evid}  (run {run})")

    for method, prefix, is_reg in [("REG", "reg", True), ("RANSAC", "ransac", False)]:
        n_grp = getattr(tree, f"{prefix}_n_vtx_groups")
        vtx_mult = getattr(tree, f"{prefix}_vtx_mult")
        print(f"  {method}:  {n_grp} vertex group(s)")
        for g in range(n_grp):
            mult = vtx_mult[g]
            print(f"    Vertex Group {g}  (multiplicity={mult})")
            for t in range(mult):
                _print_track(prefix, g, t, tree, "      ", is_reg=is_reg)

print("\n" + "=" * 80)
print("Done.")
