import sys
import time
import argparse
import os
from dataclasses import dataclass
from libraries import RunParameters, SCAN, DataArray, Optimize, RansacParameters, VolumeBoundaries, Reference, ConversionFactors, FileNames
import ROOT as root
import numpy as np
from ROOT import gROOT, gSystem, TH2F, TTree, TFile, AddressOf, TLine, TMultiGraph, TEllipse, TH1F, TH3F, TNtuple
from helper_functions.functions import (
    dbcluster,
    plot_3d_projections,
    hierarchical_clustering_with_responsibilities,
    merge_beam_clusters_by_z_centroid,
    fit_beam_tracks_pca_constrained_endpoints,
    compute_scattered_track_side,
    relabel_small_clusters_to_noise,
    get_directions,
    find_closest_beam_track,
    get_unique_colors,
    angle_between,
    calculate_phi_angle,
)
from regularize import Regularize
from ransac import find_multiple_lines_ransac
import pandas as pd
from energy import Energy
from scipy import interpolate

@dataclass
class TrackEndpoints3D:
    track_id: int
    start_point_full: np.ndarray  # shape (3,)
    end_point_full: np.ndarray    # shape (3,)
    closest_beam_id: int = -1
    closest_beam_dist_mm: float = float('inf')
    vertex: np.ndarray = None           # closest-approach point on scattered track to beam track
    trunc_start: np.ndarray = None      # start of 40mm-truncated PCA fit
    trunc_end: np.ndarray = None        # end of 40mm-truncated PCA fit
    theta_deg: float = None             # lab angle between beam and scattered track (0-180 deg)
    phi_deg: float = None               # signed azimuthal angle in YZ plane (-180 to 180 deg)
    theta2_deg: float = None            # angle from combined (scatter+beam) PCA fit
    phi2_deg: float = None              # azimuthal from combined fit
    vertex2: np.ndarray = None          # closest-approach vertex using combined fit direction
    range_scat_mm: float = None         # range from scatter-only charge profile
    energy_scat_keV: float = None       # energy from range lookup (scatter fit)
    r2d_scat: float = None              # 2D track density (scatter fit)
    delta_z_scat: float = None          # delta-z along track (scatter fit)
    cp_scat: tuple = None               # (cp_x, cp_y, cp_xs, cp_ys, ran_end, en_end, ran_max, en_max)
    range_comb_mm: float = None         # range from combined charge profile (REG only)
    energy_comb_keV: float = None       # energy from range lookup (combined fit)
    r2d_comb: float = None              # 2D track density (combined fit)
    delta_z_comb: float = None          # delta-z along track (combined fit)
    cp_comb: tuple = None               # same structure as cp_scat

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Process events from a ROOT file.')
parser.add_argument('run_info', nargs='?', default='53@0_9999', help='Format: runnumber@start_event_end_event (default: 53@0_9999)')
args = parser.parse_args()
save_canvas = RunParameters.SAVE_CANVAS.value

# Extract run number and event range
run_number, event_range = args.run_info.split('@')
start_event, end_event = map(int, event_range.split('_'))

sys.path.append(RunParameters.lib_path.value)

# Batch mode: do not show canvases (GUI), just save PNGs.
batch_mode = True
root.gROOT.SetBatch(True)

#########################################################################################
############################LINKING ROOT INTERFACES######################################
#########################################################################################

# C++ Interface Linking
event_red_str1 = RunParameters.lib_path.value + "MEventReduced.cpp"
event_red_str2 = ".L " + RunParameters.lib_path.value + "MEventReduced.cpp+g"

root.gInterpreter.GenerateDictionary(event_red_str1)
gSystem.SetFlagsOpt("-std=c++11");
gROOT.ProcessLine(event_red_str2);

# Importing user defined structures from ROOT
from ROOT import MEventReduced, ReducedData

#########################################################################################
############################DECLARATIONS#################################################
#########################################################################################

# Histogram Declarations
XY = TH2F('XY', 'XY', RunParameters.nbins_x.value, RunParameters.x_start_bin.value, RunParameters.x_end_bin.value, RunParameters.nbins_y.value, RunParameters.y_start_bin.value, RunParameters.y_end_bin.value)
XZ = TH2F('XZ', 'XZ', RunParameters.nbins_x.value, RunParameters.x_start_bin.value, RunParameters.x_end_bin.value, RunParameters.nbins_z.value, RunParameters.z_start_bin.value, RunParameters.z_end_bin.value)
YZ = TH2F('YZ', 'YZ', RunParameters.nbins_y.value, RunParameters.y_start_bin.value, RunParameters.y_end_bin.value, RunParameters.nbins_z.value, RunParameters.z_start_bin.value, RunParameters.z_end_bin.value)
z_proj = TH1F('z_proj', 'z_proj', 14000, 0, 14000)
line_xy = root.TLine
line_xz = root.TLine
line_yz = root.TLine

table = np.loadtxt(RunParameters.lookup_table.value)
filename_ = RunParameters.files_path.value + f'Tree_Run_{run_number.zfill(4)}_Merged.root'
f = root.TFile(filename_)
myTree = f.Get("ACTAR_TTree")
entry = myTree.GetEntries()
missed_pads = np.loadtxt(RunParameters.missing_pads_info.value)
x_pos = missed_pads[:,0]
y_pos = missed_pads[:,1]

# ── Calibration table and range-energy lookup for Energy class ──────────────
calibration_table = pd.read_csv(FileNames.CALIBRATION_PADS.value, sep=" ", header=None)
calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]
_range_energy_df = pd.read_excel(
    FileNames.CONFIG_FILE_EXCEL.value,
    sheet_name=FileNames.RANGE_ENERGY_CONVERSION_SHEET.value,
)
_TABLE = np.array(_range_energy_df[['Range(mm)', 'Energy(keV)']])
_TABLE = _TABLE[_TABLE[:, 0].argsort()]
_TABLE_REVERSED = _TABLE[:, [1, 0]]
_TABLE_REVERSED = _TABLE_REVERSED[_TABLE_REVERSED[:, 0].argsort()]

def range_energy_calculate(range_value):
    range_table = _TABLE[:, 0]
    energy_table = _TABLE[:, 1]
    if range_value <= range_table[0]:
        return round(float(energy_table[0]), 2)
    elif range_value >= range_table[-1]:
        return round(float(energy_table[-1]), 2)
    f_interp = interpolate.interp1d(range_table, energy_table)
    return round(float(f_interp(range_value)), 2)

# ── Output ROOT file with TTree ────────────────────────────────────────────
from array import array as c_array
os.makedirs(FileNames.OUTPUT_DIR.value, exist_ok=True)
out_file = root.TFile(os.path.join(FileNames.OUTPUT_DIR.value, f"output_run_{run_number}_{start_event}_{end_event}.root"), "RECREATE")
out_tree = root.TTree("track_data", "Per-event track analysis results")

buf_run     = c_array('i', [0])
buf_event   = c_array('i', [0])
buf_status  = c_array('i', [0])  # 0 = OK, 1 = exception
out_tree.Branch("run_number",    buf_run,    "run_number/I")
out_tree.Branch("event_id",      buf_event,  "event_id/I")
out_tree.Branch("event_status",  buf_status, "event_status/I")

# Nested branch structure per method:
#   n_vtx_groups                          (int)
#   vtx_mult                              vector<int>           [n_groups]
#   scalar per-track fields               vector<vector<T>>     [group][track]
#   charge profile arrays                 vector<vector<vector<double>>>  [group][track][bin]
_vec_int    = 'vector<int>'
_vec_dbl    = 'vector<double>'
_vvec_int   = 'vector<vector<int>>'
_vvec_dbl   = 'vector<vector<double>>'
_vvvec_dbl  = 'vector<vector<vector<double>>>'

def _make_nested_branches(prefix, tree):
    b = {}
    b['n_vtx_groups'] = c_array('i', [0])
    tree.Branch(f"{prefix}_n_vtx_groups", b['n_vtx_groups'], f"{prefix}_n_vtx_groups/I")
    # Per-group: multiplicity  [n_groups]
    b['vtx_mult'] = root.std.vector('int')()
    tree.Branch(f"{prefix}_vtx_mult", b['vtx_mult'])
    # Per-track int fields  [group][track]
    for name in ['track_id', 'closest_beam_id']:
        b[name] = root.std.vector(root.std.vector('int'))()
        tree.Branch(f"{prefix}_{name}", b[name])
    # Per-track double fields  [group][track]
    for name in [
        'theta', 'phi', 'range_mm', 'energy_keV', 'r2d', 'delta_z',
        'vertex_x', 'vertex_y', 'vertex_z',
        'trunc_start_x', 'trunc_start_y', 'trunc_start_z',
        'trunc_end_x', 'trunc_end_y', 'trunc_end_z',
        'start_x', 'start_y', 'start_z',
        'end_x', 'end_y', 'end_z',
        'closest_beam_dist',
    ]:
        b[name] = root.std.vector(root.std.vector('double'))()
        tree.Branch(f"{prefix}_{name}", b[name])
    # Charge profiles  [group][track][bin]
    for name in ['cp_x', 'cp_y', 'cp_xs', 'cp_ys']:
        b[name] = root.std.vector(root.std.vector(root.std.vector('double')))()
        tree.Branch(f"{prefix}_{name}", b[name])
    return b

ransac_br = _make_nested_branches("ransac", out_tree)
reg_br    = _make_nested_branches("reg", out_tree)

# REG also gets combined-fit branches (same nesting: [group][track])
for name in ['theta2', 'phi2', 'range_comb_mm', 'energy_comb_keV', 'r2d_comb', 'delta_z_comb',
             'vertex2_x', 'vertex2_y', 'vertex2_z']:
    reg_br[name] = root.std.vector(root.std.vector('double'))()
    out_tree.Branch(f"reg_{name}", reg_br[name])
for name in ['cp_comb_x', 'cp_comb_y', 'cp_comb_xs', 'cp_comb_ys']:
    reg_br[name] = root.std.vector(root.std.vector(root.std.vector('double')))()
    out_tree.Branch(f"reg_{name}", reg_br[name])

# # good_events = [2, 19, 21, 33, 34, 37, 44, 46, 48, 49, 51, 57, 59, 60, 66, 68, 70, 75, 78, 79, 81, 84, 88, 90, 91, 92,
#                 95, 101, 108, 112, 114, 122]

# Looping entries from Tree
for entries in myTree:
    if start_event <= entries.data.event <= end_event:  # Check if event is within the specified range
        XY.Reset()
        YZ.Reset()
        XZ.Reset()
        data_points = []
        length = entries.data.CoboAsad.size()
        print("---------------------------")
        print("Event-", entries.data.event)
        print("Timestamp-", entries.data.timestamp)
        print("---------------------------")
        _event_start_time = time.time() 
        data_points = []       
        for x in range(length):
            co = entries.data.CoboAsad[x].globalchannelid >> 11
            if co != 31 and co != 16:
                asad = (entries.data.CoboAsad[x].globalchannelid - (co << 11)) >> 9
                ag = (entries.data.CoboAsad[x].globalchannelid - (co << 11) - (asad << 9)) >> 7
                ch = entries.data.CoboAsad[x].globalchannelid - (co << 11) - (asad << 9) - (ag << 7)
                where = co * RunParameters.NB_ASAD.value * RunParameters.NB_AGET.value * RunParameters.NB_CHANNEL.value + asad * RunParameters.NB_AGET.value * RunParameters.NB_CHANNEL.value + ag * RunParameters.NB_CHANNEL.value + ch
                posX = table[where][4] * RunParameters.x_conversion_factor.value
                posY = table[where][5] * RunParameters.y_conversion_factor.value
                hitlength = entries.data.CoboAsad[int(x)].peakheight.size()
                for y in range(0, hitlength):
                    posZ = entries.data.CoboAsad[int(x)].peaktime[int(y)] * RunParameters.z_conversion_factor.value
                    Qvox = entries.data.CoboAsad[int(x)].peakheight[int(y)]
                    if not entries.data.CoboAsad[x].hasSaturation:                       
                        XY.Fill(posX, posY, Qvox)
                        XZ.Fill(posX, posZ, Qvox)
                        YZ.Fill(posY, posZ, Qvox)        
                        z_proj.Fill(entries.data.CoboAsad[int(x)].peaktime[int(y)])     
                        data_points.append([posX, posY, posZ, Qvox])                                                   
        data_points_3d = np.array(data_points)
        
        # Call HDBSCAN clustering
        if len(data_points_3d) > 0:
          try:
            _t0 = time.time()
            _t_event = _t0
            dbscan_labels, valid_cluster = dbcluster(
                data_points_3d,
                SCAN.MIN_CLUSTER_SIZE.value,
                SCAN.MIN_SAMPLES.value
            )
            
            # Add HDBSCAN labels to the data array
            data_points_3d = np.column_stack((data_points_3d, dbscan_labels))
            
            # Call hierarchical clustering with GMM, reusing the DBSCAN labels already computed above
            gmm_labels, n_comp, responsibilities, _, elapsed_dbscan, elapsed_gmm = hierarchical_clustering_with_responsibilities(data_points_3d[:, :4], max_components=3, dbscan_labels=dbscan_labels)
            _t_dbscan_done = time.time()
            print(f"  [timing] DBSCAN:       {elapsed_dbscan:.2f}s")
            print(f"  [timing] GMM:          {sum(elapsed_gmm):.2f}s")
            
            # Add GMM labels to the data array
            data_points_3d = np.column_stack((data_points_3d, gmm_labels))
            
            # Call Regularize to merge GMM labels
            _t1 = time.time()
            reg = Regularize(data_array=data_points_3d, threshold=Optimize.P_VALUE.value, merge_type='p_value', merge_algorithm='gmm')
            final_clusters = reg.merge_labels()
            _t_reg_done = time.time()
            print(f"  [timing] REG p-value:  {_t_reg_done - _t1:.2f}s")
            
            # Add regularized labels to the data array
            data_points_3d = np.column_stack((data_points_3d, final_clusters))
            
            # Call RANSAC on non-noise points only
            dbscan_labels_full = data_points_3d[:, DataArray.DBSCAN.value].astype(int)
            non_noise_mask = dbscan_labels_full != -1
            non_noise_data = data_points_3d[non_noise_mask]
            
            ransac_labels_full = -1 * np.ones(len(data_points_3d), dtype=int)
            
            if len(non_noise_data) > 0:
                _t_ransac = time.time()
                ransac_labels_non_noise, fitted_models = find_multiple_lines_ransac(
                    non_noise_data[:, :3],  # Use only X, Y, Z columns
                    max_lines=RansacParameters.MAX_LINES.value,
                    residual_threshold=RansacParameters.RESIDUAL_THRESHOLD.value,
                    n_iterations=RansacParameters.N_ITERATIONS.value,
                    min_samples=RansacParameters.MIN_SAMPLES.value,
                    min_inliers=RansacParameters.MIN_INLIERS.value
                )
                # Map ransac labels back to full array
                ransac_labels_full[non_noise_mask] = ransac_labels_non_noise
                print(f"  [timing] RANSAC:       {time.time() - _t_ransac:.2f}s")
            
            # Add RANSAC labels to the data array
            data_points_3d = np.column_stack((data_points_3d, ransac_labels_full))
            
            # Classify REGULARIZED clusters as beam or scattered tracks
            regularized_labels = data_points_3d[:, DataArray.REGULARIZED.value].astype(int)
            regularized_track_type = np.zeros(len(data_points_3d), dtype=int)  # 0=beam, 1=scattered
            
            unique_reg_clusters = np.unique(regularized_labels[regularized_labels != -1])
            for cluster_label in unique_reg_clusters:
                cluster_mask = regularized_labels == cluster_label
                y_vals = data_points_3d[cluster_mask, DataArray.Y.value]
                centroid_y = np.mean(y_vals)
                n_inside = int(np.sum((y_vals >= VolumeBoundaries.BEAM_ZONE_MIN.value) & (y_vals <= VolumeBoundaries.BEAM_ZONE_MAX.value)))
                n_outside = int(np.sum((y_vals < VolumeBoundaries.BEAM_ZONE_MIN.value) | (y_vals > VolumeBoundaries.BEAM_ZONE_MAX.value)))
                
                # If centroid Y is outside beam zone, mark as scattered (1)
                if centroid_y < VolumeBoundaries.BEAM_ZONE_MIN.value or centroid_y > VolumeBoundaries.BEAM_ZONE_MAX.value:
                    regularized_track_type[cluster_mask] = 1
                # Guard against over-merged clusters: if significant points exist both
                # inside and outside the beam zone, the p-value merge likely absorbed a
                # scatter track into a beam cluster — mark as scattered
                elif (n_inside >= Optimize.GMM_SPLIT_MIN_BEAM_PTS.value and
                      n_outside >= Optimize.GMM_SPLIT_MIN_OUTSIDE_PTS.value):
                    regularized_track_type[cluster_mask] = 1
                    print(f"    [diag] REG cluster {cluster_label}: reclassified beam->scattered "
                          f"(n_inside={n_inside}, n_outside={n_outside}, centroid_y={centroid_y:.1f})")
            
            # Add regularized track type to the data array
            data_points_3d = np.column_stack((data_points_3d, regularized_track_type))
            
            # Classify RANSAC clusters as beam or scattered tracks
            ransac_labels = data_points_3d[:, DataArray.RANSAC.value].astype(int)
            ransac_track_type = np.zeros(len(data_points_3d), dtype=int)  # 0=beam, 1=scattered
            
            unique_ransac_clusters = np.unique(ransac_labels[(ransac_labels != -1) & (ransac_labels != -20)])
            for cluster_label in unique_ransac_clusters:
                cluster_mask = ransac_labels == cluster_label
                centroid_y = np.mean(data_points_3d[cluster_mask, DataArray.Y.value])
                
                # If centroid Y is outside beam zone, mark as scattered (1)
                if centroid_y < VolumeBoundaries.BEAM_ZONE_MIN.value or centroid_y > VolumeBoundaries.BEAM_ZONE_MAX.value:
                    ransac_track_type[cluster_mask] = 1
            
            # Add RANSAC track type to the data array
            data_points_3d = np.column_stack((data_points_3d, ransac_track_type))

            z_threshold = Optimize.BEAM_Z_MERGE_THRESHOLD_MM.value
            regularized_beam_merged, reg_z_before, reg_z_after, _ = merge_beam_clusters_by_z_centroid(
                data_points_3d,
                label_column=DataArray.REGULARIZED,
                track_type_column=DataArray.REGULARIZED_TRACK_TYPE,
                z_threshold=z_threshold,
                noise_labels=(-1,),
            )
            data_points_3d = np.column_stack((data_points_3d, regularized_beam_merged))

            ransac_beam_merged, ransac_z_before, ransac_z_after, _ = merge_beam_clusters_by_z_centroid(
                data_points_3d,
                label_column=DataArray.RANSAC,
                track_type_column=DataArray.RANSAC_TRACK_TYPE,
                z_threshold=z_threshold,
                noise_labels=(-1, -20),
            )
            data_points_3d = np.column_stack((data_points_3d, ransac_beam_merged))

            # Remove small clusters in the *beam merged* label columns
            min_cluster_size = int(RunParameters.MIN_CLUSTER_SIZE_BEAM_MERGED.value)

            reg_bm_filtered = relabel_small_clusters_to_noise(
                data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value],
                min_cluster_size=min_cluster_size,
                noise_labels=(-1, -20),
                output_noise_label=-1,
            )
            ransac_bm_filtered = relabel_small_clusters_to_noise(
                data_points_3d[:, DataArray.RANSAC_BEAM_MERGED.value],
                min_cluster_size=min_cluster_size,
                noise_labels=(-1, -20),
                output_noise_label=-1,
            )

            data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value] = reg_bm_filtered
            data_points_3d[:, DataArray.RANSAC_BEAM_MERGED.value] = ransac_bm_filtered

            regularized_side = compute_scattered_track_side(
                data_points_3d,
                label_column=DataArray.REGULARIZED_BEAM_MERGED,
                track_type_column=DataArray.REGULARIZED_TRACK_TYPE,
                beam_zone_max=VolumeBoundaries.BEAM_ZONE_MAX.value,
                noise_labels=(-1,),
            )
            ransac_side = compute_scattered_track_side(
                data_points_3d,
                label_column=DataArray.RANSAC_BEAM_MERGED,
                track_type_column=DataArray.RANSAC_TRACK_TYPE,
                beam_zone_max=VolumeBoundaries.BEAM_ZONE_MAX.value,
                noise_labels=(-1, -20),
            )

            data_points_3d = np.column_stack((data_points_3d, regularized_side, ransac_side))
    
            # Additional scattered-track merging using cdist (done separately for above/below)
            # Output labels go into a new column: DataArray.RANSAC_CDIST
            ransac_cdist_labels = data_points_3d[:, DataArray.RANSAC_BEAM_MERGED.value].astype(int).copy()

            ransac_track_types = data_points_3d[:, DataArray.RANSAC_TRACK_TYPE.value].astype(int)
            ransac_side_values = data_points_3d[:, DataArray.RANSAC_SIDE.value].astype(int)
            ransac_bm_labels_for_merge = data_points_3d[:, DataArray.RANSAC_BEAM_MERGED.value].astype(int)

            scattered_mask = ransac_track_types == 1
            valid_label_mask = (ransac_bm_labels_for_merge != -1) & (ransac_bm_labels_for_merge != -20)
            above_mask = scattered_mask & valid_label_mask & (ransac_side_values == 1)
            below_mask = scattered_mask & valid_label_mask & (ransac_side_values == -1)

            def _directions_xy(track_xyz):
                return get_directions(
                    track_xyz,
                    beam_start=None,
                    beam_end=None,
                    include_z=True,
                    beam_plane_y=VolumeBoundaries.BEAM_CENTER.value,
                )

            def _offset_labels(labels, start_label, noise_labels=(-1, -20)):
                labels = np.asarray(labels, dtype=int)
                out = labels.copy()
                unique = [int(v) for v in np.unique(out) if int(v) not in noise_labels]
                mapping = {old: start_label + i for i, old in enumerate(unique)}
                for old, new in mapping.items():
                    out[out == old] = new
                return out, start_label + len(unique)

            # Start new merged labels above any existing non-noise labels
            valid_existing = ransac_cdist_labels[(ransac_cdist_labels != -1) & (ransac_cdist_labels != -20)]
            highest_label = int(np.max(valid_existing) + 1) if valid_existing.size else 0

            if np.any(above_mask):
                subset_above = data_points_3d[above_mask].copy()
                reg_above = Regularize(
                    data_array=subset_above,
                    low_energy_threshold=Optimize.C_DIST.value,
                    merge_type='cdist',
                    merge_algorithm='ransac',
                    func=_directions_xy,
                    label_column=DataArray.RANSAC_BEAM_MERGED,
                    print_g_matrix=False,
                )
                merged_above = reg_above.merge_labels().astype(int)
                merged_above, highest_label = _offset_labels(merged_above, highest_label)
                ransac_cdist_labels[above_mask] = merged_above

            if np.any(below_mask):
                subset_below = data_points_3d[below_mask].copy()
                reg_below = Regularize(
                    data_array=subset_below,
                    low_energy_threshold=Optimize.C_DIST.value,
                    merge_type='cdist',
                    merge_algorithm='ransac',
                    func=_directions_xy,
                    label_column=DataArray.RANSAC_BEAM_MERGED,
                    print_g_matrix=False,
                )
                merged_below = reg_below.merge_labels().astype(int)
                merged_below, highest_label = _offset_labels(merged_below, highest_label)
                ransac_cdist_labels[below_mask] = merged_below

            data_points_3d = np.column_stack((data_points_3d, ransac_cdist_labels))
            _t_ransac_cdist_done = time.time()
            print(f"  [timing] RANSAC cdist: {_t_ransac_cdist_done - _t_reg_done:.2f}s")

            # Additional scattered-track merging using cdist for REGULARIZED_BEAM_MERGED (above/below separately)
            # Output labels go into a new column: DataArray.REGULARIZED_CDIST
            reg_cdist_labels = data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value].astype(int).copy()

            reg_track_types = data_points_3d[:, DataArray.REGULARIZED_TRACK_TYPE.value].astype(int)
            reg_side_values = data_points_3d[:, DataArray.REGULARIZED_SIDE.value].astype(int)
            reg_bm_labels_for_merge = data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value].astype(int)

            reg_scattered_mask = reg_track_types == 1
            reg_valid_label_mask = reg_bm_labels_for_merge != -1
            reg_above_mask = reg_scattered_mask & reg_valid_label_mask & (reg_side_values == 1)
            reg_below_mask = reg_scattered_mask & reg_valid_label_mask & (reg_side_values == -1)

            valid_existing_reg = reg_cdist_labels[reg_cdist_labels != -1]
            reg_highest_label = int(np.max(valid_existing_reg) + 1) if valid_existing_reg.size else 0

            if np.any(reg_above_mask):
                reg_subset_above = data_points_3d[reg_above_mask].copy()
                reg_above = Regularize(
                    data_array=reg_subset_above,
                    low_energy_threshold=Optimize.C_DIST.value,
                    merge_type='cdist',
                    merge_algorithm='gmm',
                    func=_directions_xy,
                    label_column=DataArray.REGULARIZED_BEAM_MERGED,
                    print_g_matrix=False,
                )
                reg_merged_above = reg_above.merge_labels().astype(int)
                reg_merged_above, reg_highest_label = _offset_labels(
                    reg_merged_above, reg_highest_label, noise_labels=(-1,)
                )
                reg_cdist_labels[reg_above_mask] = reg_merged_above

            if np.any(reg_below_mask):
                reg_subset_below = data_points_3d[reg_below_mask].copy()
                reg_below = Regularize(
                    data_array=reg_subset_below,
                    low_energy_threshold=Optimize.C_DIST.value,
                    merge_type='cdist',
                    merge_algorithm='gmm',
                    func=_directions_xy,
                    label_column=DataArray.REGULARIZED_BEAM_MERGED,
                    print_g_matrix=False,
                )
                reg_merged_below = reg_below.merge_labels().astype(int)
                reg_merged_below, reg_highest_label = _offset_labels(
                    reg_merged_below, reg_highest_label, noise_labels=(-1,)
                )
                reg_cdist_labels[reg_below_mask] = reg_merged_below

            data_points_3d = np.column_stack((data_points_3d, reg_cdist_labels))
            _t_reg_cdist_done = time.time()
            print(f"  [timing] REG cdist:    {_t_reg_cdist_done - _t_ransac_cdist_done:.2f}s")

            event_id = int(entries.data.event)

            # Per-event scattered-track endpoint stores (3D)
            # These are overwritten each event (not stored for all events).
            GMM_REG = []
            RANSAC = []

            if batch_mode and save_canvas:
                zmin = RunParameters.z_start_bin.value
                zmax = RunParameters.z_end_bin.value

                h_reg_before = TH1F(
                    f"h_reg_before_{event_id}",
                    "Regularized beam Z centroids (before)",
                    100,
                    zmin,
                    zmax,
                )
                h_reg_after = TH1F(
                    f"h_reg_after_{event_id}",
                    "Regularized beam Z centroids (after)",
                    100,
                    zmin,
                    zmax,
                )
                h_ransac_before = TH1F(
                    f"h_ransac_before_{event_id}",
                    "RANSAC beam Z centroids (before)",
                    100,
                    zmin,
                    zmax,
                )
                h_ransac_after = TH1F(
                    f"h_ransac_after_{event_id}",
                    "RANSAC beam Z centroids (after)",
                    100,
                    zmin,
                    zmax,
                )

                for zc in reg_z_before.values():
                    h_reg_before.Fill(zc)
                for zc in reg_z_after.values():
                    h_reg_after.Fill(zc)
                for zc in ransac_z_before.values():
                    h_ransac_before.Fill(zc)
                for zc in ransac_z_after.values():
                    h_ransac_after.Fill(zc)

                c2 = root.TCanvas(
                    f"c2_{event_id}",
                    "Beam Z centroids (before/after)",
                    1200,
                    800,
                )
                c2.Divide(2, 2)

                c2.cd(1)
                h_reg_before.Draw()
                h_reg_before.GetXaxis().SetTitle("Z (mm)")
                h_reg_before.GetYaxis().SetTitle("Count")

                c2.cd(2)
                h_reg_after.Draw()
                h_reg_after.GetXaxis().SetTitle("Z (mm)")
                h_reg_after.GetYaxis().SetTitle("Count")

                c2.cd(3)
                h_ransac_before.Draw()
                h_ransac_before.GetXaxis().SetTitle("Z (mm)")
                h_ransac_before.GetYaxis().SetTitle("Count")

                c2.cd(4)
                h_ransac_after.Draw()
                h_ransac_after.GetXaxis().SetTitle("Z (mm)")
                h_ransac_after.GetYaxis().SetTitle("Count")

                c2.Update()
            
            if not batch_mode:
                print("  [WARNING] batch_mode=False: track processing and TTree fill are skipped. "
                      "Set batch_mode=True for physics output.")
                buf_run[0] = int(run_number)
                buf_event[0] = event_id
                buf_status[0] = 2  # status=2 for skipped (non-batch)
                for _br in (ransac_br, reg_br):
                    for v in _br.values():
                        if hasattr(v, 'clear'):
                            v.clear()
                    _br['n_vtx_groups'][0] = 0
                out_tree.Fill()

            if batch_mode:
                if save_canvas:
                    c1 = root.TCanvas(
                        f"c1_{event_id}",
                        'Clustering Comparison: DBSCAN, GMM, Regularized, RANSAC',
                        1200,
                        1300,
                    )
                    c1.Divide(3, 4)

                    # Set to None to plot all labels, or specify a label number (e.g., 61) to plot only that label
                    filter_label = None

                    # Row 1: DBSCAN labels (XY, YZ, XZ)
                    graphs_dbscan = plot_3d_projections(data_points_3d, DataArray.DBSCAN, c1, [1, 2, 3], filter_label=filter_label)

                    # Row 2: GMM labels (XY, YZ, XZ)
                    graphs_gmm = plot_3d_projections(data_points_3d, DataArray.GMM, c1, [4, 5, 6], filter_label=filter_label)

                    # Row 3: Regularized cdist-merged labels (XY, YZ, XZ)
                    graphs_reg = plot_3d_projections(data_points_3d, DataArray.REGULARIZED_CDIST, c1, [7, 8, 9], filter_label=filter_label)

                # Constrained PCA line fits for beam tracks (fixed XY endpoints)
                x_start = RunParameters.x_start_bin.value
                x_end = RunParameters.x_end_bin.value
                y_fixed = VolumeBoundaries.BEAM_CENTER.value

                reg_endpoints = fit_beam_tracks_pca_constrained_endpoints(
                    data_points_3d,
                    label_column=DataArray.REGULARIZED_BEAM_MERGED,
                    track_type_column=DataArray.REGULARIZED_TRACK_TYPE,
                    x_start=x_start,
                    x_end=x_end,
                    y_fixed=y_fixed,
                    noise_labels=(-1,),
                    beam_value=0,
                )
                ransac_endpoints = fit_beam_tracks_pca_constrained_endpoints(
                    data_points_3d,
                    label_column=DataArray.RANSAC_BEAM_MERGED,
                    track_type_column=DataArray.RANSAC_TRACK_TYPE,
                    x_start=x_start,
                    x_end=x_end,
                    y_fixed=y_fixed,
                    noise_labels=(-1, -20),
                    beam_value=0,
                )

                # Scattered Regularized tracks: plot PCA-based (XYZ) start/end markers on pads 7/8/9
                # Color: side=1 (above) -> red, else -> black
                # Marker: start=cross, end=circle
                reg_track_types = data_points_3d[:, DataArray.REGULARIZED_TRACK_TYPE.value].astype(int)
                reg_bm_labels = data_points_3d[:, DataArray.REGULARIZED_CDIST.value].astype(int)
                reg_side = data_points_3d[:, DataArray.REGULARIZED_SIDE.value].astype(int)

                scattered_labels = np.unique(reg_bm_labels[(reg_bm_labels != -1) & (reg_track_types == 1)])
                scattered_markers = []
                _t_reg_approach = time.time()
                _t_resp_total = 0.0

                for cluster_label in map(int, scattered_labels):
                    cluster_mask = (reg_bm_labels == cluster_label) & (reg_track_types == 1)
                    points_xyz = data_points_3d[cluster_mask][:, [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                    if points_xyz.shape[0] < 2:
                        continue

                    try:
                          end_point, start_point, _beam_vec, dirVecTrackNorm, *_ = get_directions(points_xyz, include_z=True)
                    except Exception:
                        continue

                    beam_label, beam_dist, vertex, trunc_start, trunc_end = find_closest_beam_track(
                        points_xyz, start_point, dirVecTrackNorm, reg_endpoints
                    )

                    # Compute lab angle (theta) and azimuthal angle (phi) using truncated scatter fit
                    theta_deg = None
                    phi_deg = None
                    theta2_deg = None
                    phi2_deg = None
                    comb_dir_unit = None
                    vertex2 = None
                    if (beam_label != -1 and beam_label in reg_endpoints
                            and trunc_start is not None and trunc_end is not None):
                        bp0_a, bp1_a = reg_endpoints[beam_label]
                        bp0_a = np.asarray(bp0_a, dtype=float)
                        bp1_a = np.asarray(bp1_a, dtype=float)
                        # Beam vector: from x=0 (start) toward x=x_end
                        beam_vec_a = (bp1_a - bp0_a) if bp0_a[0] <= bp1_a[0] else (bp0_a - bp1_a)
                        # Scatter vector: trunc_start (near beam plane) -> trunc_end (away)
                        scat_vec_a = np.asarray(trunc_end, dtype=float) - np.asarray(trunc_start, dtype=float)
                        theta_deg = angle_between(beam_vec_a, scat_vec_a)
                        phi_deg = calculate_phi_angle(scat_vec_a, beam_vec_a)
                        # Combined fit: truncated scatter pts + high-responsibility beam pts -> theta2, phi2
                        trunc_len_c = np.linalg.norm(scat_vec_a)
                        if trunc_len_c > 1e-9:
                            trunc_dir_cu = scat_vec_a / trunc_len_c
                            projs_c = (points_xyz - np.asarray(trunc_start, dtype=float)) @ trunc_dir_cu
                            trunc_mask_c = (projs_c >= 0) & (projs_c <= trunc_len_c)
                            trunc_scat_pts = points_xyz[trunc_mask_c]
                        else:
                            trunc_mask_c = np.ones(len(points_xyz), dtype=bool)
                            trunc_scat_pts = points_xyz
                        # Find dominant GMM component among NON-truncated scatter points
                        # (far from vertex, where scatter identity is unambiguous)
                        cluster_data_c = data_points_3d[cluster_mask]
                        non_trunc_gmm_vals_c = cluster_data_c[~trunc_mask_c, DataArray.GMM.value].astype(int)
                        valid_gmm_c = non_trunc_gmm_vals_c[non_trunc_gmm_vals_c >= 0]
                        # Fallback to truncated region if no valid non-truncated points
                        if len(valid_gmm_c) == 0:
                            trunc_gmm_vals_c = cluster_data_c[trunc_mask_c, DataArray.GMM.value].astype(int)
                            valid_gmm_c = trunc_gmm_vals_c[trunc_gmm_vals_c >= 0]
                        hi_beam_pts_c = np.empty((0, 3), dtype=float)
                        if len(valid_gmm_c) > 0 and responsibilities is not None:
                            _t_resp0 = time.time()
                            dom_gmm_c = int(np.bincount(valid_gmm_c).argmax())
                            if dom_gmm_c < responsibilities.shape[1]:
                                beam_all_mask_c = (
                                    (data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value].astype(int) == beam_label) &
                                    (data_points_3d[:, DataArray.REGULARIZED_TRACK_TYPE.value].astype(int) == 0)
                                )
                                beam_idx_c = np.where(beam_all_mask_c)[0]
                                if len(beam_idx_c) > 0:
                                    resp_vals_c = responsibilities[beam_idx_c, dom_gmm_c]
                                    _t_resp_total += time.time() - _t_resp0
                                    # Filter out invalid responsibilities (-1 means cross-HDBSCAN boundary)
                                    valid_resp_mask = resp_vals_c >= 0
                                    if not np.any(valid_resp_mask):
                                        print(f"    [diag] Scatter {cluster_label}: beam {beam_label} has no valid responsibilities "
                                              f"for GMM comp {dom_gmm_c} (cross-HDBSCAN boundary, combined fit = scatter-only)")
                                    hi_beam_idx_c = beam_idx_c[valid_resp_mask & (resp_vals_c > Optimize.GAMMA.value)]
                                    if len(hi_beam_idx_c) > 0:
                                        hi_beam_pts_c = data_points_3d[hi_beam_idx_c][:,
                                            [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                        if trunc_scat_pts.shape[0] >= 2:
                            from sklearn.decomposition import PCA as _PCA_comb
                            _combined_pts = (np.vstack([trunc_scat_pts, hi_beam_pts_c])
                                             if hi_beam_pts_c.shape[0] >= 1
                                             else trunc_scat_pts)
                            _pca_c = _PCA_comb(n_components=1)
                            _pca_c.fit(_combined_pts)
                            comb_dir = _pca_c.components_[0]
                            if np.dot(comb_dir, scat_vec_a) < 0:
                                comb_dir = -comb_dir
                            theta2_deg = angle_between(beam_vec_a, comb_dir)
                            phi2_deg = calculate_phi_angle(comb_dir, beam_vec_a)
                            comb_dir_unit = comb_dir.copy()
                            # Recompute vertex using combined direction (skew-line closest approach)
                            if beam_label != -1 and beam_label in reg_endpoints:
                                bp0_v2 = np.asarray(reg_endpoints[beam_label][0], dtype=float)
                                bp1_v2 = np.asarray(reg_endpoints[beam_label][1], dtype=float)
                                d_beam_v2 = bp1_v2 - bp0_v2
                                norm_v2 = np.linalg.norm(d_beam_v2)
                                if norm_v2 > 1e-9:
                                    d_beam_v2 = d_beam_v2 / norm_v2
                                    mean_comb = _combined_pts.mean(axis=0)
                                    w_v2 = mean_comb - bp0_v2
                                    b_v2 = float(np.dot(comb_dir_unit, d_beam_v2))
                                    d_v2 = float(np.dot(comb_dir_unit, w_v2))
                                    e_v2 = float(np.dot(d_beam_v2, w_v2))
                                    denom_v2 = 1.0 - b_v2 * b_v2
                                    if abs(denom_v2) < 1e-9:
                                        s_v2 = -d_v2
                                    else:
                                        s_v2 = (b_v2 * e_v2 - d_v2) / denom_v2
                                    vertex2 = mean_comb + s_v2 * comb_dir_unit

                    # Charge profile computation (scatter-only fit + combined fit)
                    range_scat_mm = None
                    energy_scat_keV = None
                    r2d_scat = None
                    delta_z_scat = None
                    cp_scat = None
                    range_comb_mm = None
                    energy_comb_keV = None
                    r2d_comb = None
                    delta_z_comb = None
                    cp_comb = None
                    if (vertex is not None and trunc_start is not None and trunc_end is not None):
                        cut_data_charge = data_points_3d[cluster_mask][:,
                            [DataArray.X.value, DataArray.Y.value, DataArray.Z.value, DataArray.Q.value]]
                        # Exclude points inside the beam attenuation zone
                        bz_mask = (cut_data_charge[:, 1] < VolumeBoundaries.BEAM_ZONE_MIN.value) | \
                                  (cut_data_charge[:, 1] > VolumeBoundaries.BEAM_ZONE_MAX.value)
                        cut_data_charge = cut_data_charge[bz_mask]
                        scat_dir_e = np.asarray(trunc_end, dtype=float) - np.asarray(trunc_start, dtype=float)
                        scat_len_e = np.linalg.norm(scat_dir_e)
                        full_len_e = np.linalg.norm(
                            np.asarray(end_point, dtype=float) - np.asarray(start_point, dtype=float))
                        extend_total = full_len_e + Reference.RANGE_EXTEND.value
                        if scat_len_e > 1e-9 and cut_data_charge.shape[0] >= 2:
                            scat_unit_e = scat_dir_e / scat_len_e
                            vt = np.asarray(vertex, dtype=float)
                            endpts_scat = np.vstack([vt, vt + scat_unit_e * extend_total])
                            try:
                                en_obj = Energy(cut_data_charge, endpts_scat, calibration_table)
                                new_pos, fit_en, lv_s, uv_3d, ll_2d, lv_e, hist_arr = en_obj.calculate_profiles()
                                r2d_e, sd_e, ran_end_e, en_end_e, ran_max_e, en_max_e, cp_x, cp_y, cp_xs, cp_ys = \
                                    en_obj.energy_weighted(Optimize.ALPHA.value, new_pos, fit_en, lv_s, uv_3d, ll_2d, lv_e, hist_arr)
                                range_scat_mm = ran_end_e
                                energy_scat_keV = range_energy_calculate(ran_end_e)
                                r2d_scat = r2d_e
                                delta_z_scat = sd_e
                                cp_scat = (cp_x.copy(), cp_y.copy(), cp_xs.copy(), cp_ys.copy(),
                                           ran_end_e, en_end_e, ran_max_e, en_max_e)
                            except Exception as e_cp:
                                pass
                            # Combined fit charge profile (same scattered data, different direction)
                            if comb_dir_unit is not None:
                                vt2 = np.asarray(vertex2, dtype=float) if vertex2 is not None else vt
                                endpts_comb = np.vstack([vt2, vt2 + comb_dir_unit * extend_total])
                                try:
                                    en_obj2 = Energy(cut_data_charge, endpts_comb, calibration_table)
                                    new_pos2, fit_en2, lv_s2, uv2, ll2, lv_e2, harr2 = en_obj2.calculate_profiles()
                                    r2d_c, sd_c, ran_end_c, en_end_c, ran_max_c, en_max_c, cp_x2, cp_y2, cp_xs2, cp_ys2 = \
                                        en_obj2.energy_weighted(Optimize.ALPHA.value, new_pos2, fit_en2, lv_s2, uv2, ll2, lv_e2, harr2)
                                    range_comb_mm = ran_end_c
                                    energy_comb_keV = range_energy_calculate(ran_end_c)
                                    r2d_comb = r2d_c
                                    delta_z_comb = sd_c
                                    cp_comb = (cp_x2.copy(), cp_y2.copy(), cp_xs2.copy(), cp_ys2.copy(),
                                               ran_end_c, en_end_c, ran_max_c, en_max_c)
                                except Exception as e_cp2:
                                    pass

                    # Store full 3D endpoints
                    GMM_REG.append(
                        TrackEndpoints3D(
                            track_id=int(cluster_label),
                            start_point_full=np.asarray(start_point, dtype=float).copy(),
                            end_point_full=np.asarray(end_point, dtype=float).copy(),
                            closest_beam_id=beam_label,
                            closest_beam_dist_mm=beam_dist,
                            vertex=np.asarray(vertex, dtype=float).copy() if vertex is not None else None,
                            trunc_start=np.asarray(trunc_start, dtype=float).copy() if trunc_start is not None else None,
                            trunc_end=np.asarray(trunc_end, dtype=float).copy() if trunc_end is not None else None,
                            theta_deg=theta_deg,
                            phi_deg=phi_deg,
                            theta2_deg=theta2_deg,
                            phi2_deg=phi2_deg,
                            vertex2=np.asarray(vertex2, dtype=float).copy() if vertex2 is not None else None,
                            range_scat_mm=range_scat_mm,
                            energy_scat_keV=energy_scat_keV,
                            r2d_scat=r2d_scat,
                            delta_z_scat=delta_z_scat,
                            cp_scat=cp_scat,
                            range_comb_mm=range_comb_mm,
                            energy_comb_keV=energy_comb_keV,
                            r2d_comb=r2d_comb,
                            delta_z_comb=delta_z_comb,
                            cp_comb=cp_comb,
                        )
                    )

                    if not save_canvas:
                        continue
                    side_value = int(reg_side[cluster_mask][0]) if np.any(cluster_mask) else 0
                    marker_color = root.kRed if side_value == 1 else root.kBlack

                    # XY (pad 7)
                    c1.cd(7)
                    start_marker_xy = root.TMarker(float(start_point[0]), float(start_point[1]), 25)
                    start_marker_xy.SetMarkerColor(marker_color)
                    start_marker_xy.SetMarkerSize(1.4)
                    start_marker_xy.Draw()
                    scattered_markers.append(start_marker_xy)

                    end_marker_xy = root.TMarker(float(end_point[0]), float(end_point[1]), 21)
                    end_marker_xy.SetMarkerColor(marker_color)
                    end_marker_xy.SetMarkerSize(1.4)
                    end_marker_xy.Draw()
                    scattered_markers.append(end_marker_xy)

                    # YZ (pad 8)
                    c1.cd(8)
                    start_marker_yz = root.TMarker(float(start_point[1]), float(start_point[2]), 25)
                    start_marker_yz.SetMarkerColor(marker_color)
                    start_marker_yz.SetMarkerSize(1.4)
                    start_marker_yz.Draw()
                    scattered_markers.append(start_marker_yz)

                    end_marker_yz = root.TMarker(float(end_point[1]), float(end_point[2]), 21)
                    end_marker_yz.SetMarkerColor(marker_color)
                    end_marker_yz.SetMarkerSize(1.4)
                    end_marker_yz.Draw()
                    scattered_markers.append(end_marker_yz)

                    # XZ (pad 9)
                    c1.cd(9)
                    start_marker_xz = root.TMarker(float(start_point[0]), float(start_point[2]), 25)
                    start_marker_xz.SetMarkerColor(marker_color)
                    start_marker_xz.SetMarkerSize(1.4)
                    start_marker_xz.Draw()
                    scattered_markers.append(start_marker_xz)

                    end_marker_xz = root.TMarker(float(end_point[0]), float(end_point[2]), 21)
                    end_marker_xz.SetMarkerColor(marker_color)
                    end_marker_xz.SetMarkerSize(1.4)
                    end_marker_xz.Draw()
                    scattered_markers.append(end_marker_xz)

                    # Fitted scattered-track line (solid)
                    c1.cd(7)
                    scat_ln_xy = root.TLine(float(start_point[0]), float(start_point[1]), float(end_point[0]), float(end_point[1]))
                    scat_ln_xy.SetLineColor(marker_color)
                    scat_ln_xy.SetLineWidth(2)
                    scat_ln_xy.Draw()
                    scattered_markers.append(scat_ln_xy)

                    c1.cd(8)
                    scat_ln_yz = root.TLine(float(start_point[1]), float(start_point[2]), float(end_point[1]), float(end_point[2]))
                    scat_ln_yz.SetLineColor(marker_color)
                    scat_ln_yz.SetLineWidth(2)
                    scat_ln_yz.Draw()
                    scattered_markers.append(scat_ln_yz)

                    c1.cd(9)
                    scat_ln_xz = root.TLine(float(start_point[0]), float(start_point[2]), float(end_point[0]), float(end_point[2]))
                    scat_ln_xz.SetLineColor(marker_color)
                    scat_ln_xz.SetLineWidth(2)
                    scat_ln_xz.Draw()
                    scattered_markers.append(scat_ln_xz)

                    # Closest beam-track line (dashed, same color)
                    if beam_label != -1 and beam_label in reg_endpoints:
                        bp0, bp1 = reg_endpoints[beam_label]
                        c1.cd(7)
                        bm_ln_xy = root.TLine(float(bp0[0]), float(bp0[1]), float(bp1[0]), float(bp1[1]))
                        bm_ln_xy.SetLineColor(marker_color)
                        bm_ln_xy.SetLineWidth(4)
                        bm_ln_xy.SetLineStyle(2)
                        bm_ln_xy.Draw()
                        scattered_markers.append(bm_ln_xy)

                        c1.cd(8)
                        bm_ln_yz = root.TLine(float(bp0[1]), float(bp0[2]), float(bp1[1]), float(bp1[2]))
                        bm_ln_yz.SetLineColor(marker_color)
                        bm_ln_yz.SetLineWidth(4)
                        bm_ln_yz.SetLineStyle(2)
                        bm_ln_yz.Draw()
                        scattered_markers.append(bm_ln_yz)

                        c1.cd(9)
                        bm_ln_xz = root.TLine(float(bp0[0]), float(bp0[2]), float(bp1[0]), float(bp1[2]))
                        bm_ln_xz.SetLineColor(marker_color)
                        bm_ln_xz.SetLineWidth(4)
                        bm_ln_xz.SetLineStyle(2)
                        bm_ln_xz.Draw()
                        scattered_markers.append(bm_ln_xz)

                    # Vertex: open circle on all 3 projections
                    if vertex is not None:
                        c1.cd(7)
                        vtx_xy = root.TMarker(float(vertex[0]), float(vertex[1]), 24)
                        vtx_xy.SetMarkerColor(marker_color)
                        vtx_xy.SetMarkerSize(2.0)
                        vtx_xy.Draw()
                        scattered_markers.append(vtx_xy)

                        c1.cd(8)
                        vtx_yz = root.TMarker(float(vertex[1]), float(vertex[2]), 24)
                        vtx_yz.SetMarkerColor(marker_color)
                        vtx_yz.SetMarkerSize(2.0)
                        vtx_yz.Draw()
                        scattered_markers.append(vtx_yz)

                        c1.cd(9)
                        vtx_xz = root.TMarker(float(vertex[0]), float(vertex[2]), 24)
                        vtx_xz.SetMarkerColor(marker_color)
                        vtx_xz.SetMarkerSize(2.0)
                        vtx_xz.Draw()
                        scattered_markers.append(vtx_xz)

                # Row 4: RANSAC cdist-merged labels (XY, YZ, XZ)
                _t_reg_approach_done = time.time()
                _reg_total = elapsed_dbscan + sum(elapsed_gmm) + (_t_reg_done - _t1) + (_t_reg_cdist_done - _t_ransac_cdist_done) + _t_resp_total
                print(f"  [timing] REG approach total:          {_reg_total:.2f}s  (responsibilities recompute: {_t_resp_total:.2f}s)")
                if save_canvas:
                    graphs_ransac = plot_3d_projections(data_points_3d, DataArray.RANSAC_CDIST, c1, [10, 11, 12], filter_label=filter_label)

                # Scattered RANSAC tracks: plot PCA-based (XYZ) start/end markers on pads 10/11/12
                # Color: side=1 (above) -> red, else -> black
                # Marker: square (start=open square, end=filled square)
                ransac_track_types = data_points_3d[:, DataArray.RANSAC_TRACK_TYPE.value].astype(int)
                ransac_bm_labels = data_points_3d[:, DataArray.RANSAC_CDIST.value].astype(int)
                ransac_side = data_points_3d[:, DataArray.RANSAC_SIDE.value].astype(int)

                ransac_scattered_labels = np.unique(
                    ransac_bm_labels[(ransac_bm_labels != -1) & (ransac_bm_labels != -20) & (ransac_track_types == 1)]
                )
                ransac_scattered_markers = []
                _t_ransac_approach = time.time()

                for cluster_label in map(int, ransac_scattered_labels):
                    cluster_mask = (ransac_bm_labels == cluster_label) & (ransac_track_types == 1)
                    points_xyz = data_points_3d[cluster_mask][:, [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                    if points_xyz.shape[0] < 2:
                        continue

                    try:
                        end_point, start_point, _beam_vec, dirVecTrackNorm, *_ = get_directions(points_xyz, include_z=True)
                    except Exception:
                        continue

                    beam_label, beam_dist, vertex, trunc_start, trunc_end = find_closest_beam_track(
                        points_xyz, start_point, dirVecTrackNorm, ransac_endpoints
                    )

                    # Compute lab angle (theta) and azimuthal angle (phi) using truncated scatter fit
                    theta_deg = None
                    phi_deg = None
                    if (beam_label != -1 and beam_label in ransac_endpoints
                            and trunc_start is not None and trunc_end is not None):
                        bp0_b, bp1_b = ransac_endpoints[beam_label]
                        bp0_b = np.asarray(bp0_b, dtype=float)
                        bp1_b = np.asarray(bp1_b, dtype=float)
                        # Beam vector: from x=0 (start) toward x=x_end
                        beam_vec_b = (bp1_b - bp0_b) if bp0_b[0] <= bp1_b[0] else (bp0_b - bp1_b)
                        # Scatter vector: trunc_start (near beam plane) -> trunc_end (away)
                        scat_vec_b = np.asarray(trunc_end, dtype=float) - np.asarray(trunc_start, dtype=float)
                        theta_deg = angle_between(beam_vec_b, scat_vec_b)
                        phi_deg = calculate_phi_angle(scat_vec_b, beam_vec_b)

                    # Charge profile computation (scatter-only fit)
                    range_scat_mm = None
                    energy_scat_keV = None
                    r2d_scat = None
                    delta_z_scat = None
                    cp_scat = None
                    if (vertex is not None and trunc_start is not None and trunc_end is not None):
                        cut_data_charge_r = data_points_3d[cluster_mask][:,
                            [DataArray.X.value, DataArray.Y.value, DataArray.Z.value, DataArray.Q.value]]
                        # Exclude points inside the beam attenuation zone
                        bz_mask_r = (cut_data_charge_r[:, 1] < VolumeBoundaries.BEAM_ZONE_MIN.value) | \
                                    (cut_data_charge_r[:, 1] > VolumeBoundaries.BEAM_ZONE_MAX.value)
                        cut_data_charge_r = cut_data_charge_r[bz_mask_r]
                        scat_dir_r = np.asarray(trunc_end, dtype=float) - np.asarray(trunc_start, dtype=float)
                        scat_len_r = np.linalg.norm(scat_dir_r)
                        full_len_r = np.linalg.norm(
                            np.asarray(end_point, dtype=float) - np.asarray(start_point, dtype=float))
                        extend_total_r = full_len_r + Reference.RANGE_EXTEND.value
                        if scat_len_r > 1e-9 and cut_data_charge_r.shape[0] >= 2:
                            scat_unit_r = scat_dir_r / scat_len_r
                            vt_r = np.asarray(vertex, dtype=float)
                            endpts_scat_r = np.vstack([vt_r, vt_r + scat_unit_r * extend_total_r])
                            try:
                                en_obj_r = Energy(cut_data_charge_r, endpts_scat_r, calibration_table)
                                new_pos_r, fit_en_r, lv_s_r, uv_r, ll_r, lv_e_r, harr_r = en_obj_r.calculate_profiles()
                                r2d_r, sd_r, ran_end_r, en_end_r, ran_max_r, en_max_r, cp_xr, cp_yr, cp_xsr, cp_ysr = \
                                    en_obj_r.energy_weighted(Optimize.ALPHA.value, new_pos_r, fit_en_r, lv_s_r, uv_r, ll_r, lv_e_r, harr_r)
                                range_scat_mm = ran_end_r
                                energy_scat_keV = range_energy_calculate(ran_end_r)
                                r2d_scat = r2d_r
                                delta_z_scat = sd_r
                                cp_scat = (cp_xr.copy(), cp_yr.copy(), cp_xsr.copy(), cp_ysr.copy(),
                                           ran_end_r, en_end_r, ran_max_r, en_max_r)
                            except Exception as e_cp_r:
                                pass

                    # Store full 3D endpoints
                    RANSAC.append(
                        TrackEndpoints3D(
                            track_id=int(cluster_label),
                            start_point_full=np.asarray(start_point, dtype=float).copy(),
                            end_point_full=np.asarray(end_point, dtype=float).copy(),
                            closest_beam_id=beam_label,
                            closest_beam_dist_mm=beam_dist,
                            vertex=np.asarray(vertex, dtype=float).copy() if vertex is not None else None,
                            trunc_start=np.asarray(trunc_start, dtype=float).copy() if trunc_start is not None else None,
                            trunc_end=np.asarray(trunc_end, dtype=float).copy() if trunc_end is not None else None,
                            theta_deg=theta_deg,
                            phi_deg=phi_deg,
                            range_scat_mm=range_scat_mm,
                            energy_scat_keV=energy_scat_keV,
                            r2d_scat=r2d_scat,
                            delta_z_scat=delta_z_scat,
                            cp_scat=cp_scat,
                        )
                    )

                    if not save_canvas:
                        continue
                    side_value = int(ransac_side[cluster_mask][0]) if np.any(cluster_mask) else 0
                    marker_color = root.kRed if side_value == 1 else root.kBlack

                    # XY (pad 10)
                    c1.cd(10)
                    start_marker_xy = root.TMarker(float(start_point[0]), float(start_point[1]), 25)
                    start_marker_xy.SetMarkerColor(marker_color)
                    start_marker_xy.SetMarkerSize(1.3)
                    start_marker_xy.Draw()
                    ransac_scattered_markers.append(start_marker_xy)

                    end_marker_xy = root.TMarker(float(end_point[0]), float(end_point[1]), 21)
                    end_marker_xy.SetMarkerColor(marker_color)
                    end_marker_xy.SetMarkerSize(1.3)
                    end_marker_xy.Draw()
                    ransac_scattered_markers.append(end_marker_xy)

                    # YZ (pad 11)
                    c1.cd(11)
                    start_marker_yz = root.TMarker(float(start_point[1]), float(start_point[2]), 25)
                    start_marker_yz.SetMarkerColor(marker_color)
                    start_marker_yz.SetMarkerSize(1.3)
                    start_marker_yz.Draw()
                    ransac_scattered_markers.append(start_marker_yz)

                    end_marker_yz = root.TMarker(float(end_point[1]), float(end_point[2]), 21)
                    end_marker_yz.SetMarkerColor(marker_color)
                    end_marker_yz.SetMarkerSize(1.3)
                    end_marker_yz.Draw()
                    ransac_scattered_markers.append(end_marker_yz)

                    # XZ (pad 12)
                    c1.cd(12)
                    start_marker_xz = root.TMarker(float(start_point[0]), float(start_point[2]), 25)
                    start_marker_xz.SetMarkerColor(marker_color)
                    start_marker_xz.SetMarkerSize(1.3)
                    start_marker_xz.Draw()
                    ransac_scattered_markers.append(start_marker_xz)

                    end_marker_xz = root.TMarker(float(end_point[0]), float(end_point[2]), 21)
                    end_marker_xz.SetMarkerColor(marker_color)
                    end_marker_xz.SetMarkerSize(1.3)
                    end_marker_xz.Draw()
                    ransac_scattered_markers.append(end_marker_xz)

                    # Fitted scattered-track line (solid)
                    c1.cd(10)
                    scat_ln_xy = root.TLine(float(start_point[0]), float(start_point[1]), float(end_point[0]), float(end_point[1]))
                    scat_ln_xy.SetLineColor(marker_color)
                    scat_ln_xy.SetLineWidth(2)
                    scat_ln_xy.Draw()
                    ransac_scattered_markers.append(scat_ln_xy)

                    c1.cd(11)
                    scat_ln_yz = root.TLine(float(start_point[1]), float(start_point[2]), float(end_point[1]), float(end_point[2]))
                    scat_ln_yz.SetLineColor(marker_color)
                    scat_ln_yz.SetLineWidth(2)
                    scat_ln_yz.Draw()
                    ransac_scattered_markers.append(scat_ln_yz)

                    c1.cd(12)
                    scat_ln_xz = root.TLine(float(start_point[0]), float(start_point[2]), float(end_point[0]), float(end_point[2]))
                    scat_ln_xz.SetLineColor(marker_color)
                    scat_ln_xz.SetLineWidth(2)
                    scat_ln_xz.Draw()
                    ransac_scattered_markers.append(scat_ln_xz)

                    # Closest beam-track line (dashed, same color)
                    if beam_label != -1 and beam_label in ransac_endpoints:
                        bp0, bp1 = ransac_endpoints[beam_label]
                        c1.cd(10)
                        bm_ln_xy = root.TLine(float(bp0[0]), float(bp0[1]), float(bp1[0]), float(bp1[1]))
                        bm_ln_xy.SetLineColor(marker_color)
                        bm_ln_xy.SetLineWidth(4)
                        bm_ln_xy.SetLineStyle(2)
                        bm_ln_xy.Draw()
                        ransac_scattered_markers.append(bm_ln_xy)

                        c1.cd(11)
                        bm_ln_yz = root.TLine(float(bp0[1]), float(bp0[2]), float(bp1[1]), float(bp1[2]))
                        bm_ln_yz.SetLineColor(marker_color)
                        bm_ln_yz.SetLineWidth(4)
                        bm_ln_yz.SetLineStyle(2)
                        bm_ln_yz.Draw()
                        ransac_scattered_markers.append(bm_ln_yz)

                        c1.cd(12)
                        bm_ln_xz = root.TLine(float(bp0[0]), float(bp0[2]), float(bp1[0]), float(bp1[2]))
                        bm_ln_xz.SetLineColor(marker_color)
                        bm_ln_xz.SetLineWidth(4)
                        bm_ln_xz.SetLineStyle(2)
                        bm_ln_xz.Draw()
                        ransac_scattered_markers.append(bm_ln_xz)

                    # Vertex: open circle on all 3 projections
                    if vertex is not None:
                        c1.cd(10)
                        vtx_xy = root.TMarker(float(vertex[0]), float(vertex[1]), 24)
                        vtx_xy.SetMarkerColor(marker_color)
                        vtx_xy.SetMarkerSize(2.0)
                        vtx_xy.Draw()
                        ransac_scattered_markers.append(vtx_xy)

                        c1.cd(11)
                        vtx_yz = root.TMarker(float(vertex[1]), float(vertex[2]), 24)
                        vtx_yz.SetMarkerColor(marker_color)
                        vtx_yz.SetMarkerSize(2.0)
                        vtx_yz.Draw()
                        ransac_scattered_markers.append(vtx_yz)

                        c1.cd(12)
                        vtx_xz = root.TMarker(float(vertex[0]), float(vertex[2]), 24)
                        vtx_xz.SetMarkerColor(marker_color)
                        vtx_xz.SetMarkerSize(2.0)
                        vtx_xz.Draw()
                        ransac_scattered_markers.append(vtx_xz)

                _t_ransac_approach_done = time.time()
                _ransac_total = elapsed_dbscan + (_t_ransac_cdist_done - _t_ransac)
                print(f"  [timing] RANSAC approach total:       {_ransac_total:.2f}s")

                if save_canvas:
                    fitted_lines = []

                    # XY pads: fixed segment (same for all clusters)
                    c1.cd(7)
                    line_xy_reg = root.TLine(x_start, y_fixed, x_end, y_fixed)
                    line_xy_reg.SetLineColor(root.kBlack)
                    line_xy_reg.SetLineWidth(2)
                    line_xy_reg.SetLineStyle(2)
                    line_xy_reg.Draw()
                    fitted_lines.append(line_xy_reg)

                    c1.cd(10)
                    line_xy_ransac = root.TLine(x_start, y_fixed, x_end, y_fixed)
                    line_xy_ransac.SetLineColor(root.kBlack)
                    line_xy_ransac.SetLineWidth(2)
                    line_xy_ransac.SetLineStyle(2)
                    line_xy_ransac.Draw()
                    fitted_lines.append(line_xy_ransac)

                    for (p0, p1) in reg_endpoints.values():
                        c1.cd(8)
                        line_yz = root.TLine(y_fixed, p0[2], y_fixed, p1[2])
                        line_yz.SetLineColor(root.kBlack)
                        line_yz.SetLineWidth(2)
                        line_yz.SetLineStyle(2)
                        line_yz.Draw()
                        fitted_lines.append(line_yz)

                        c1.cd(9)
                        line_xz = root.TLine(p0[0], p0[2], p1[0], p1[2])
                        line_xz.SetLineColor(root.kBlack)
                        line_xz.SetLineWidth(2)
                        line_xz.SetLineStyle(2)
                        line_xz.Draw()
                        fitted_lines.append(line_xz)

                    for (p0, p1) in ransac_endpoints.values():
                        c1.cd(11)
                        line_yz = root.TLine(y_fixed, p0[2], y_fixed, p1[2])
                        line_yz.SetLineColor(root.kBlack)
                        line_yz.SetLineWidth(2)
                        line_yz.SetLineStyle(2)
                        line_yz.Draw()
                        fitted_lines.append(line_yz)

                        c1.cd(12)
                        line_xz = root.TLine(p0[0], p0[2], p1[0], p1[2])
                        line_xz.SetLineColor(root.kBlack)
                        line_xz.SetLineWidth(2)
                        line_xz.SetLineStyle(2)
                        line_xz.Draw()
                        fitted_lines.append(line_xz)

                    c1.Update()
                    c2.Update()
                    _images_dir = FileNames.IMAGES_DIR.value
                    os.makedirs(_images_dir, exist_ok=True)
                    c1.SaveAs(f"{_images_dir}/event_{event_id}_clustering.png")
                    # c2.SaveAs(f"{_images_dir}/event_{event_id}_beam_centroids.png")

                # ── Vertex multiplicity: group nearby vertices, draw zoomed canvases ──────
                vertex_group_radius = Optimize.VERTEX_GROUP_RADIUS_MM.value
                vertex_zoom_margin  = Optimize.VERTEX_ZOOM_MARGIN_MM.value

                def _find_root(par, x):
                    while par[x] != x:
                        par[x] = par[par[x]]
                        x = par[x]
                    return x

                vtx_groups_by_method = {}
                for method_tag, endpoints_list, endpoints_dict, lbl_col, type_col in [
                    ("REG",   GMM_REG, reg_endpoints,
                     DataArray.REGULARIZED_CDIST.value, DataArray.REGULARIZED_TRACK_TYPE.value),
                    ("RANSAC", RANSAC, ransac_endpoints,
                     DataArray.RANSAC_CDIST.value,      DataArray.RANSAC_TRACK_TYPE.value),
                ]:
                    valid_eps = [ep for ep in endpoints_list if ep.vertex is not None]
                    if not valid_eps:
                        continue

                    # Union-find grouping by 3D vertex proximity
                    n = len(valid_eps)
                    par = list(range(n))
                    positions = np.array([ep.vertex for ep in valid_eps])
                    for i in range(n):
                        for j in range(i + 1, n):
                            if np.linalg.norm(positions[i] - positions[j]) <= vertex_group_radius:
                                ri, rj = _find_root(par, i), _find_root(par, j)
                                if ri != rj:
                                    par[ri] = rj

                    groups = {}
                    for idx in range(n):
                        groups.setdefault(_find_root(par, idx), []).append(valid_eps[idx])
                    vtx_groups_by_method[method_tag] = list(groups.values())

                    if not save_canvas:
                        continue
                    for grp_idx, group_eps in enumerate(groups.values()):
                        # Axis limits from all start/end/vertex points + margin
                        ref_pts = np.array([
                            pt for ep in group_eps
                            for pt in [
                                ep.trunc_start if ep.trunc_start is not None else ep.start_point_full,
                                ep.trunc_end   if ep.trunc_end   is not None else ep.end_point_full,
                                ep.vertex,
                            ]
                        ])
                        xmin_z = ref_pts[:, 0].min() - vertex_zoom_margin
                        xmax_z = ref_pts[:, 0].max() + vertex_zoom_margin
                        ymin_z = ref_pts[:, 1].min() - vertex_zoom_margin
                        ymax_z = ref_pts[:, 1].max() + vertex_zoom_margin
                        zmin_z = ref_pts[:, 2].min() - vertex_zoom_margin
                        zmax_z = ref_pts[:, 2].max() + vertex_zoom_margin

                        mult = len(group_eps)
                        cz = root.TCanvas(
                            f"cz_{method_tag}_{event_id}_{grp_idx}",
                            f"{method_tag} Event {event_id} VtxGroup {grp_idx} mult={mult}",
                            1800, 1200,
                        )
                        cz.Divide(3, 2)
                        zoom_objs = []
                        colors_grp = get_unique_colors(mult)
                        beam_pixel_color = root.kAzure + 2  # constant color for all beam pixels

                        # Physical pixel half-sizes for TBox rendering
                        px_dx = RunParameters.x_conversion_factor.value / 2.0
                        px_dy = RunParameters.y_conversion_factor.value / 2.0
                        px_dz = RunParameters.z_conversion_factor.value / 2.0

                        # Draw empty frames to set zoom axes, with grid lines
                        for pad_n, x0, y0, x1, y1, title in [
                            (1, xmin_z, ymin_z, xmax_z, ymax_z,
                             f"{method_tag} XY VtxGrp {grp_idx} mult={mult};X (mm);Y (mm)"),
                            (2, ymin_z, zmin_z, ymax_z, zmax_z,
                             f"{method_tag} YZ VtxGrp {grp_idx} mult={mult};Y (mm);Z (mm)"),
                            (3, xmin_z, zmin_z, xmax_z, zmax_z,
                             f"{method_tag} XZ VtxGrp {grp_idx} mult={mult};X (mm);Z (mm)"),
                        ]:
                            cz.cd(pad_n)
                            root.gPad.SetGridx(1)
                            root.gPad.SetGridy(1)
                            fr = root.gPad.DrawFrame(x0, y0, x1, y1)
                            fr.SetTitle(title)
                            # Tick labels every 10 mm (readable); grid follows primary ticks
                            fr.GetXaxis().SetNdivisions(-int(round((x1 - x0) / 10.0)), False)
                            fr.GetYaxis().SetNdivisions(-int(round((y1 - y0) / 10.0)), False)
                            zoom_objs.append(fr)

                        # ── Pass 1: draw all pixel boxes (scatter + beam) ──────────────────
                        drawn_beam_labels = set()
                        for ep_i, ep in enumerate(group_eps):
                            color    = colors_grp[ep_i]
                            beam_lbl = ep.closest_beam_id

                            # Raw scattered-track data points (filled pixel boxes)
                            track_mask = (
                                (data_points_3d[:, lbl_col].astype(int) == ep.track_id) &
                                (data_points_3d[:, type_col].astype(int) == 1)
                            )
                            pts = data_points_3d[track_mask][:,
                                [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                            if pts.shape[0] > 0:
                                for pad_n, ax, ay, ha, hb in [
                                    (1, pts[:, 0], pts[:, 1], px_dx, px_dy),
                                    (2, pts[:, 1], pts[:, 2], px_dy, px_dz),
                                    (3, pts[:, 0], pts[:, 2], px_dx, px_dz),
                                ]:
                                    cz.cd(pad_n)
                                    for k in range(len(pts)):
                                        bx = root.TBox(
                                            float(ax[k]) - ha, float(ay[k]) - hb,
                                            float(ax[k]) + ha, float(ay[k]) + hb,
                                        )
                                        bx.SetFillColor(color)
                                        bx.SetLineColor(color)
                                        bx.Draw()
                                        zoom_objs.append(bx)

                            # Raw closest beam-track data points (solid filled, constant beam color)
                            if beam_lbl != -1 and beam_lbl not in drawn_beam_labels:
                                drawn_beam_labels.add(beam_lbl)
                                beam_pts_mask = (
                                    (data_points_3d[:, lbl_col].astype(int) == beam_lbl) &
                                    (data_points_3d[:, type_col].astype(int) == 0)
                                )
                                beam_pts = data_points_3d[beam_pts_mask][:,
                                    [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                                if beam_pts.shape[0] > 0:
                                    for pad_n, ax, ay, ha, hb in [
                                        (1, beam_pts[:, 0], beam_pts[:, 1], px_dx, px_dy),
                                        (2, beam_pts[:, 1], beam_pts[:, 2], px_dy, px_dz),
                                        (3, beam_pts[:, 0], beam_pts[:, 2], px_dx, px_dz),
                                    ]:
                                        cz.cd(pad_n)
                                        for k in range(len(beam_pts)):
                                            bx = root.TBox(
                                                float(ax[k]) - ha, float(ay[k]) - hb,
                                                float(ax[k]) + ha, float(ay[k]) + hb,
                                            )
                                            bx.SetFillColor(beam_pixel_color)
                                            bx.SetLineColor(beam_pixel_color)
                                            bx.Draw()
                                            zoom_objs.append(bx)

                            # Beam points with high GMM responsibility for this scattered track
                            # (only for REG: only meaningful when beam & scatter shared the same DBSCAN cluster)
                            if (method_tag == "REG"
                                    and beam_lbl != -1
                                    and responsibilities is not None
                                    and ep.trunc_start is not None
                                    and ep.trunc_end is not None):
                                gamma_val = Optimize.GAMMA.value
                                trunc_dir = (np.asarray(ep.trunc_end, dtype=float)
                                             - np.asarray(ep.trunc_start, dtype=float))
                                trunc_len_v = np.linalg.norm(trunc_dir)
                                scat_all_mask = (
                                    (data_points_3d[:, lbl_col].astype(int) == ep.track_id) &
                                    (data_points_3d[:, type_col].astype(int) == 1)
                                )
                                scat_xyz_all = data_points_3d[scat_all_mask][:,
                                    [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                                if trunc_len_v > 1e-9 and scat_xyz_all.shape[0] > 0:
                                    trunc_dir_u = trunc_dir / trunc_len_v
                                    projs_s = (scat_xyz_all - ep.trunc_start) @ trunc_dir_u
                                    trunc_local = (projs_s >= 0) & (projs_s <= trunc_len_v)
                                    gmm_col = DataArray.GMM.value
                                    scat_gmm_vals = data_points_3d[scat_all_mask][trunc_local, gmm_col].astype(int)
                                    valid_gmm = scat_gmm_vals[scat_gmm_vals >= 0]
                                    if len(valid_gmm) > 0:
                                        dom_gmm = int(np.bincount(valid_gmm).argmax())
                                        if dom_gmm < responsibilities.shape[1]:
                                            beam_all_mask = (
                                                (data_points_3d[:, lbl_col].astype(int) == beam_lbl) &
                                                (data_points_3d[:, type_col].astype(int) == 0)
                                            )
                                            beam_idx = np.where(beam_all_mask)[0]
                                            if len(beam_idx) > 0:
                                                resp_vals = responsibilities[beam_idx, dom_gmm]
                                                hi_idx = beam_idx[resp_vals > gamma_val]
                                                if len(hi_idx) > 0:
                                                    hi_pts = data_points_3d[hi_idx][:,
                                                        [DataArray.X.value, DataArray.Y.value, DataArray.Z.value]]
                                                    hi_color = root.kOrange + 3
                                                    for pad_n, ax, ay, ha, hb in [
                                                        (1, hi_pts[:, 0], hi_pts[:, 1], px_dx, px_dy),
                                                        (2, hi_pts[:, 1], hi_pts[:, 2], px_dy, px_dz),
                                                        (3, hi_pts[:, 0], hi_pts[:, 2], px_dx, px_dz),
                                                    ]:
                                                        cz.cd(pad_n)
                                                        for k in range(len(hi_pts)):
                                                            bx = root.TBox(
                                                                float(ax[k]) - ha, float(ay[k]) - hb,
                                                                float(ax[k]) + ha, float(ay[k]) + hb,
                                                            )
                                                            bx.SetFillColor(hi_color)
                                                            bx.SetLineColor(hi_color)
                                                            bx.Draw()
                                                            zoom_objs.append(bx)
                        bz_min = float(VolumeBoundaries.BEAM_ZONE_MIN.value)
                        bz_max = float(VolumeBoundaries.BEAM_ZONE_MAX.value)
                        for bz_y in (bz_min, bz_max):
                            # XY pad: horizontal line at y = bz_y
                            cz.cd(1)
                            bz_ln_xy = root.TLine(xmin_z, bz_y, xmax_z, bz_y)
                            bz_ln_xy.SetLineColor(root.kGreen + 2)
                            bz_ln_xy.SetLineWidth(2)
                            bz_ln_xy.SetLineStyle(2)
                            bz_ln_xy.Draw()
                            zoom_objs.append(bz_ln_xy)
                            # YZ pad: vertical line at x = bz_y (Y is on x-axis)
                            cz.cd(2)
                            bz_ln_yz = root.TLine(bz_y, zmin_z, bz_y, zmax_z)
                            bz_ln_yz.SetLineColor(root.kGreen + 2)
                            bz_ln_yz.SetLineWidth(2)
                            bz_ln_yz.SetLineStyle(2)
                            bz_ln_yz.Draw()
                            zoom_objs.append(bz_ln_yz)

                        # ── Pass 2: draw fit lines, vertex markers, angle labels ──────────
                        for ep_i, ep in enumerate(group_eps):
                            color    = colors_grp[ep_i]
                            sp       = ep.trunc_start if ep.trunc_start is not None else ep.start_point_full
                            ep_end   = ep.trunc_end   if ep.trunc_end   is not None else ep.end_point_full
                            vt       = ep.vertex
                            beam_lbl = ep.closest_beam_id

                            # Fitted scattered track line (solid black — distinct from pixel color)
                            fit_color = root.kBlack
                            cz.cd(1)
                            sl_xy = root.TLine(float(sp[0]), float(sp[1]), float(ep_end[0]), float(ep_end[1]))
                            sl_xy.SetLineColor(fit_color); sl_xy.SetLineWidth(2)
                            sl_xy.Draw(); zoom_objs.append(sl_xy)
                            cz.cd(2)
                            sl_yz = root.TLine(float(sp[1]), float(sp[2]), float(ep_end[1]), float(ep_end[2]))
                            sl_yz.SetLineColor(fit_color); sl_yz.SetLineWidth(2)
                            sl_yz.Draw(); zoom_objs.append(sl_yz)
                            cz.cd(3)
                            sl_xz = root.TLine(float(sp[0]), float(sp[2]), float(ep_end[0]), float(ep_end[2]))
                            sl_xz.SetLineColor(fit_color); sl_xz.SetLineWidth(2)
                            sl_xz.Draw(); zoom_objs.append(sl_xz)

                            # Closest beam track line (dashed, beam color)
                            if beam_lbl != -1 and beam_lbl in endpoints_dict:
                                bp0, bp1 = endpoints_dict[beam_lbl]
                                cz.cd(1)
                                bl_xy = root.TLine(float(bp0[0]), float(bp0[1]), float(bp1[0]), float(bp1[1]))
                                bl_xy.SetLineColor(beam_pixel_color); bl_xy.SetLineWidth(4); bl_xy.SetLineStyle(2)
                                bl_xy.Draw(); zoom_objs.append(bl_xy)
                                cz.cd(2)
                                bl_yz = root.TLine(float(bp0[1]), float(bp0[2]), float(bp1[1]), float(bp1[2]))
                                bl_yz.SetLineColor(beam_pixel_color); bl_yz.SetLineWidth(4); bl_yz.SetLineStyle(2)
                                bl_yz.Draw(); zoom_objs.append(bl_yz)
                                cz.cd(3)
                                bl_xz = root.TLine(float(bp0[0]), float(bp0[2]), float(bp1[0]), float(bp1[2]))
                                bl_xz.SetLineColor(beam_pixel_color); bl_xz.SetLineWidth(4); bl_xz.SetLineStyle(2)
                                bl_xz.Draw(); zoom_objs.append(bl_xz)

                            # Vertex: open circle
                            cz.cd(1)
                            vm_xy = root.TMarker(float(vt[0]), float(vt[1]), 24)
                            vm_xy.SetMarkerColor(color); vm_xy.SetMarkerSize(2.0)
                            vm_xy.Draw(); zoom_objs.append(vm_xy)
                            cz.cd(2)
                            vm_yz = root.TMarker(float(vt[1]), float(vt[2]), 24)
                            vm_yz.SetMarkerColor(color); vm_yz.SetMarkerSize(2.0)
                            vm_yz.Draw(); zoom_objs.append(vm_yz)
                            cz.cd(3)
                            vm_xz = root.TMarker(float(vt[0]), float(vt[2]), 24)
                            vm_xz.SetMarkerColor(color); vm_xz.SetMarkerSize(2.0)
                            vm_xz.Draw(); zoom_objs.append(vm_xz)

                            # Angle labels: theta and phi in NDC on pad 3 (XZ)
                            if ep.theta_deg is not None and ep.phi_deg is not None:
                                cz.cd(3)
                                has_comb = (
                                    method_tag == "REG"
                                    and ep.theta2_deg is not None
                                    and ep.phi2_deg is not None
                                )
                                text_size = 0.045 if has_comb else 0.055
                                y_step = 0.13 if has_comb else 0.10
                                y_ndc = 0.88 - ep_i * y_step
                                lbl = root.TLatex()
                                lbl.SetNDC(True)
                                lbl.SetTextColor(color)
                                lbl.SetTextSize(text_size)
                                lbl.DrawLatex(
                                    0.12, y_ndc,
                                    f"Trk{ep_i}: #theta={ep.theta_deg:.1f}#circ  #phi={ep.phi_deg:.1f}#circ  (scat)"
                                )
                                zoom_objs.append(lbl)
                                if has_comb:
                                    lbl2 = root.TLatex()
                                    lbl2.SetNDC(True)
                                    lbl2.SetTextColor(color)
                                    lbl2.SetTextSize(text_size)
                                    lbl2.DrawLatex(
                                        0.12, y_ndc - 0.055,
                                        f"      #theta={ep.theta2_deg:.1f}#circ  #phi={ep.phi2_deg:.1f}#circ  (comb)"
                                    )
                                    zoom_objs.append(lbl2)

                        # ── Pass 3: draw charge profiles on pads 4-5 ──────────────────────
                        # Pad 4: scatter-only fit charge profiles
                        cz.cd(4)
                        root.gPad.SetGridx(1)
                        root.gPad.SetGridy(1)
                        mg_scat = root.TMultiGraph()
                        mg_scat.SetTitle("Charge Profile (scat fit);Range (mm);Charge (a.u.)")
                        has_scat_cp = False
                        scat_vlines = []
                        for ep_i, ep in enumerate(group_eps):
                            if ep.cp_scat is not None:
                                has_scat_cp = True
                                color = colors_grp[ep_i]
                                cp_x, cp_y, cp_xs, cp_ys, ran_end, en_end, ran_max, en_max = ep.cp_scat
                                # Raw profile (dotted)
                                gr_raw = root.TGraph(len(cp_x), np.array(cp_x, dtype='d'), np.array(cp_y, dtype='d'))
                                gr_raw.SetLineColor(color)
                                gr_raw.SetLineStyle(3)
                                mg_scat.Add(gr_raw)
                                zoom_objs.append(gr_raw)
                                # Smoothed profile (solid, thicker)
                                gr_sm = root.TGraph(len(cp_xs), np.array(cp_xs, dtype='d'), np.array(cp_ys, dtype='d'))
                                gr_sm.SetLineColor(color)
                                gr_sm.SetLineWidth(2)
                                mg_scat.Add(gr_sm)
                                zoom_objs.append(gr_sm)
                                scat_vlines.append((ran_end, ran_max, color,
                                    ep.range_scat_mm, ep.energy_scat_keV))
                        if has_scat_cp:
                            mg_scat.Draw("AL")
                            zoom_objs.append(mg_scat)
                            y_top = mg_scat.GetYaxis().GetXmax() if mg_scat.GetYaxis() else 1.0
                            for v_i, (ran_e, ran_m, vcol, r_mm, e_keV) in enumerate(scat_vlines):
                                # Vertical line at ran_end (range)
                                vl = root.TLine(float(ran_e), 0, float(ran_e), y_top)
                                vl.SetLineColor(vcol); vl.SetLineStyle(2); vl.SetLineWidth(2)
                                vl.Draw(); zoom_objs.append(vl)
                                # Vertical line at ran_max (max energy position)
                                vl_m = root.TLine(float(ran_m), 0, float(ran_m), y_top)
                                vl_m.SetLineColor(vcol); vl_m.SetLineStyle(3); vl_m.SetLineWidth(1)
                                vl_m.Draw(); zoom_objs.append(vl_m)
                                # Range/energy text label
                                lbl_r = root.TLatex()
                                lbl_r.SetNDC(True)
                                lbl_r.SetTextColor(vcol)
                                lbl_r.SetTextSize(0.045)
                                r_txt = f"R={r_mm:.1f}mm" if r_mm is not None else "R=?"
                                e_txt = f"E={e_keV:.1f}keV" if e_keV is not None else "E=?"
                                lbl_r.DrawLatex(0.55, 0.88 - v_i * 0.08, f"Trk{v_i}: {r_txt}  {e_txt}")
                                zoom_objs.append(lbl_r)

                        # Pad 5: combined fit charge profiles (REG only)
                        if method_tag == "REG":
                            cz.cd(5)
                            root.gPad.SetGridx(1)
                            root.gPad.SetGridy(1)
                            mg_comb = root.TMultiGraph()
                            mg_comb.SetTitle("Charge Profile (comb fit);Range (mm);Charge (a.u.)")
                            has_comb_cp = False
                            comb_vlines = []
                            for ep_i, ep in enumerate(group_eps):
                                if ep.cp_comb is not None:
                                    has_comb_cp = True
                                    color = colors_grp[ep_i]
                                    cp_x2, cp_y2, cp_xs2, cp_ys2, ran_end2, en_end2, ran_max2, en_max2 = ep.cp_comb
                                    gr_raw2 = root.TGraph(len(cp_x2), np.array(cp_x2, dtype='d'), np.array(cp_y2, dtype='d'))
                                    gr_raw2.SetLineColor(color)
                                    gr_raw2.SetLineStyle(3)
                                    mg_comb.Add(gr_raw2)
                                    zoom_objs.append(gr_raw2)
                                    gr_sm2 = root.TGraph(len(cp_xs2), np.array(cp_xs2, dtype='d'), np.array(cp_ys2, dtype='d'))
                                    gr_sm2.SetLineColor(color)
                                    gr_sm2.SetLineWidth(2)
                                    mg_comb.Add(gr_sm2)
                                    zoom_objs.append(gr_sm2)
                                    comb_vlines.append((ran_end2, ran_max2, color,
                                        ep.range_comb_mm, ep.energy_comb_keV))
                            if has_comb_cp:
                                mg_comb.Draw("AL")
                                zoom_objs.append(mg_comb)
                                y_top2 = mg_comb.GetYaxis().GetXmax() if mg_comb.GetYaxis() else 1.0
                                for v_i, (ran_e2, ran_m2, vcol2, r_mm2, e_keV2) in enumerate(comb_vlines):
                                    vl2 = root.TLine(float(ran_e2), 0, float(ran_e2), y_top2)
                                    vl2.SetLineColor(vcol2); vl2.SetLineStyle(2); vl2.SetLineWidth(2)
                                    vl2.Draw(); zoom_objs.append(vl2)
                                    vl2_m = root.TLine(float(ran_m2), 0, float(ran_m2), y_top2)
                                    vl2_m.SetLineColor(vcol2); vl2_m.SetLineStyle(3); vl2_m.SetLineWidth(1)
                                    vl2_m.Draw(); zoom_objs.append(vl2_m)
                                    lbl_c = root.TLatex()
                                    lbl_c.SetNDC(True)
                                    lbl_c.SetTextColor(vcol2)
                                    lbl_c.SetTextSize(0.045)
                                    r_txt2 = f"R={r_mm2:.1f}mm" if r_mm2 is not None else "R=?"
                                    e_txt2 = f"E={e_keV2:.1f}keV" if e_keV2 is not None else "E=?"
                                    lbl_c.DrawLatex(0.55, 0.88 - v_i * 0.08, f"Trk{v_i}: {r_txt2}  {e_txt2}")
                                    zoom_objs.append(lbl_c)

                        cz.Update()
                        cz.SaveAs(
                            f"{FileNames.IMAGES_DIR.value}/event_{event_id}_{method_tag}_vtxgroup_{grp_idx}.png"
                        )
                        root.gROOT.GetListOfCanvases().Remove(cz)
                        del cz

                # ── Fill output TTree for this event ──────────────────────────────
                def _fill_nested(br, groups_list, is_reg=False):
                    """Clear all vectors and fill nested structure."""
                    # clear everything
                    for v in br.values():
                        if hasattr(v, 'clear'):
                            v.clear()
                    br['n_vtx_groups'][0] = len(groups_list)
                    _nan = -999.0

                    for grp in groups_list:
                        br['vtx_mult'].push_back(len(grp))

                        # Build inner vectors for this group's tracks
                        iv_track_id      = root.std.vector('int')()
                        iv_beam_id       = root.std.vector('int')()
                        # double inner vectors
                        dbl_fields = [
                            'theta', 'phi', 'range_mm', 'energy_keV', 'r2d', 'delta_z',
                            'vertex_x', 'vertex_y', 'vertex_z',
                            'trunc_start_x', 'trunc_start_y', 'trunc_start_z',
                            'trunc_end_x', 'trunc_end_y', 'trunc_end_z',
                            'start_x', 'start_y', 'start_z',
                            'end_x', 'end_y', 'end_z',
                            'closest_beam_dist',
                        ]
                        iv_dbl = {n: root.std.vector('double')() for n in dbl_fields}
                        # charge profile inner-inner vectors
                        cp_names = ['cp_x', 'cp_y', 'cp_xs', 'cp_ys']
                        iv_cp = {n: root.std.vector(root.std.vector('double'))() for n in cp_names}
                        # REG combined
                        if is_reg:
                            comb_dbl_names = ['theta2', 'phi2', 'range_comb_mm', 'energy_comb_keV', 'r2d_comb', 'delta_z_comb',
                                              'vertex2_x', 'vertex2_y', 'vertex2_z']
                            iv_comb_dbl = {n: root.std.vector('double')() for n in comb_dbl_names}
                            comb_cp_names = ['cp_comb_x', 'cp_comb_y', 'cp_comb_xs', 'cp_comb_ys']
                            iv_comb_cp = {n: root.std.vector(root.std.vector('double'))() for n in comb_cp_names}

                        for ep in grp:
                            iv_track_id.push_back(ep.track_id)
                            iv_beam_id.push_back(ep.closest_beam_id)
                            iv_dbl['closest_beam_dist'].push_back(
                                ep.closest_beam_dist_mm if ep.closest_beam_dist_mm != float('inf') else _nan)
                            iv_dbl['theta'].push_back(ep.theta_deg if ep.theta_deg is not None else _nan)
                            iv_dbl['phi'].push_back(ep.phi_deg if ep.phi_deg is not None else _nan)
                            iv_dbl['range_mm'].push_back(ep.range_scat_mm if ep.range_scat_mm is not None else _nan)
                            iv_dbl['energy_keV'].push_back(ep.energy_scat_keV if ep.energy_scat_keV is not None else _nan)
                            iv_dbl['r2d'].push_back(ep.r2d_scat if ep.r2d_scat is not None else _nan)
                            iv_dbl['delta_z'].push_back(ep.delta_z_scat if ep.delta_z_scat is not None else _nan)
                            # Vertex
                            if ep.vertex is not None:
                                iv_dbl['vertex_x'].push_back(float(ep.vertex[0]))
                                iv_dbl['vertex_y'].push_back(float(ep.vertex[1]))
                                iv_dbl['vertex_z'].push_back(float(ep.vertex[2]))
                            else:
                                iv_dbl['vertex_x'].push_back(_nan)
                                iv_dbl['vertex_y'].push_back(_nan)
                                iv_dbl['vertex_z'].push_back(_nan)
                            # Trunc start/end
                            if ep.trunc_start is not None:
                                iv_dbl['trunc_start_x'].push_back(float(ep.trunc_start[0]))
                                iv_dbl['trunc_start_y'].push_back(float(ep.trunc_start[1]))
                                iv_dbl['trunc_start_z'].push_back(float(ep.trunc_start[2]))
                            else:
                                iv_dbl['trunc_start_x'].push_back(_nan)
                                iv_dbl['trunc_start_y'].push_back(_nan)
                                iv_dbl['trunc_start_z'].push_back(_nan)
                            if ep.trunc_end is not None:
                                iv_dbl['trunc_end_x'].push_back(float(ep.trunc_end[0]))
                                iv_dbl['trunc_end_y'].push_back(float(ep.trunc_end[1]))
                                iv_dbl['trunc_end_z'].push_back(float(ep.trunc_end[2]))
                            else:
                                iv_dbl['trunc_end_x'].push_back(_nan)
                                iv_dbl['trunc_end_y'].push_back(_nan)
                                iv_dbl['trunc_end_z'].push_back(_nan)
                            # Full-fit start/end
                            iv_dbl['start_x'].push_back(float(ep.start_point_full[0]))
                            iv_dbl['start_y'].push_back(float(ep.start_point_full[1]))
                            iv_dbl['start_z'].push_back(float(ep.start_point_full[2]))
                            iv_dbl['end_x'].push_back(float(ep.end_point_full[0]))
                            iv_dbl['end_y'].push_back(float(ep.end_point_full[1]))
                            iv_dbl['end_z'].push_back(float(ep.end_point_full[2]))
                            # Charge profiles [track][bin]
                            for vv_name, cp_idx in [('cp_x', 0), ('cp_y', 1), ('cp_xs', 2), ('cp_ys', 3)]:
                                inner = root.std.vector('double')()
                                if ep.cp_scat is not None:
                                    for val in ep.cp_scat[cp_idx]:
                                        inner.push_back(float(val))
                                iv_cp[vv_name].push_back(inner)
                            # Combined-fit (REG only)
                            if is_reg:
                                iv_comb_dbl['theta2'].push_back(ep.theta2_deg if ep.theta2_deg is not None else _nan)
                                iv_comb_dbl['phi2'].push_back(ep.phi2_deg if ep.phi2_deg is not None else _nan)
                                iv_comb_dbl['range_comb_mm'].push_back(
                                    ep.range_comb_mm if ep.range_comb_mm is not None else _nan)
                                iv_comb_dbl['energy_comb_keV'].push_back(
                                    ep.energy_comb_keV if ep.energy_comb_keV is not None else _nan)
                                iv_comb_dbl['r2d_comb'].push_back(
                                    ep.r2d_comb if ep.r2d_comb is not None else _nan)
                                iv_comb_dbl['delta_z_comb'].push_back(
                                    ep.delta_z_comb if ep.delta_z_comb is not None else _nan)
                                if ep.vertex2 is not None:
                                    iv_comb_dbl['vertex2_x'].push_back(float(ep.vertex2[0]))
                                    iv_comb_dbl['vertex2_y'].push_back(float(ep.vertex2[1]))
                                    iv_comb_dbl['vertex2_z'].push_back(float(ep.vertex2[2]))
                                else:
                                    iv_comb_dbl['vertex2_x'].push_back(_nan)
                                    iv_comb_dbl['vertex2_y'].push_back(_nan)
                                    iv_comb_dbl['vertex2_z'].push_back(_nan)
                                for vv_name, cp_idx in [('cp_comb_x', 0), ('cp_comb_y', 1), ('cp_comb_xs', 2), ('cp_comb_ys', 3)]:
                                    inner = root.std.vector('double')()
                                    if ep.cp_comb is not None:
                                        for val in ep.cp_comb[cp_idx]:
                                            inner.push_back(float(val))
                                    iv_comb_cp[vv_name].push_back(inner)

                        # Push this group's inner vectors into the outer vectors
                        br['track_id'].push_back(iv_track_id)
                        br['closest_beam_id'].push_back(iv_beam_id)
                        for n in dbl_fields:
                            br[n].push_back(iv_dbl[n])
                        for n in cp_names:
                            br[n].push_back(iv_cp[n])
                        if is_reg:
                            for n in comb_dbl_names:
                                br[n].push_back(iv_comb_dbl[n])
                            for n in comb_cp_names:
                                br[n].push_back(iv_comb_cp[n])

                buf_run[0] = int(run_number)
                buf_event[0] = event_id
                buf_status[0] = 0
                _fill_nested(ransac_br, vtx_groups_by_method.get("RANSAC", []))
                _fill_nested(reg_br, vtx_groups_by_method.get("REG", []), is_reg=True)
                out_tree.Fill()
                print(f"Event {event_id} done in {time.time() - _event_start_time:.2f}s")
          except Exception as _evt_exc:
            import traceback
            traceback.print_exc()
            buf_run[0] = int(run_number)
            buf_event[0] = int(entries.data.event)
            buf_status[0] = 1
            for _br in (ransac_br, reg_br):
                for v in _br.values():
                    if hasattr(v, 'clear'):
                        v.clear()
                _br['n_vtx_groups'][0] = 0
            out_tree.Fill()
            print(f"Event {int(entries.data.event)} failed in {time.time() - _event_start_time:.2f}s")
            continue

out_file.cd()
out_tree.Write()
out_file.Close()
print(f"Output written to {os.path.join(FileNames.OUTPUT_DIR.value, f'output_run_{run_number}_{start_event}_{end_event}.root')}")