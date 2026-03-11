import sys
import argparse
import os
from dataclasses import dataclass
from libraries import RunParameters, SCAN, DataArray, Optimize, RansacParameters, VolumeBoundaries
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
)
from regularize import Regularize
from ransac import find_multiple_lines_ransac

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

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Process events from a ROOT file.')
parser.add_argument('run_info', nargs='?', default='53@0_9999', help='Format: runnumber@start_event_end_event (default: 53@0_9999)')
args = parser.parse_args()

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
tree_tracks = TTree('tracks', 'Tree containing Tracks found through clustering')
missed_pads = np.loadtxt(RunParameters.missing_pads_info.value)
x_pos = missed_pads[:,0]
y_pos = missed_pads[:,1]
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
        
        # Call DBSCAN clustering
        if len(data_points_3d) > 0:
            dbscan_labels, valid_cluster, epsilon_ = dbcluster(
                data_points_3d,
                SCAN.N_PROC.value,
                SCAN.NN_NEIGHBOR.value,
                SCAN.NN_RADIUS.value,
                SCAN.DB_MIN_SAMPLES.value,
                SCAN.SENSITIVITY.value,
                SCAN.EPS_THRESHOLD.value,
                SCAN.EPS_MODE.value
            )
            
            # Add DBSCAN labels to the data array
            data_points_3d = np.column_stack((data_points_3d, dbscan_labels))
            
            # Call hierarchical clustering with GMM (pass only X, Y, Z, Q columns)
            gmm_labels, n_comp, responsibilities, dbscan_labels_gmm, elapsed_dbscan, elapsed_gmm = hierarchical_clustering_with_responsibilities(data_points_3d[:, :4], max_components=10)
            
            # Add GMM labels to the data array
            data_points_3d = np.column_stack((data_points_3d, gmm_labels))
            
            # Call Regularize to merge GMM labels
            reg = Regularize(data_array=data_points_3d, threshold=Optimize.P_VALUE.value, merge_type='p_value', merge_algorithm='gmm')
            final_clusters = reg.merge_labels()
            
            # Add regularized labels to the data array
            data_points_3d = np.column_stack((data_points_3d, final_clusters))
            
            # Call RANSAC on non-noise points only
            dbscan_labels_full = data_points_3d[:, DataArray.DBSCAN.value].astype(int)
            non_noise_mask = dbscan_labels_full != -1
            non_noise_data = data_points_3d[non_noise_mask]
            
            ransac_labels_full = -1 * np.ones(len(data_points_3d), dtype=int)
            
            if len(non_noise_data) > 0:
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
            
            # Add RANSAC labels to the data array
            data_points_3d = np.column_stack((data_points_3d, ransac_labels_full))
            
            # Classify REGULARIZED clusters as beam or scattered tracks
            regularized_labels = data_points_3d[:, DataArray.REGULARIZED.value].astype(int)
            regularized_track_type = np.zeros(len(data_points_3d), dtype=int)  # 0=beam, 1=scattered
            
            unique_reg_clusters = np.unique(regularized_labels[regularized_labels != -1])
            for cluster_label in unique_reg_clusters:
                cluster_mask = regularized_labels == cluster_label
                centroid_y = np.mean(data_points_3d[cluster_mask, DataArray.Y.value])
                
                # If centroid Y is outside beam zone, mark as scattered (1)
                if centroid_y < VolumeBoundaries.BEAM_ZONE_MIN.value or centroid_y > VolumeBoundaries.BEAM_ZONE_MAX.value:
                    regularized_track_type[cluster_mask] = 1
            
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
                label_column=DataArray.REGULARIZED,
                track_type_column=DataArray.REGULARIZED_TRACK_TYPE,
                beam_zone_max=VolumeBoundaries.BEAM_ZONE_MAX.value,
                noise_labels=(-1,),
            )
            ransac_side = compute_scattered_track_side(
                data_points_3d,
                label_column=DataArray.RANSAC,
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
                    print_g_matrix=True,
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
                    print_g_matrix=True,
                )
                merged_below = reg_below.merge_labels().astype(int)
                merged_below, highest_label = _offset_labels(merged_below, highest_label)
                ransac_cdist_labels[below_mask] = merged_below

            data_points_3d = np.column_stack((data_points_3d, ransac_cdist_labels))

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
                    print_g_matrix=True,
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
                    print_g_matrix=True,
                )
                reg_merged_below = reg_below.merge_labels().astype(int)
                reg_merged_below, reg_highest_label = _offset_labels(
                    reg_merged_below, reg_highest_label, noise_labels=(-1,)
                )
                reg_cdist_labels[reg_below_mask] = reg_merged_below

            data_points_3d = np.column_stack((data_points_3d, reg_cdist_labels))

            event_id = int(entries.data.event)

            # Per-event scattered-track endpoint stores (3D)
            # These are overwritten each event (not stored for all events).
            GMM_REG = []
            RANSAC = []

            if batch_mode:
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
            
            if batch_mode:
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
                    print(f"  REG scattered {cluster_label} -> beam {beam_label} ({beam_dist:.1f} mm)")

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
                        )
                    )

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
                    print(f"  RANSAC scattered {cluster_label} -> beam {beam_label} ({beam_dist:.1f} mm)")

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
                        )
                    )

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
                os.makedirs("images_2", exist_ok=True)
                c1.SaveAs(f"images_2/event_{event_id}_clustering.png")
                # c2.SaveAs(f"images/event_{event_id}_beam_centroids.png")

                # ── Vertex multiplicity: group nearby vertices, draw zoomed canvases ──────
                vertex_group_radius = Optimize.VERTEX_GROUP_RADIUS_MM.value
                vertex_zoom_margin  = Optimize.VERTEX_ZOOM_MARGIN_MM.value

                def _find_root(par, x):
                    while par[x] != x:
                        par[x] = par[par[x]]
                        x = par[x]
                    return x

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

                    for grp_idx, group_eps in enumerate(groups.values()):
                        # Axis limits from all start/end/vertex points + margin
                        ref_pts = np.array([
                            pt for ep in group_eps
                            for pt in [ep.start_point_full, ep.end_point_full, ep.vertex]
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
                            1800, 600,
                        )
                        cz.Divide(3, 1)
                        zoom_objs = []
                        colors_grp = get_unique_colors(mult)

                        # Physical pixel half-sizes for TBox rendering
                        px_dx = RunParameters.x_conversion_factor.value / 2.0
                        px_dy = RunParameters.y_conversion_factor.value / 2.0
                        px_dz = RunParameters.z_conversion_factor.value / 2.0

                        # Draw empty frames to set zoom axes
                        cz.cd(1)
                        frame_xy = root.gPad.DrawFrame(xmin_z, ymin_z, xmax_z, ymax_z)
                        frame_xy.SetTitle(
                            f"{method_tag} XY VtxGrp {grp_idx} mult={mult};X (mm);Y (mm)"
                        )
                        zoom_objs.append(frame_xy)

                        cz.cd(2)
                        frame_yz = root.gPad.DrawFrame(ymin_z, zmin_z, ymax_z, zmax_z)
                        frame_yz.SetTitle(
                            f"{method_tag} YZ VtxGrp {grp_idx} mult={mult};Y (mm);Z (mm)"
                        )
                        zoom_objs.append(frame_yz)

                        cz.cd(3)
                        frame_xz = root.gPad.DrawFrame(xmin_z, zmin_z, xmax_z, zmax_z)
                        frame_xz.SetTitle(
                            f"{method_tag} XZ VtxGrp {grp_idx} mult={mult};X (mm);Z (mm)"
                        )
                        zoom_objs.append(frame_xz)

                        for ep_i, ep in enumerate(group_eps):
                            color  = colors_grp[ep_i]
                            sp     = ep.start_point_full
                            ep_end = ep.end_point_full
                            vt     = ep.vertex
                            beam_lbl = ep.closest_beam_id

                            # Raw scattered-track data points (2mm×2mm physical pixel boxes)
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

                            # Raw closest beam-track data points (hatched boxes, same color)
                            if beam_lbl != -1:
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
                                            bx.SetFillColor(color)
                                            bx.SetLineColor(color)
                                            bx.SetFillStyle(3004)
                                            bx.Draw()
                                            zoom_objs.append(bx)

                            # Fitted scattered track line (solid)
                            cz.cd(1)
                            sl_xy = root.TLine(float(sp[0]), float(sp[1]), float(ep_end[0]), float(ep_end[1]))
                            sl_xy.SetLineColor(color); sl_xy.SetLineWidth(2)
                            sl_xy.Draw(); zoom_objs.append(sl_xy)
                            cz.cd(2)
                            sl_yz = root.TLine(float(sp[1]), float(sp[2]), float(ep_end[1]), float(ep_end[2]))
                            sl_yz.SetLineColor(color); sl_yz.SetLineWidth(2)
                            sl_yz.Draw(); zoom_objs.append(sl_yz)
                            cz.cd(3)
                            sl_xz = root.TLine(float(sp[0]), float(sp[2]), float(ep_end[0]), float(ep_end[2]))
                            sl_xz.SetLineColor(color); sl_xz.SetLineWidth(2)
                            sl_xz.Draw(); zoom_objs.append(sl_xz)

                            # Closest beam track line (dashed, same color)
                            if beam_lbl != -1 and beam_lbl in endpoints_dict:
                                bp0, bp1 = endpoints_dict[beam_lbl]
                                cz.cd(1)
                                bl_xy = root.TLine(float(bp0[0]), float(bp0[1]), float(bp1[0]), float(bp1[1]))
                                bl_xy.SetLineColor(color); bl_xy.SetLineWidth(4); bl_xy.SetLineStyle(2)
                                bl_xy.Draw(); zoom_objs.append(bl_xy)
                                cz.cd(2)
                                bl_yz = root.TLine(float(bp0[1]), float(bp0[2]), float(bp1[1]), float(bp1[2]))
                                bl_yz.SetLineColor(color); bl_yz.SetLineWidth(4); bl_yz.SetLineStyle(2)
                                bl_yz.Draw(); zoom_objs.append(bl_yz)
                                cz.cd(3)
                                bl_xz = root.TLine(float(bp0[0]), float(bp0[2]), float(bp1[0]), float(bp1[2]))
                                bl_xz.SetLineColor(color); bl_xz.SetLineWidth(4); bl_xz.SetLineStyle(2)
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

                        cz.Update()
                        cz.SaveAs(
                            f"images_2/event_{event_id}_{method_tag}_vtxgroup_{grp_idx}.png"
                        )
                        root.gROOT.GetListOfCanvases().Remove(cz)
                        del cz