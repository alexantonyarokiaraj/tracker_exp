import sys
import argparse
import os
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
)
from regularize import Regularize
from ransac import find_multiple_lines_ransac

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

            scattered_mask = ransac_track_types == 1
            above_mask = scattered_mask & (ransac_side_values == 1)
            below_mask = scattered_mask & (ransac_side_values == -1)

            def _directions_xy(track_xyz):
                return get_directions(track_xyz, include_z=False)

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
                )
                merged_below = reg_below.merge_labels().astype(int)
                merged_below, highest_label = _offset_labels(merged_below, highest_label)
                ransac_cdist_labels[below_mask] = merged_below

            data_points_3d = np.column_stack((data_points_3d, ransac_cdist_labels))

            event_id = int(entries.data.event)

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
                
                # Row 3: Regularized Beam Merged labels (XY, YZ, XZ)
                graphs_reg = plot_3d_projections(data_points_3d, DataArray.REGULARIZED_BEAM_MERGED, c1, [7, 8, 9], filter_label=filter_label)

                # Scattered Regularized tracks: plot PCA-based (XY only) start/end markers on pad 7
                # Color: side=1 (above) -> red, else -> black
                # Marker: start=cross, end=circle
                reg_track_types = data_points_3d[:, DataArray.REGULARIZED_TRACK_TYPE.value].astype(int)
                reg_bm_labels = data_points_3d[:, DataArray.REGULARIZED_BEAM_MERGED.value].astype(int)
                reg_side = data_points_3d[:, DataArray.REGULARIZED_SIDE.value].astype(int)

                scattered_labels = np.unique(reg_bm_labels[(reg_bm_labels != -1) & (reg_track_types == 1)])
                scattered_markers = []

                c1.cd(7)
                for cluster_label in map(int, scattered_labels):
                    cluster_mask = (reg_bm_labels == cluster_label) & (reg_track_types == 1)
                    points_xy = data_points_3d[cluster_mask][:, [DataArray.X.value, DataArray.Y.value]]
                    if points_xy.shape[0] < 2:
                        continue

                    try:
                        end_point, start_point, *_ = get_directions(points_xy, include_z=False)
                    except Exception:
                        continue

                    side_value = int(reg_side[cluster_mask][0]) if np.any(cluster_mask) else 0
                    marker_color = root.kRed if side_value == 1 else root.kBlack

                    start_marker = root.TMarker(float(start_point[0]), float(start_point[1]), 25)
                    start_marker.SetMarkerColor(marker_color)
                    start_marker.SetMarkerSize(1.4)
                    start_marker.Draw()
                    scattered_markers.append(start_marker)

                    end_marker = root.TMarker(float(end_point[0]), float(end_point[1]), 21)
                    end_marker.SetMarkerColor(marker_color)
                    end_marker.SetMarkerSize(1.4)
                    end_marker.Draw()
                    scattered_markers.append(end_marker)
                
                # Row 4: RANSAC cdist-merged labels (XY, YZ, XZ)
                graphs_ransac = plot_3d_projections(data_points_3d, DataArray.RANSAC_CDIST, c1, [10, 11, 12], filter_label=filter_label)

                # Scattered RANSAC tracks: plot PCA-based (XY only) start/end markers on pad 10
                # Color: side=1 (above) -> red, else -> black
                # Marker: square (start=open square, end=filled square)
                ransac_track_types = data_points_3d[:, DataArray.RANSAC_TRACK_TYPE.value].astype(int)
                ransac_bm_labels = data_points_3d[:, DataArray.RANSAC_CDIST.value].astype(int)
                ransac_side = data_points_3d[:, DataArray.RANSAC_SIDE.value].astype(int)

                ransac_scattered_labels = np.unique(
                    ransac_bm_labels[(ransac_bm_labels != -1) & (ransac_bm_labels != -20) & (ransac_track_types == 1)]
                )
                ransac_scattered_markers = []

                c1.cd(10)
                for cluster_label in map(int, ransac_scattered_labels):
                    cluster_mask = (ransac_bm_labels == cluster_label) & (ransac_track_types == 1)
                    points_xy = data_points_3d[cluster_mask][:, [DataArray.X.value, DataArray.Y.value]]
                    if points_xy.shape[0] < 2:
                        continue

                    try:
                        end_point, start_point, *_ = get_directions(points_xy, include_z=False)
                    except Exception:
                        continue

                    side_value = int(ransac_side[cluster_mask][0]) if np.any(cluster_mask) else 0
                    marker_color = root.kRed if side_value == 1 else root.kBlack

                    start_marker = root.TMarker(float(start_point[0]), float(start_point[1]), 25)
                    start_marker.SetMarkerColor(marker_color)
                    start_marker.SetMarkerSize(1.3)
                    start_marker.Draw()
                    ransac_scattered_markers.append(start_marker)

                    end_marker = root.TMarker(float(end_point[0]), float(end_point[1]), 21)
                    end_marker.SetMarkerColor(marker_color)
                    end_marker.SetMarkerSize(1.3)
                    end_marker.Draw()
                    ransac_scattered_markers.append(end_marker)

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
                os.makedirs("images", exist_ok=True)
                c1.SaveAs(f"images/event_{event_id}_clustering.png")
                # c2.SaveAs(f"images/event_{event_id}_beam_centroids.png")