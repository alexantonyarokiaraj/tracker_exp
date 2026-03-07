import sys
import argparse
from libraries import RunParameters, SCAN, DataArray, Optimize, RansacParameters
import ROOT as root
import numpy as np
from ROOT import gROOT, gSystem, TH2F, TTree, TFile, AddressOf, TLine, TMultiGraph, TEllipse, TH1F, TH3F, TNtuple
from helper_functions.functions import dbcluster, plot_3d_projections, hierarchical_clustering_with_responsibilities
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

batch_mode = False  # Execute without showing any ROOT or Python Plots

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
# good_events = [2, 19, 21, 33, 34, 37, 44, 46, 48, 49, 51, 57, 59, 60, 66, 68, 70, 75, 78, 79, 81, 84, 88, 90, 91, 92,
                # 95, 101, 108, 112, 114, 122]

good_events = [101]

# Looping entries from Tree
for entries in myTree:
    if entries.data.event in good_events: #if start_event <= entries.data.event <= end_event:  # Check if event is within the specified range
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
            
            # Print the number of unique RANSAC labels
            unique_ransac_labels = np.unique(ransac_labels_full[ransac_labels_full != -1])
            print(f"Number of unique RANSAC labels (excluding noise): {len(unique_ransac_labels)}")
        
            if not batch_mode:
                c1 = root.TCanvas('c1', 'Clustering Comparison: DBSCAN, GMM, Regularized, RANSAC', 1200, 1300)
                c1.Divide(3, 4)
                
                # Set to None to plot all labels, or specify a label number (e.g., 61) to plot only that label
                filter_label = None
                
                # Row 1: DBSCAN labels (XY, YZ, XZ)
                graphs_dbscan = plot_3d_projections(data_points_3d, DataArray.DBSCAN, c1, [1, 2, 3], filter_label=filter_label)
                
                # Row 2: GMM labels (XY, YZ, XZ)
                graphs_gmm = plot_3d_projections(data_points_3d, DataArray.GMM, c1, [4, 5, 6], filter_label=filter_label)
                
                # Row 3: Regularized labels (XY, YZ, XZ)
                graphs_reg = plot_3d_projections(data_points_3d, DataArray.REGULARIZED, c1, [7, 8, 9], filter_label=filter_label)
                
                # Row 4: RANSAC labels (XY, YZ, XZ)
                graphs_ransac = plot_3d_projections(data_points_3d, DataArray.RANSAC, c1, [10, 11, 12], filter_label=filter_label)
                
                c1.Update()
                # c1.SaveAs(f"images/event_{entries.data.event}.png")
                c1.WaitPrimitive()