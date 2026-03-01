import sys
import argparse
from libraries import RunParameters, SCAN
import ROOT as root
import numpy as np
from ROOT import gROOT, gSystem, TH2F, TTree, TFile, AddressOf, TLine, TMultiGraph, TEllipse, TH1F, TH3F, TNtuple
from helper_functions.functions import dbcluster

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Process events from a ROOT file.')
parser.add_argument('run_info', nargs='?', default='53@0_9999', help='Format: runnumber@start_event_end_event (default: 53@0_9999)')
args = parser.parse_args()

# Extract run number and event range
run_number, event_range = args.run_info.split('@')
start_event, end_event = map(int, event_range.split('_'))

sys.path.append(RunParameters.lib_path.value)

batch_mode = False  # Execute without showing any ROOT or Python Plots

if not batch_mode:
    c1 = root.TCanvas('c1', 'Events', 800, 600)
    c1.Divide(2, 2)

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
good_events = [2, 19, 21, 33, 34, 37, 44, 46, 48, 49, 51, 57, 59, 60, 66, 68, 70, 75, 78, 79, 81, 84, 88, 90, 91, 92,
               95, 101, 108, 112, 114, 122]

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
        
        if not batch_mode:
            # Create graphs colored by DBSCAN labels
            unique_labels = np.unique(dbscan_labels)
            colors = [root.kBlue, root.kRed, root.kGreen, root.kMagenta, root.kCyan, root.kYellow, root.kBlack, root.kGray]
            
            # Store graph objects to prevent garbage collection
            graphs_xy = []
            graphs_yz = []
            graphs_xz = []
            
            # XY projection
            c1.cd(1)
            for label_idx, label in enumerate(unique_labels):
                mask = dbscan_labels == label
                if label == -1:  # Noise points
                    color = root.kGray
                else:
                    color = colors[label_idx % len(colors)]
                
                points_xy = data_points_3d[mask]
                if len(points_xy) > 0:
                    graph_xy = root.TGraph(len(points_xy))
                    for i, point in enumerate(points_xy):
                        graph_xy.SetPoint(i, point[0], point[1])
                    
                    graph_xy.SetMarkerColor(color)
                    graph_xy.SetMarkerStyle(20)
                    graph_xy.SetMarkerSize(0.8)
                    if label_idx == 0:
                        graph_xy.Draw("AP")
                        graph_xy.SetTitle("XY Projection")
                        graph_xy.GetXaxis().SetTitle("X (mm)")
                        graph_xy.GetYaxis().SetTitle("Y (mm)")
                    else:
                        graph_xy.Draw("P same")
                    graphs_xy.append(graph_xy)
            
            # YZ projection
            c1.cd(2)
            for label_idx, label in enumerate(unique_labels):
                mask = dbscan_labels == label
                if label == -1:  # Noise points
                    color = root.kGray
                else:
                    color = colors[label_idx % len(colors)]
                
                points_yz = data_points_3d[mask]
                if len(points_yz) > 0:
                    graph_yz = root.TGraph(len(points_yz))
                    for i, point in enumerate(points_yz):
                        graph_yz.SetPoint(i, point[1], point[2])
                    
                    graph_yz.SetMarkerColor(color)
                    graph_yz.SetMarkerStyle(20)
                    graph_yz.SetMarkerSize(0.8)
                    if label_idx == 0:
                        graph_yz.Draw("AP")
                        graph_yz.SetTitle("YZ Projection")
                        graph_yz.GetXaxis().SetTitle("Y (mm)")
                        graph_yz.GetYaxis().SetTitle("Z (mm)")
                    else:
                        graph_yz.Draw("P same")
                    graphs_yz.append(graph_yz)
            
            # XZ projection
            c1.cd(3)
            for label_idx, label in enumerate(unique_labels):
                mask = dbscan_labels == label
                if label == -1:  # Noise points
                    color = root.kGray
                else:
                    color = colors[label_idx % len(colors)]
                
                points_xz = data_points_3d[mask]
                if len(points_xz) > 0:
                    graph_xz = root.TGraph(len(points_xz))
                    for i, point in enumerate(points_xz):
                        graph_xz.SetPoint(i, point[0], point[2])
                    
                    graph_xz.SetMarkerColor(color)
                    graph_xz.SetMarkerStyle(20)
                    graph_xz.SetMarkerSize(0.8)
                    if label_idx == 0:
                        graph_xz.Draw("AP")
                        graph_xz.SetTitle("XZ Projection")
                        graph_xz.GetXaxis().SetTitle("X (mm)")
                        graph_xz.GetYaxis().SetTitle("Z (mm)")
                    else:
                        graph_xz.Draw("P same")
                    graphs_xz.append(graph_xz)
            
            # Z projection histogram
            c1.cd(4)
            z_proj.Draw()
            
            c1.Update()
            c1.WaitPrimitive()
          