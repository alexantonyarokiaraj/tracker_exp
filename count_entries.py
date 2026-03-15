#!/usr/bin/env python3
"""Print total TTree entries for a given run number."""
import os
import sys
import ROOT as root
from libraries import RunParameters

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <run_number>")
    sys.exit(1)

run = sys.argv[1]
filename = RunParameters.files_path.value + f"Tree_Run_{run.zfill(4)}_Merged.root"

if not os.path.exists(filename):
    print(f"File not found: {filename}")
    sys.exit(1)

f = root.TFile(filename)
tree = f.Get("ACTAR_TTree")
print(f"Run {run}: {tree.GetEntries()} entries")
