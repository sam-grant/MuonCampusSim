"""
Samuel Grant 
Feb 2024

Analyse the phase space at the end of M5
"""
# External libraries
import numpy as np
import pandas as pd

# Internal libraries 
import Utils as ut

# --------------------
# Run
# --------------------

def Run():
    
    # Get configs and their aliases
    configs_ = { 
        "NoWedge" : "No wedge"
        ,"WedgeMinus5mm" : "$\minus5$ mm"
        ,"WedgePlus5mm" : "$\plus5$ mm"
    }

    # Define column names
    columns_ = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]  

    # Histogram holder, keys are the aliases (labels)
    hists_ = {alias: [] for alias in configs_.values()}


    for config, alias in configs_.items():

        finName = f"../../Data/{config}/muon_all_end.dat"

        print(f"---> Analysing {finName}")

        # Get data
        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # Plot single distribution
        ut.Plot1D(data=data["pz"], nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/{config}/h1_muons_end_pz_{config}.png") 

        # Append to list
        hists_[alias].append(data["pz"])

    # Overlay the histograms
    ut.Plot1DOverlay(hists_, nbins=78, xmin=-0.09, xmax=0.09, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.0025", fout=f"../../Images/{config}/h1_muons_end_pz_overlay.png", includeBlack=True) 
        
    return

# --------------------
# Main
# --------------------

def main():
    Run() 

if __name__ == "__main__":
    main()


