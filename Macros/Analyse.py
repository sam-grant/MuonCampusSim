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

def Run(config):
    
    # Get file name 
    finName = f"../Data/{config}/muon_all_end.dat"
    # Define column names
    columns = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]
    # Get data
    data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns)

    ut.Plot1D(data=data["pz"], nBins=28, xmin=-0.07, xmax=0.07, xlabel="$\delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../Images/{config}/h1_muons_end_pz_{config}.png") 
    
    return

# --------------------
# Main
# --------------------

def main():
    config="NoWedge"
    Run(config) 

if __name__ == "__main__":
    main()


