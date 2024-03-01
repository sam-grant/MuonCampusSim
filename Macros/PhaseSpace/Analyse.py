"""
Samuel Grant 
Feb 2024

Analyse the phase space at the end of M5
"""
# External libraries
import numpy as np
import pandas as pd
import math

# Internal libraries 
import Utils as ut

# --------------------
# Run
# --------------------

def RunOffsetScan():
    
    # Get configs and their aliases
    configs_ = { 
        "NoWedge": "No wedge",
        "WedgeMinus10mm": "$-10$ mm",
        "WedgeMinus9mm": "$-9$ mm",
        "WedgeMinus8mm": "$-8$ mm",
        "WedgeMinus7mm": "$-7$ mm",
        "WedgeMinus6mm": "$-6$ mm",
        "WedgeMinus5mm": "$-5$ mm",
        "WedgeMinus4mm": "$-4$ mm",
        "WedgeMinus3mm": "$-3$ mm",
        "WedgeMinus2mm": "$-2$ mm",
        "WedgeMinus1mm": "$-1$ mm",
        "Wedge0mm": "$0$ mm",
        "WedgePlus1mm": "$+1$ mm",
        "WedgePlus2mm": "$+2$ mm",
        "WedgePlus3mm": "$+3$ mm",
        "WedgePlus4mm": "$+4$ mm",
        "WedgePlus5mm": "$+5$ mm",
        "WedgePlus6mm": "$+6$ mm",
        "WedgePlus7mm": "$+7$ mm",
        "WedgePlus8mm": "$+8$ mm",
        "WedgePlus9mm": "$+9$ mm",
        "WedgePlus10mm": "$+10$ mm"
    }

    # Define column names
    columns_ = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]  

    # Histogram holder, keys are the aliases (labels)
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}

    # Graph lists 
    offsets_ = []
    normEntries_ = []
    normEntriesError_ = []

    for config, alias in configs_.items():

        finName = f"../../Data/{config}/muon_all_end.dat"

        print(f"---> Analysing {finName}")

        # --------------------
        # Get data
        # --------------------
        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # --------------------
        # Fill histogram data
        # --------------------
        # Plot single distribution
        ut.Plot1D(data=data["pz"], nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/{config}/h1_muons_end_pz_{config}.png") 
        # Append to list
        hists_[alias] = data["pz"]
        mask = (data["pz"] <= 0.002) & (data["pz"] >= -0.002)
        histsMasked_[alias] = data["pz"][mask]

        # --------------------
        # Fill graph data
        # --------------------
        if alias == "No wedge": continue
        # Convert the alias to a number
        offset = int(alias.split(" ")[0].strip("$").replace("+", ""))
        # Append offset
        offsets_.append(offset)
        # Append normalised entries with %2 of magic momentum.
        normEntries = len(histsMasked_[alias]) / len(histsMasked_["No wedge"])
        normEntriesError = normEntries * math.sqrt( (math.sqrt(len(histsMasked_[alias]))/len(histsMasked_[alias]))**2 + (math.sqrt(len(histsMasked_["No wedge"]))/len(histsMasked_["No wedge"]))**2 ) 
        normEntries_.append(normEntries) 
        normEntriesError_.append(normEntriesError)

    # Overlay the histograms
    # Just a selection, the graph is too messy otherwise
    selectedHists_ = {}
    for alias, hist_data in hists_.items():
        if alias in ["No wedge", "$-10$ mm", "$-5$ mm", "$0$ mm", "$+5$ mm", "$+10$ mm"]: # "WedgeMinus10mm", "WedgeMinus5mm", "Wedge0mm", "WedgePlus5mm", "WedgePlus10mm"]:
            selectedHists_[alias] = hist_data

    selectedHistsMasked_ = {}
    for alias, hist_data in histsMasked_.items():
        if alias in ["No wedge", "$-10$ mm", "$-5$ mm", "$0$ mm", "$+5$ mm", "$+10$ mm"]: # "WedgeMinus10mm", "WedgeMinus5mm", "Wedge0mm", "WedgePlus5mm", "WedgePlus10mm"]:
            selectedHistsMasked_[alias] = hist_data

    ut.Plot1DOverlay(selectedHists_, nbins=70, xmin=-0.07, xmax=0.07, xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_end_pz_overlay.png", includeBlack=True, colours_extended=False) 
    ut.Plot1DOverlay(selectedHistsMasked_, nbins=70, xmin=-0.07, xmax=0.07, title=r"$|\Delta p / p_{0}| \leq 0.2\%$", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_end_pz_overlay_pm0.002.png", includeBlack=True, colours_extended=False) 

    # Plot the 'ctags' 
    ut.PlotNormEntriesGraph(x=offsets_, y=normEntries_, yerr=normEntriesError_, xlabel="Wedge offset [mm]", ylabel=r"$\mu^{+} (|\Delta p / p_{0}| \leq 0.2\%)$ [normalised]", fout="../../Images/gr_norm_entries_pm0.002.png")

    return


def RunWedgeCooling(config="WedgeMinus7mmBeforeAfter"):

    # --------------------
    # Get data
    # --------------------
    # File names
    finNameBefore = f"../../Data/{config}/muon_all_before_wedge.dat"
    finNameAfter = f"../../Data/{config}/muon_all_after_wedge.dat"
    print(f"---> Analysing {finNameBefore} and {finNameAfter}")
    # Define column names
    columns_ = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]  
    # DataFrames
    dataBefore = pd.read_csv(finNameBefore, delim_whitespace=True, header=None, names=columns_)
    dataAfter = pd.read_csv(finNameAfter, delim_whitespace=True, header=None, names=columns_)

    # mask = dataBefore["pz"] > 0.02
    # dataBefore = dataBefore[mask]

    # Merge 
    # These should be the same muons before and after
    dataWedge = dataBefore.merge(dataAfter, on=["PID"], suffixes=("_before", "_after"), how="inner")

    print("\nBefore, After, Common")
    print(len(dataBefore),",",len(dataAfter),",",len(dataWedge))
    print()

    # Plot 1D momentum distributions for sanity
    # Not sure why we have such low stats
    # are these the ones that actually intersect with the wedge? Surely not? 
    # Is it because I exclude the lost or decayed ones? 
    ut.Plot1D(data=dataWedge["pz_before"], nbins=28, xmin=-0.07, xmax=0.07, title="Before", xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/{config}/h1_pz_muons_before.png") 
    ut.Plot1D(data=dataWedge["pz_after"], nbins=28, xmin=-0.07, xmax=0.07, title="After", xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/{config}/h1_pz_muons_after.png") 
    hists_ = { 
        "Before" : dataWedge["pz_before"]
        ,"After" : dataWedge["pz_after"]
    }
    # ut.Plot1DOverlay(hists_, nbins=70, xmin=-0.07, xmax=0.07, title=r"0 mm offset, $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_before_after.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay(hists_, nbins=70, xmin=-0.07, xmax=0.07, title="0 mm offset", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_before_after.png", includeBlack=False, colours_extended=False) 
    ut.Plot1DOverlay(hists_, nbins=28, xmin=-0.07, xmax=0.07, title="$-7$ mm offset", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_before_after.png", includeBlack=False, colours_extended=False) 

    ut.Plot2D(x=dataWedge["pz_before"], y=dataWedge["pz_after"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title="0 mm offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/{config}/h2_pz_muons_before_after.png") # , log=True) 
    ut.Plot2DWith1DProj(x=dataWedge["pz_before"], y=dataWedge["pz_after"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title="0 mm offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/{config}/h2_proj_pz_muons_before_after.png" , logZ=False) 

    return
# --------------------
# Main
# --------------------

def main():

    # RunOffsetScan() 

    RunWedgeCooling() 

if __name__ == "__main__":
    main()


