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
# Globals
# --------------------

# Define column names
columns_ = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]  

# configs and their aliases
configs_ = { 
    "NoWedge": "No wedge",
    "Minus10mm": "$-10$ mm",
    "Minus9mm": "$-9$ mm",
    "Minus8mm": "$-8$ mm",
    "Minus7mm": "$-7$ mm",
    "Minus6mm": "$-6$ mm",
    "Minus5mm": "$-5$ mm",
    "Minus4mm": "$-4$ mm",
    "Minus3mm": "$-3$ mm",
    "Minus2mm": "$-2$ mm",
    "Minus1mm": "$-1$ mm",
    "0mm": "$0$ mm",
    "Plus1mm": "$+1$ mm",
    "Plus2mm": "$+2$ mm",
    "Plus3mm": "$+3$ mm",
    "Plus4mm": "$+4$ mm",
    "Plus5mm": "$+5$ mm",
    "Plus6mm": "$+6$ mm",
    "Plus7mm": "$+7$ mm",
    "Plus8mm": "$+8$ mm",
    "Plus9mm": "$+9$ mm",
    "Plus10mm": "$+10$ mm"
}

# --------------------
# Run
# --------------------

def RunOffsetScan(ele="end"):

    # Histogram holder, keys are the aliases (labels)
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}

    # Graph lists 
    offsets_ = []
    normEntries_ = []
    normEntriesError_ = []

    for config, alias in configs_.items():

        finName = f"../../Data/Partial/{config}/muon_all_{ele}.dat"

        print(f"---> Analysing {finName}")

        # --------------------
        # Get data
        # --------------------
        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # --------------------
        # Fill histogram data
        # --------------------
        # Plot single distribution
        ut.Plot1D(data=data["pz"], nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/Partial/{ele}/h1_muons_end_pz_{config}.png") 
        # Append to list
        hists_[alias] = data["pz"]
        mask = (data["pz"] <= 0.002) & (data["pz"] >= -0.002)
        histsMasked_[alias] = data["pz"][mask]

        # --------------------
        # Fill graph data
        # --------------------
        if config == "NoWedge": continue
        # Convert the alias to a number
        offset = int(alias.split(" ")[0].strip("$").replace("+", ""))
        # Append offset
        offsets_.append(offset)
        # Append normalised entries with %2 of magic momentum.
        normEntries = len(histsMasked_[alias]) / len(histsMasked_[configs_["NoWedge"]]) 
        normEntriesError = normEntries * math.sqrt( (math.sqrt(len(histsMasked_[alias]))/len(histsMasked_[alias]))**2 + (math.sqrt(len(histsMasked_["No wedge"]))/len(histsMasked_["No wedge"]))**2 ) 
        normEntries_.append(normEntries) 
        normEntriesError_.append(normEntriesError)

    # Overlay the histograms
    # Just a selection, the graph is too messy otherwise
    selectedHists_ = {}
    for alias, hist_data in hists_.items():
        # It's messy to use the aliases like this, but it works
        if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]: 
            selectedHists_[alias] = hist_data

    selectedHistsMasked_ = {}
    for alias, hist_data in histsMasked_.items():
        if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]:
            selectedHistsMasked_[alias] = hist_data

    ut.Plot1DOverlay(selectedHists_, nbins=70, xmin=-0.07, xmax=0.07, title=ele, xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/Partial/{ele}/h1_muons_end_pz_overlay.png", includeBlack=True, colours_extended=False) 
    ut.Plot1DOverlay(selectedHistsMasked_, nbins=70, xmin=-0.07, xmax=0.07, title=r"{ele}, $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/Partial/{ele}/h1_muons_end_pz_overlay_pm0.002.png", includeBlack=True, colours_extended=False) 

    # Plot the relative number of muons
    ut.PlotNormEntriesGraph(x=offsets_, y=normEntries_, yerr=normEntriesError_, title=ele, xlabel="Wedge offset [mm]", ylabel=r"$\mu^{+} (|\Delta p / p_{0}| \leq 0.2\%)$ [normalised]", fout=f"../../Images/Partial/{ele}/gr_norm_entries_pm0.002.png")

    return

def RunMuonLosses(): 

    lastEle = 6195
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}

    for config, alias in configs_.items():

        finName = f"../../Data/Partial/{config}/muon_lost.dat"
        print(f"---> Analysing {finName}")

        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # ut.Plot1D(data=data["ele"], nbins=lastEle, xmin=0, xmax=lastEle, title=alias, xlabel="Lattice element", ylabel="Muon losses / element", fout=f"../../Images/Partial/Losses/h1_muons_losses_{config}.png") 

        hists_[alias] = data["ele"]
        
        mask = (data["pz"] <= 0.002) & (data["pz"] >= -0.002)
        histsMasked_[alias] = data["ele"][mask]

    selectedHists_ = {}
    # norm = hists_[configs_["NoWedge"]]
    for alias, hist_data in reversed(hists_.items()): # go in reverse so the overlay is clearer
        # It's messy to use the aliases like this, but it works
        # if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]: 
        #     selectedHists_[alias] = hist_data
        if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]: 
            selectedHists_[alias] = hist_data # / norm
        # ../../Images/Partial/Losses/h1_muons_losses_overlay_pm0.002.png

    selectedHistsMasked_ = {}
    # norm = histsMasked_[configs_["NoWedge"]]
    for alias, hist_data in reversed(histsMasked_.items()): # go in reverse so the overlay is clearer
        # It's messy to use the aliases like this, but it works
        if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]: 
            selectedHistsMasked_[alias] = hist_data # / norm

    # ut.Plot1DOverlay(hists_, nbins=lastEle, xmin=0, xmax=lastEle, xlabel="Lattice element", ylabel="Muon losses / element", fout=f"../../Images/Partial/Losses/h1_muons_losses_overlay_all.png")  
    # print(selectedHists_)
    ut.Plot1DLossesOverlay(selectedHists_, nbins=lastEle, xmin=0, xmax=lastEle, xlabel="Beamline element ID", ylabel="Muon losses / element", fout=f"../../Images/Partial/Losses/h1_muons_losses_overlay.png")  
    ut.Plot1DLossesOverlay(selectedHistsMasked_, nbins=lastEle, xmin=0, xmax=lastEle, title=r"$|\Delta p / p_{0}| \leq 0.2\%$", xlabel="Beamline element ID", ylabel="Muon losses / element", fout=f"../../Images/Partial/Losses/h1_muons_losses_overlay_pm0.002.png")  
     

    return

def RunWedgeCooling(config="0mm"):

    # --------------------
    # Get data
    # --------------------

    # File names
    finNameBefore = f"../../Data/Partial/{config}/muon_all_before_wedge.dat"
    finNameAfter = f"../../Data/Partial/{config}/muon_all_after_wedge.dat"
    finNameEnd = f"../../Data/Partial/{config}/muon_all_end.dat"
    finNameLoss = f"../../Data/Partial/{config}/muon_lost.dat"

    print(f"---> Analysing {finNameBefore}, {finNameAfter}, {finNameEnd}")

    # Read DataFrames
    dataBefore = pd.read_csv(finNameBefore, delim_whitespace=True, header=None, names=columns_)
    dataAfter = pd.read_csv(finNameAfter, delim_whitespace=True, header=None, names=columns_)
    dataEnd = pd.read_csv(finNameEnd, delim_whitespace=True, header=None, names=columns_)
    dataLoss = pd.read_csv(finNameLoss, delim_whitespace=True, header=None, names=columns_)

    # Perform merges
    dataBeforeToEnd = dataBefore.merge(dataEnd, on=["PID"], suffixes=("", "_end"), how="inner")
    dataAfterToEnd = dataAfter.merge(dataEnd, on=["PID"], suffixes=("", "_end"), how="inner") 

    dataBeforeToLoss = dataBefore.merge(dataLoss, on=["PID"], suffixes=("", "_lost"), how="inner")
    dataAfterToLoss = dataAfter.merge(dataLoss, on=["PID"], suffixes=("", "_lost"), how="inner") 

    print("\nBefore, After, End, BeforeToEnd, AfterToEnd, BeforeToLoss, AfterToLoss")
    print(len(dataBefore),",",len(dataAfter),",",len(dataEnd),",",len(dataBeforeToEnd),",",len(dataAfterToEnd),",",len(dataBeforeToLoss),",",len(dataAfterToLoss))

    # # Plot 1D momentum distributions
    # momHists_ = { "Before" : dataBeforeToEnd["pz_before"] ,"After" : dataAfterToEnd["pz_after"] }
    #     # ,"End" : dataEnd["pz"]
    # }

    # print(dataAfterToEnd)
    # print(dataEnd)

    # ut.Plot1DOverlay(hists_, nbins=70, xmin=-0.07, xmax=0.07, title=r"0 mm offset, $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_before_after.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay(hists_, nbins=70, xmin=-0.07, xmax=0.07, title="0 mm offset", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_before_after.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1D(momHists_["After"], nbins=28, xmin=-0.07, xmax=0.07, title=f"{configs_[config]} offset", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_after_to_end.png") 
    # ut.Plot1D(momHists_["End"], nbins=28, xmin=-0.07, xmax=0.07, title=f"{configs_[config]} offset", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/h1_muons_end.png") 
    
    ut.Plot1DOverlay({ "Before" : dataBefore["pz"] ,"After" : dataAfter["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"All $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/Partial/{config}/h1_muons_before_after_all_overlay.png", includeBlack=False, colours_extended=False) 
    ut.Plot1DOverlay({ "Before" : dataBeforeToLoss["pz"] ,"After" : dataAfterToLoss["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"Lost $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/Partial/{config}/h1_muons_before_after_loss_overlay.png", includeBlack=False, colours_extended=False) 
    ut.Plot1DOverlay({ "Before" : dataBeforeToEnd["pz"] ,"After" : dataAfterToEnd["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"Surviving $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/Partial/{config}/h1_muons_before_after_end_overlay.png", includeBlack=False, colours_extended=False) 

    ut.Plot2D(x=dataBefore["x"]*1e3, y=dataBefore["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"All $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_before_all.png") 
    ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_after_all.png")
    ut.Plot2D(x=dataBeforeToLoss["x"]*1e3, y=dataBeforeToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_before_loss.png") 
    ut.Plot2D(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_after_loss.png")
    ut.Plot2D(x=dataBefore["x"]*1e3, y=dataBefore["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Surviving $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_before_end.png") 
    ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Surviving $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_after_end.png")
    
    # ut.Plot2D(x=dataAfter["x"], y=dataAfter["y"], nBinsX=100, xmin=np.min(dataAfter["x"]), xmax=np.min(dataAfter["x"]), nBinsY=100, ymin=np.min(dataAfter["y"]), ymax=np.max(dataAfter["y"]), title="After wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/Partial/{config}/h2_xy_muons_after_all.png")
    ut.Plot2D(x=dataBefore["pz"], y=dataAfter["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_pz_muons_before_after_all.png") # , logZ=True) 
    ut.Plot2D(x=dataBeforeToLoss["pz"], y=dataAfterToLoss["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Lost $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_pz_muons_before_after_loss.png") # , logZ=True) 
    ut.Plot2D(x=dataBeforeToEnd["pz"], y=dataAfterToEnd["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Surviving $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_pz_muons_before_after_end.png") # , logZ=True) 

    ut.Plot2DWith1DProj(x=dataBefore["pz"], y=dataAfter["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_proj_pz_muons_before_after_all.png", logZ=True) 
    ut.Plot2DWith1DProj(x=dataBeforeToLoss["pz"], y=dataAfterToLoss["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Lost $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_proj_pz_muons_before_after_loss.png", logZ=True) 
    ut.Plot2DWith1DProj(x=dataBeforeToEnd["pz"], y=dataAfterToEnd["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Surviving $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/Partial/{config}/h2_proj_pz_muons_before_after_end.png", logZ=True) 


    ut.Plot3D(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, z=np.abs(dataAfterToLoss["pz"]), nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, zmax=0.07, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", zlabel="$|\Delta p / p_{0}|$", fout=f"../../Images/Partial/{config}/h3_xypz_muons_after_end.png")
    p_t_ = np.sqrt(dataAfterToLoss["px"]**2+dataAfterToLoss["py"]**2)
    ut.Plot3D(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, z=p_t_, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, zmax=0.07, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", zlabel="$p_{xy}/p_{0}$", fout=f"../../Images/Partial/{config}/h3_xypt_muons_after_end.png")
    
    return

# def RunBeamProfile(ele="end"):

#     return
# --------------------
# Main
# --------------------

def main():
    
    # [RunOffsetScan(ele) for ele in ["end", "before_wedge", "after_wedge"]]
    # RunMuonLosses()
    RunWedgeCooling() # can loop this guy
    # RunBeamProfile("after")


if __name__ == "__main__":
    main()


