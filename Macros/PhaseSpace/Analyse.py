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

stats = "high_stats"

# --------------------
# Helpers 
# --------------------

import re
# def ExtractNumber(string):
#     # Define a regular expression pattern to match a float number, including optional leading minus sign
#     pattern = r'-?\d+(\.\d+)?'
    
#     # Search for the pattern in the string
#     match = re.search(pattern, string)
    
#     if match:
#         # Convert the matched string to a float
#         return float(match.group())
#     else:
#         # Return None or raise an exception if no number is found
#         return None

def ExtractNumber(string):
    # Define a regular expression pattern to match "Minus" or "Plus" followed by a float number
    pattern = r'(Minus|Plus)?-?\d+(\.\d+)?'
    
    # Search for the pattern in the string
    match = re.search(pattern, string)
    
    if match:
        # Get the matched string
        number_str = match.group()
        
        # Handle the "Minus" and "Plus" prefixes
        if "Minus" in number_str:
            number_str = number_str.replace("Minus", "-")
        elif "Plus" in number_str:
            number_str = number_str.replace("Plus", "")
        
        # Convert the matched string to a float
        return float(number_str)
    else:
        # Return None or raise an exception if no number is found
        return None

# --------------------
# Globals
# --------------------

# Define column names
columns_ = ["PID", "s", "x", "px", "y", "py", "z", "pz", "ele", "sx", "sy", "sz"]  

# configs and their aliases
# configs_ = { 
#     "NoWedge": "No wedge",
#     "Minus10mm": "$-10$ mm",
#     "Minus9.55mm": "$-9.55$ mm",
#     "Minus9mm": "$-9$ mm",
#     "Minus8mm": "$-8$ mm",
#     "Minus7mm": "$-7$ mm",
#     "Minus6mm": "$-6$ mm",
#     "Minus5mm": "$-5$ mm",
#     "Minus4mm": "$-4$ mm",
#     "Minus3mm": "$-3$ mm",
#     "Minus2mm": "$-2$ mm",
#     "Minus1mm": "$-1$ mm",
#     "0mm": "$0$ mm",
#     "Plus1mm": "$+1$ mm",
#     "Plus2mm": "$+2$ mm",
#     "Plus3mm": "$+3$ mm",
#     "Plus4mm": "$+4$ mm",
#     "Plus5mm": "$+5$ mm",
#     "Plus6mm": "$+6$ mm",
#     "Plus7mm": "$+7$ mm",
#     "Plus8mm": "$+8$ mm",
#     "Plus9mm": "$+9$ mm",
#     "Plus9.55mm": "$+9.55$ mm",
#     "Plus10mm": "$+10$ mm"
# }

# configs_ = { 
#     "NoWedge": "No wedge",
#     "Minus10mm": "$-10$ mm",
#     "Minus5mm": "$-5$ mm",
#     "0mm": "$0$ mm",
#     "Plus5mm": "$+5$ mm",
#     "Plus10mm": "$+10$ mm"
# }

# configs_ = { 
#     "Minus10mm": "$-10$ mm",
#     "Plus10mm": "$+10$ mm"
# }

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

def RunSingleOffset(ele="end", offset="0mm", maskMom=False, MeV=False):

    print("\n---> RunSingleOffset():")

    # Histogram holder, keys are the aliases (labels)
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}
    # File names
    finNameNoWedge = f"../../../output/{stats}/NoWedge/muon_{ele}.dat"
    finNameWedge = f"../../../output/{stats}/{offset}/muon_{ele}.dat"

    print(f"---> Analysing:\n{finNameWedge}, {finNameNoWedge}")

    # Get data
    dataNoWedge = pd.read_csv(finNameNoWedge, delim_whitespace=True, header=None, names=columns_)
    dataWedge = pd.read_csv(finNameWedge, delim_whitespace=True, header=None, names=columns_)

    # Mask file tag
    maskMomTag = ""
    if maskMom: 
        # Mask +/- 0.2% 
        dataNoWedge = dataNoWedge[(dataNoWedge["pz"] <= 0.002) & (dataNoWedge["pz"] >= -0.002)]
        dataWedge = dataWedge[(dataWedge["pz"] <= 0.002) & (dataWedge["pz"] >= -0.002)]
        maskMomTag += "_accepted"

    if MeV: 
        # Convert to MeV/c
        print("Converting to MeV/c")
        p_magic = 3094
        dataNoWedge["pz"] = dataNoWedge["pz"]*p_magic + p_magic # - 1 ??? 
        dataWedge["pz"] = dataWedge["pz"]*p_magic + p_magic
        print(f"Mean momentum with no wedge = {np.mean(dataNoWedge['pz'])}")
        # Fill histogram
        ut.Plot1DOverlayWithStats({"No wedge" : dataNoWedge["pz"], configs_[offset] : dataWedge["pz"] }, nbins=70, xmin=2980, xmax=3220, xlabel="Momentum [MeV/c]", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/SingleOffset/h1_muons_{ele}_pz_MeV_NoWedge_vs_{offset}{maskMomTag}.png") 
        # ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xpz_muons_after_all_{config}.png")
    else: 
        ut.Plot1DOverlayWithStats({"No wedge" : dataNoWedge["pz"], configs_[offset] : dataWedge["pz"] }, nbins=28, xmin=-0.07, xmax=0.07, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/PhaseSpace/{stats}/SingleOffset/h1_muons_{ele}_pz_NoWedge_vs_{offset}{maskMomTag}.png") 
        # ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xpz_muons_after_all_{config}.png")

    
    return

def RunOffsetScan(ele="end"):

    print("\n---> RunOffsetScan():")
    # Histogram holder, keys are the aliases (labels)
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}

    # Graph lists 
    offsets_ = []
    R_ = []
    deltaR_ = []

    for config, alias in configs_.items():

        finName = f"../../../output/{stats}/{config}/muon_all_{ele}.dat"

        print(f"---> Analysing {finName}")

        # Get data
        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # Plot single distribution
        # ut.Plot1D(data=data["pz"], nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/PhaseSpace/{stats}/h1_muons_{ele}_pz_{config}.png") 

        # Append to list
        hists_[alias] = data["pz"]
        mask = (data["pz"] <= 0.002) & (data["pz"] >= -0.002)
        histsMasked_[alias] = data["pz"][mask]

        # Fill graph data
        if config == "NoWedge": continue

        # Convert the alias to a number
        offset = int(alias.split(" ")[0].strip("$").replace("+", ""))
        # Append offset
        offsets_.append(offset)
        # Append normalised entries with %2 of magic momentum.
        N1 = len(histsMasked_[alias])
        N2 = len(histsMasked_["No wedge"])
        R = N1 / N2 
        # Calculate the stat error 
        # Some of these uncertainties are highly correlated. 
        # Using Poisson statistics 
        # var(N) = N
        # Cov(N1, N2) = min(N1, N2)
        # (deltaR/R)^2 = (deltaN1/N1)^2 + (deltaN2/N2)^2 - 2*Cov(N1,N2)/N1*N2
        # (deltaR = R * sqrt( (1/N1) + (1/N2) - 2*Cov(N1,N2)/N1*N2)
        deltaR = R * math.sqrt((1/N1) + (1/N2) * (2*min(N1, N2)/(N1*N2)))
        R_.append(R) 
        deltaR_.append(deltaR)

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

    print()

    ut.PlotOffsetScanHists(selectedHists_, nbins=70, xmin=-0.07, xmax=0.07, title=ele, xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/OffsetScan/h1_muons_{ele}_pz_overlay.png", includeBlack=True, colours_extended=False) 
    ut.PlotOffsetScanHists(selectedHistsMasked_, nbins=70, xmin=-0.07, xmax=0.07, title=f"{ele}, $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="$\Delta p / p_{0}$", ylabel="$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/OffsetScan/h1_muons_{ele}_pz_overlay_pm0.002.png", includeBlack=True, colours_extended=False) 

    # Plot the relative number of muons
    ut.PlotOffsetScanGraph(x=offsets_, y=R_, yerr=deltaR_, title=ele, xlabel="Wedge offset [mm]", ylabel=r"$\mu^{+} (|\Delta p / p_{0}| \leq 0.2\%)$ [normalised]", fout=f"../../Images/PhaseSpace/{stats}/OffsetScan/gr_muons_{ele}_norm_entries_pm0.002.png")

    # for i, R in enumerate(R_): 
    # i_Rmax = configs_[offsets_[np.argmax(R_)]]

    print(f"The offset with the best performance is:\t{offsets_[np.argmax(R_)]} mm, with a performance of {np.max(R_)*100}+-{deltaR_[np.argmax(R_)]*100}%")
    return

def RunWedgeCooling(config="0mm"):

    print("\n---> RunWedgeCooling():")

    # Get data
    finNameBefore = f"../../../output/{stats}/{config}/muon_all_before_wedge.dat"
    finNameAfter = f"../../../output/{stats}/{config}/muon_all_after_wedge.dat"
    finNameEnd = f"../../../output/{stats}/{config}/muon_all_end.dat"
    finNameLoss = f"../../../output/{stats}/{config}/muon_lost.dat"

    print(f"---> Analysing {finNameBefore}, {finNameAfter}, {finNameEnd}")

    # Read DataFrames
    dataBefore = pd.read_csv(finNameBefore, delim_whitespace=True, header=None, names=columns_)
    dataAfter = pd.read_csv(finNameAfter, delim_whitespace=True, header=None, names=columns_)
    dataEnd = pd.read_csv(finNameEnd, delim_whitespace=True, header=None, names=columns_)
    dataLoss = pd.read_csv(finNameLoss, delim_whitespace=True, header=None, names=columns_)

    # Perform merges
    dataBeforeToAfter = dataBefore.merge(dataAfter, on=["PID"], suffixes=("", "_after"), how="inner")

    dataBeforeToEnd = dataBefore.merge(dataEnd, on=["PID"], suffixes=("", "_end"), how="inner")
    dataAfterToEnd = dataAfter.merge(dataEnd, on=["PID"], suffixes=("", "_end"), how="inner") 

    dataBeforeToLoss = dataBefore.merge(dataLoss, on=["PID"], suffixes=("", "_lost"), how="inner")
    dataAfterToLoss = dataAfter.merge(dataLoss, on=["PID"], suffixes=("", "_lost"), how="inner") 

    print("\nBefore, After, End, BeforeToAfter, BeforeToEnd, AfterToEnd, BeforeToLoss, AfterToLoss")
    print(len(dataBefore),",",len(dataAfter),",",len(dataEnd),",",len(dataBeforeToAfter),",",len(dataBeforeToEnd),",",len(dataAfterToEnd),",",len(dataBeforeToLoss),",",len(dataAfterToLoss))

    # # Longtiduinal momentum before and after the wedge
    # ut.Plot1DOverlay({ "Before" : dataBefore["pz"] ,"After" : dataAfter["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"All $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_before_after_all_overlay_{config}.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay({ "Before" : dataBeforeToLoss["pz"] ,"After" : dataAfterToLoss["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"Lost $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_before_after_loss_overlay_{config}.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay({ "Before" : dataBeforeToEnd["pz"] ,"After" : dataAfterToEnd["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=r"Surviving $\mu^{+}$, "+configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_before_after_end_overlay_{config}.png", includeBlack=False, colours_extended=False) 

    # # Transverse momentum before and after the wedge
    # ut.Plot1DOverlay({ "Before" : np.sqrt(dataBefore["px"]**2+dataBefore["py"]**2) ,"After" : np.sqrt(dataAfter["px"]**2+dataAfter["py"]**2) }, nbins=80, xmin=0, xmax=0.04, title=r"All $\mu^{+}$, "+configs_[config]+" offset", xlabel="$p_{T}/p_{0}$", ylabel=r"$\mu^{+}$ / 0.0005", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_pT_before_after_all_overlay_{config}.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay({ "Before" : np.sqrt(dataBeforeToLoss["px"]**2+dataBeforeToLoss["py"]**2) ,"After" : np.sqrt(dataAfterToLoss["px"]**2+dataAfterToLoss["py"]**2) }, nbins=80, xmin=0, xmax=0.04, title=r"Lost $\mu^{+}$, "+configs_[config]+" offset", xlabel="$p_{T}/p_{0}$", ylabel=r"$\mu^{+}$ / 0.0005", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_pT_before_after_loss_overlay_{config}.png", includeBlack=False, colours_extended=False) 
    # ut.Plot1DOverlay({ "Before" : np.sqrt(dataBeforeToEnd["px"]**2+dataBeforeToEnd["py"]**2) ,"After" : np.sqrt(dataAfterToEnd["px"]**2+dataAfterToEnd["py"]**2) }, nbins=80, xmin=0, xmax=0.04, title=r"Surviving $\mu^{+}$, "+configs_[config]+" offset", xlabel="$p_{T}/p_{0}$", ylabel=r"$\mu^{+}$ / 0.0005", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_pT_before_after_end_overlay_{config}.png", includeBlack=False, colours_extended=False) 

    # More inclusive overlays
    ut.Plot1DOverlay({ "Before wedge" : np.sqrt(dataBefore["px"]**2+dataBefore["py"]**2) ,"After wedge" : np.sqrt(dataAfter["px"]**2+dataAfter["py"]**2), "Lost before end" : np.sqrt(dataAfterToLoss["px"]**2+dataAfterToLoss["py"]**2), "Surviving to end" : np.sqrt(dataAfterToEnd["px"]**2+dataAfterToEnd["py"]**2) }, nbins=80, xmin=0, xmax=0.04, title=configs_[config]+" offset", xlabel="$p_{T}/p_{0}$", ylabel=r"$\mu^{+}$ / 0.0005", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_pT_main_overlay_{config}.png", includeBlack=False, colours_extended=False) 
    ut.Plot1DOverlay({ "Before wedge" : dataBefore["pz"], "After wedge" : dataAfter["pz"],  "Lost before end" : dataAfterToLoss["pz"], "Surviving to end" : dataAfterToEnd["pz"] }, nbins=70, xmin=-0.07, xmax=0.07, title=configs_[config]+" offset", xlabel="$\Delta p / p_{0}$", ylabel=r"$\mu^{+}$ / 0.002", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h1_muons_pz_main_overlay_{config}.png", includeBlack=False, colours_extended=False) 

    # return

    # Emittance momentum before and after the wedge
    # p_magic = 3094
    # dataNoWedge["pz"] = dataNoWedge["pz"]*p_magic + p_magic
    ut.Plot2D(x=dataBefore["x"]*1e3, y=dataBefore["px"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{x}$"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$p_{x}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_x_muons_before_{config}.png", logZ=False) 
    ut.Plot2D(x=dataBefore["y"]*1e3, y=dataBefore["py"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{y}$"+" before wedge, "+configs_[config]+" offset", xlabel="y [mm]", ylabel=r"$p_{y}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_y_muons_before_{config}.png", logZ=False) 
    # ut.Plot2D(x=dataBefore["z"]*1e3, y=dataBefore["pz"], nBinsX=2000, xmin=-200, xmax=200, nBinsY=200, ymin=-0.07, ymax=0.07, title=r"$\epsilon_{z}$", xlabel="z [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_z_muons_before_{config}.png") 
    
    ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["px"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{x}$"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$p_{x}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_x_muons_after_{config}.png", logZ=False)  
    ut.Plot2D(x=dataAfter["y"]*1e3, y=dataAfter["py"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{y}$"+" after wedge, "+configs_[config]+" offset", xlabel="y [mm]", ylabel=r"$p_{y}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_y_muons_after_{config}.png", logZ=False)  

    ut.Plot2D(x=dataLoss["x"]*1e3, y=dataLoss["px"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{x}$"+" lost, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$p_{x}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_x_muons_lost_{config}.png", logZ=False)  
    ut.Plot2D(x=dataLoss["y"]*1e3, y=dataLoss["py"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{y}$"+" lost, "+configs_[config]+" offset", xlabel="y [mm]", ylabel=r"$p_{y}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_y_muons_lost_{config}.png", logZ=False)  

    ut.Plot2D(x=dataEnd["x"]*1e3, y=dataEnd["px"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{x}$"+" end, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$p_{x}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_x_muons_end_{config}.png", logZ=False) 
    ut.Plot2D(x=dataEnd["y"]*1e3, y=dataEnd["py"], nBinsX=160, xmin=-40, xmax=40, nBinsY=100, ymin=-0.02, ymax=0.02, title=r"$\epsilon_{y}$"+" end, "+configs_[config]+" offset", xlabel="y [mm]", ylabel=r"$p_{y}/p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_y_muons_end_{config}.png", logZ=False) 

    # ut.Plot2D(x=dataAfter["z"]*1e3, y=dataAfter["pz"], nBinsX=2000, xmin=-200, xmax=200, nBinsY=200, ymin=-0.07, ymax=0.07, title=r"$\epsilon_{z}$", xlabel="z [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_epsilon_z_muons_after_{config}.png") 

    # Transverse position before and after the wedge and at the end of the beamline
    ut.Plot2D(x=dataBefore["x"]*1e3, y=dataBefore["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"All $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_before_all_{config}.png") 
    ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_after_all_{config}.png")
    ut.Plot2D(x=dataBeforeToLoss["x"]*1e3, y=dataBeforeToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_before_loss_{config}.png") 
    ut.Plot2D(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_after_loss_{config}.png")
    ut.Plot2D(x=dataEnd["x"]*1e3, y=dataEnd["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"All $\mu^{+}$,"+" end, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_end_all_{config}.png") 
  
    # With wedge illustation
    if config != "NoWedge":
        x_offset=ExtractNumber(config)
        ut.Plot2DWith2DWedge(x=dataBefore["x"]*1e3, y=dataBefore["y"]*1e3, nBinsX=160, xmin=-40, xmax=40, nBinsY=160, ymin=-40, ymax=40, x_offset=x_offset, title=r"All $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_before_all_with_wedge_{config}.png") 
        ut.Plot2DWith2DWedge(x=dataAfter["x"]*1e3, y=dataAfter["y"]*1e3, nBinsX=160, xmin=-40, xmax=40, nBinsY=160, ymin=-40, ymax=40, x_offset=x_offset, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_after_all_with_wedge_{config}.png")
        ut.Plot2DWith2DWedge(x=dataBeforeToLoss["x"]*1e3, y=dataBeforeToLoss["y"]*1e3, nBinsX=160, xmin=-40, xmax=40, nBinsY=160, ymin=-40, ymax=40, x_offset=x_offset, title=r"Lost $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_before_loss_with_wedge_{config}.png") 
        ut.Plot2DWith2DWedge(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, nBinsX=160, xmin=-40, xmax=40, nBinsY=160, ymin=-40, ymax=40, x_offset=x_offset, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_after_loss_with_wedge_{config}.png")

    # Momentum versus x-position before and after the wedge
    ut.Plot2D(x=dataBefore["x"]*1e3, y=dataBefore["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xpz_muons_before_all_{config}.png") 
    ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xpz_muons_after_all_{config}.png")

    # ut.Plot2D(x=(dataAfter["pz"] - dataBefore["pz"]), y=dataBefore["x"]*1e3, nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=80, ymin=-40, ymax=40,  title=r"All $\mu^{+}$,"+" through wedge, "+configs_[config]+" offset", xlabel=r"$(\Delta p / p_{0})_{\mathrm{after}} - (\Delta p / p_{0})_{\mathrm{before}}$", ylabel="x [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_deltapz_vs_x_muons_before_after_all_{config}.png") 

    # Really useful plot showing that the wedge is orientated correctly 
    ut.Plot2D(x=dataBefore["x"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_x_vs_deltapz_muons_before_after_all_{config}.png")     
    ut.Plot2D(x=dataBefore["x"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  logZ=True, fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_x_vs_deltapz_muons_before_after_all_log_{config}.png") 
    ut.Plot2D(x=dataBefore["y"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_y_vs_deltapz_muons_before_after_all_{config}.png")     
    ut.Plot2D(x=dataBefore["y"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  logZ=True, fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_y_vs_deltapz_muons_before_after_all_log_{config}.png") 

    if config != "NoWedge":
        x_offset=ExtractNumber(config)
        ut.Plot2DWith1DWedge(x=dataBefore["x"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, x_offset=x_offset, title=configs_[config], xlabel="x [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_x_vs_deltapz_muons_before_after_all_with_wedge_{config}.png")     
        ut.Plot2DWith1DWedge(x=dataBefore["x"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, x_offset=x_offset, title=configs_[config], xlabel="x [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  logZ=True, fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_x_vs_deltapz_muons_before_after_all_with_wedge_log_{config}.png") 
        # ut.Plot2DWith1DWedge(x=dataBefore["y"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=80, xmin=-40, xmax=40, nBinsY=28, ymin=-0.07, ymax=0.07, x_offset=x_offset, title=configs_[config], xlabel="y [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_y_vs_deltapz_muons_before_after_all_with_wedge_{config}.png")     
        # ut.Plot2DWith1DWedge(x=dataBefore["y"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=80, xmin=-40, xmax=40, nBinsY=28, ymin=-0.07, ymax=0.07, x_offset=x_offset, title=configs_[config], xlabel="y [mm]", ylabel=r" $\Delta (\Delta p / p_{0})$",  logZ=True, fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_y_vs_deltapz_muons_before_after_all_with_wedge_log_{config}.png") 

    # ut.Plot2D( x=dataBefore["x"]*1e3, y=(dataAfter["pz"] - dataBefore["pz"]), nBinsX=80, xmin=-40, xmax=40, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" through wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$(\Delta p / p_{0})_{\mathrm{after}} - (\Delta p / p_{0})_{\mathrm{before}}$",  fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_x_vs_deltapz_muons_before_after_all_{config}.png")     
    # ut.Plot2D(x=dataAfter["x"]*1e3, y=dataAfter["pz"], nBinsX=80, xmin=-40, xmax=40, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xpz_muons_after_all_{config}.png")
    # ut.Plot2D(x=dataBeforeToLoss["x"]*1e3, y=dataBeforeToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" before wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_before_loss_{config}.png") 
    # ut.Plot2D(x=dataAfterToLoss["x"]*1e3, y=dataAfterToLoss["y"]*1e3, nBinsX=80, xmin=-40, xmax=40, nBinsY=80, ymin=-40, ymax=40, title=r"Lost $\mu^{+}$,"+" after wedge, "+configs_[config]+" offset", xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_xy_muons_after_loss_{config}.png")
   
    # Momentum after versus before with projections
    # print(len(dataBefore["pz"]), len(dataAfter["pz"]), len(dataBeforeToAfter["pz"]))

    ut.Plot2DWith1DProj(x=dataBeforeToAfter["pz"], y=dataAfter["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"All $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_proj_pz_muons_before_after_all_{config}.png", logZ=True) 
    ut.Plot2DWith1DProj(x=dataBeforeToLoss["pz"], y=dataAfterToLoss["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Lost $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_proj_pz_muons_before_after_loss_{config}.png", logZ=True) 
    ut.Plot2DWith1DProj(x=dataBeforeToEnd["pz"], y=dataAfterToEnd["pz"], nBinsX=28, xmin=-0.07, xmax=0.07, nBinsY=28, ymin=-0.07, ymax=0.07, title=r"Surviving $\mu^{+}$,"+configs_[config]+" offset", xlabel=r"$\Delta p / p_{0}$ (before wedge)", ylabel=r"$\Delta p / p_{0}$ (after wedge)", fout=f"../../Images/PhaseSpace/{stats}/WedgeCooling/h2_proj_pz_muons_before_after_end_{config}.png", logZ=True) 

    return

# def RunCheckPosZ(config="0mm"):

#     print("\n---> RunCheckPosZ():")

#     # Get data
#     finName = f"../../../output/{stats}/{config}/muon_decay.dat"
#     print(f"---> Analysing {finName}")

#     # Data
#     data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

#     # print(data)
#     print(data["ele"])

#     # # Get everything at the wedge and box 
#     mask = ((data["ele"] == 6000) | (data["ele"] == 6001))
#     data = data[mask]
#     print(data["ele"])
    

#     return

def RunMuonLosses(): 

    lastEle = 6195
    hists_ = {alias: [] for alias in configs_.values()}
    histsMasked_ = {alias: [] for alias in configs_.values()}

    for config, alias in configs_.items():

        finName = f"../../Data/Partial/{config}/muon_lost.dat"
        finName = f"../../../output/{stats}/{config}/muon_lost.dat"
        print(f"---> Analysing {finName}")

        data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

        # ut.Plot1D(data=data["ele"], nbins=lastEle, xmin=0, xmax=lastEle, title=alias, xlabel="Lattice element", ylabel="Muon losses / element", fout=f"../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_{config}.png") 

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
        # ../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_overlay_pm0.002.png

    selectedHistsMasked_ = {}
    # norm = histsMasked_[configs_["NoWedge"]]
    for alias, hist_data in reversed(histsMasked_.items()): # go in reverse so the overlay is clearer
        # It's messy to use the aliases like this, but it works
        if alias in [configs_["NoWedge"], configs_["Minus10mm"], configs_["Minus5mm"], configs_["0mm"], configs_["Plus5mm"], configs_["Plus10mm"]]: 
            selectedHistsMasked_[alias] = hist_data # / norm

    # ut.Plot1DOverlay(hists_, nbins=lastEle, xmin=0, xmax=lastEle, xlabel="Lattice element", ylabel="Muon losses / element", fout=f"../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_overlay_all.png")  
    # print(selectedHists_)
    ut.Plot1DLossesOverlay(selectedHists_, nbins=lastEle, xmin=0, xmax=lastEle, xlabel="Beamline element ID", ylabel="Muon losses / element", fout=f"../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_overlay.png")  
    ut.Plot1DLossesOverlay(selectedHists_, nbins=lastEle-5990, xmin=5990, xmax=lastEle, xlabel="Beamline element ID", ylabel="Muon losses / element", fout=f"../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_overlay_range.png")  
    ut.Plot1DLossesOverlay(selectedHistsMasked_, nbins=lastEle, xmin=0, xmax=lastEle, title=r"$|\Delta p / p_{0}| \leq 0.2\%$", xlabel="Beamline element ID", ylabel="Muon losses / element", fout=f"../../Images/PhaseSpace/{stats}/Losses/h1_muons_losses_overlay_pm0.002.png")  
     

    return


def RunSpinPolarisation(config="NoWedge", ele="end"):

    # --------------------
    # Get data
    # --------------------

    # File names
    finNameEnd = f"../../Data/Partial/{config}/muon_all_end.dat"

    print(f"---> Analysing {finNameEnd}")

    # Read DataFrames
    dataEnd = pd.read_csv(finNameEnd, delim_whitespace=True, header=None, names=columns_)

    # Define spin components
    sx = dataEnd["sx"]
    sy = dataEnd["sy"]
    sz = dataEnd["sz"]

    s_ = np.sqrt(sx**2 + sy**2 + sz**2)

    print(np.min(s_), np.max(s_))

    # ut.Plot1D(s_, nbins=200, xmin=0, xmax=2, title="End of M5, no wedge", xlabel="Spin polarisation", ylabel=r"$\mu^{+}$ / 0.01", fout=f"../../Images/PhaseSpace/{stats}/{config}/h1_muons_end_spin_total.png") 
    ut.Plot1DOverlay({ r"$s_{x}$" : sx , r"$s_{y}$" : sy ,  r"$s_{z}$" : sz}, nbins=200, xmin=-1.0, xmax=1.0, title="End of M5, no wedge", xlabel="Spin polarisation", ylabel=r"$\mu^{+}$ / 0.01", fout=f"../../Images/PhaseSpace/{stats}/{config}/h1_muons_end_spin_components.png", includeBlack=False, colours_extended=False) 

    print("sx", "sy", "sz")
    print(np.mean(sx), np.mean(sy), np.mean(sz))

    tot = len(sx)
    sx_A = np.sum(sx) / tot
    sy_A = np.sum(sy) / tot
    sz_A = np.sum(sz) / tot

    print(sx_A, sz_A)

    # Define unit vector in the y-direction
    # y_unit_vector = np.array([0, 1, 0])

    # # Compute dot products
    # dot_sx_y = np.dot(sx, y_unit_vector)
    # dot_sy_y = np.dot(sy, y_unit_vector)
    # dot_sz_y = np.dot(sz, y_unit_vector)

    # # Plot the dot products
    # ut.Plot1DOverlay({r"$s_{x}\cdot y$" : dot_sx_y,
    #                 r"$s_{y}\cdot y$" : dot_sy_y,
    #                 r"$s_{z}\cdot y$" : dot_sz_y},
    #                 nbins=200,
    #                 xmin=-1.0,
    #                 xmax=1.0,
    #                 title="End of M5, no wedge",
    #                 xlabel="Spin polarisation",
    #                 ylabel=r"$\mu^{+}$ / 0.01",
    #                 fout=f"../../Images/PhaseSpace/{stats}/{config}/h1_muons_end_spin_dot_y.png",
    #                 includeBlack=False,
    #                 colours_extended=False)

    return

# import matplotlib.pyplot as plt




# def PlotH(sim, data): 

#     # Create figure and axes
#     fig, ax = plt.subplots()
#     colour="black"
#     logY=False
#     counts, bin_edges, _ = ax.hist(sim["x"]*1e3, bins=48, range=(-48, 48), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=True, color=colour, label="Sim", log=logY)


#     # Plot scatter with error bars
#     xerr=[] 
#     yerr=[]
#     if len(xerr)==0: xerr = [0] * data["x"] # Sometimes we only use yerr
#     if len(yerr)==0: yerr = [0] * data["y"] # Sometimes we only use yerr

#     ax.errorbar(x=data["x"], y=(data["y"]/np.max(data["y"]))*np.max(counts), xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None')

#     # for i in range(len(dataHPWC["x"])):
#     #     ax.plot([dataHPWC["x"][i], dataHPWC["x"][i]], [dataHPWC["y"][i], 0], color='black', linestyle='-')

#     plt.savefig("../../Images/PWCs/HOR.png", dpi=300, bbox_inches="tight")

#     print("Plotted.")

#     return

# def PlotV(sim, data): 

#     # Create figure and axes
#     fig, ax = plt.subplots()
#     colour="black"
#     logY=False
#     counts, bin_edges, _ = ax.hist(sim["y"]*1e3, bins=48, range=(-48, 48), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=True, color=colour, label="Sim", log=logY)


#     # Plot scatter with error bars
#     xerr=[] 
#     yerr=[]
#     if len(xerr)==0: xerr = [0] * data["x"] # Sometimes we only use yerr
#     if len(yerr)==0: yerr = [0] * data["y"] # Sometimes we only use yerr

#     ax.errorbar(x=data["x"], y=(data["y"]/np.max(data["y"]))*np.max(counts), xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None')

#     # for i in range(len(dataHPWC["x"])):
#     #     ax.plot([dataHPWC["x"][i], dataHPWC["x"][i]], [dataHPWC["y"][i], 0], color='black', linestyle='-')

#     plt.savefig("../../Images/PWCs/VER.png", dpi=300, bbox_inches="tight")

#     print("Plotted.")

#     return


def RunComparePWCs(config="NoWedge", date="JAN20", pions=False):

    print("\n---> RunComparePWCs():")

    configAlias = ""
    if config=="NoWedge":
        configAlias += "NOWEDGE"
    else: 
        configAlias += "WEDGEIN"

    # Get data
    finNameEnd = f"../../../output/{stats}/{config}/muon_all_end.dat"
    if pions: finNameEnd = f"../../../output/{stats}/{config}/pion_end.dat"
    finNameHPWC = f"../../../output/PWCs/PWC025_{configAlias}_{date}_HOR.csv"
    finNameVPWC = f"../../../output/PWCs/PWC025_{configAlias}_{date}_VER.csv"

    print(f"---> Analysing:\n{finNameEnd}\n{finNameHPWC}\n{finNameVPWC}")

    # Read DataFrames
    dataEnd = pd.read_csv(finNameEnd, delim_whitespace=True, header=None, names=columns_)
    dataHPWC = pd.read_csv(finNameHPWC) 
    dataVPWC = pd.read_csv(finNameVPWC) 

    pionTag = ""
    if pions: pionTag += "_pion"

    ut.PlotHistAndGraph(dataEnd["x"], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config], xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025H_{date}{pionTag}.png")
    ut.PlotHistAndGraph(dataEnd["y"], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config], xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025V_{date}{pionTag}.png")

    ut.PlotHistAndGraphWithRatio(dataEnd["x"], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config], xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_ratio_{config}_PWC025H_{date}{pionTag}.png")
    ut.PlotHistAndGraphWithRatio(dataEnd["y"], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config], xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_ratio_{config}_PWC025V_{date}{pionTag}.png", limitRatio=True)

    # mask = (dataEnd["pz"] <= 0.002) & (dataEnd["pz"] >= -0.002)

    # ut.PlotHistAndGraph(dataEnd["x"][mask], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config]+", $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025H_{date}_pzmask_pion.png")
    # ut.PlotHistAndGraph(dataEnd["y"][mask], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config]+", $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025V_{date}_pzmask_pion.png")

    # Doesn't really belong here...
    # Compare xy vs pz 
    # ut.Plot2D(x=dataEnd["x"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_pz_vs_x_PWC025_pion.png")
    # ut.Plot2D(x=dataEnd["y"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_pz_vs_y_PWC025_pion.png")

    # ut.Plot2DWith1DProj(x=dataEnd["x"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_wproj_pz_vs_x_PWC025_pion.png")
    # ut.Plot2DWith1DProj(x=dataEnd["y"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_wproj_pz_vs_y_PWC025_pion.png")



    return

def RunInjectionAngle(config="NoWedge"):

    print("\n---> RunInjectionAngle():")

    # Get data
    finNameEnd = f"../../../output/{stats}/{config}/muon_all_end.dat"

    print(f"---> Analysing:\n{finNameEnd}") 

    # Read DataFrames
    dataEnd = pd.read_csv(finNameEnd, delim_whitespace=True, header=None, names=columns_)

    # Convert to MeV/c
    if True: 
        p_magic = 3094
        dataEnd["px"] = dataEnd["px"] * p_magic
        dataEnd["py"] = dataEnd["py"] * p_magic
        dataEnd["pz"] = dataEnd["pz"] * p_magic + p_magic

    # Plot injection angle
    dataEnd["pT"] = np.sqrt(pow(dataEnd["px"], 2) + pow(dataEnd["py"], 2))
    dataEnd["p"] = np.sqrt(pow(dataEnd["px"], 2) + pow(dataEnd["py"], 2) + pow(dataEnd["pz"], 2))

    mask = (dataEnd["pz"] <= (0.002* p_magic + p_magic)) & (dataEnd["pz"] >= (-0.002 * p_magic + p_magic))
    
    if True:
        ut.Plot2DWith1DProj(x=dataEnd["p"], y=dataEnd["pT"]/dataEnd["p"], nBinsX=100, xmin=2980, xmax=3220, nBinsY=100, ymin=0, ymax=0.0175, title=configs_[config], ylabel=r"$p_{T} / p$", xlabel=r"p [MeV/c]", fout=f"../../Images/PhaseSpace/{stats}/InjectionAngle/h2_injection_angle_muons_{config}_MeV.png") 
        ut.Plot2DWith1DProj(x=dataEnd["p"][mask], y=dataEnd["pT"][mask]/dataEnd["p"][mask], nBinsX=100, xmin=-0.002 * p_magic + p_magic, xmax=0.002* p_magic + p_magic, nBinsY=100, ymin=0, ymax=0.0175, title=configs_[config], ylabel=r"$p_{T} / p$", xlabel=r"p [MeV/c]", fout=f"../../Images/PhaseSpace/{stats}/InjectionAngle/h2_injection_angle_muons_{config}_MeV_accepted.png") 
        
        # x-profiles
        x, xerr, y, yerr, yrms = ut.ProfileX(x=dataEnd["p"], y=dataEnd["pT"]/dataEnd["p"], nBinsX=100, xmin=2980, xmax=3220, nBinsY=100, ymin=0, ymax=0.0175)
        ut.PlotGraph(x, y, xerr, yerr, title=configs_[config], ylabel=r"$p_{T} / p$", xlabel=r"p [MeV/c]", fout=f"../../Images/PhaseSpace/{stats}/InjectionAngle/px_injection_angle_muons_{config}_MeV.png")

        x, xerr, y, yerr, yrms = ut.ProfileX(x=dataEnd["p"][mask], y=dataEnd["pT"][mask]/dataEnd["p"][mask], nBinsX=100, xmin=-0.002 * p_magic + p_magic, xmax=0.002* p_magic + p_magic, nBinsY=100, ymin=0, ymax=0.0175)
        ut.PlotGraph(x, y, xerr, yerr, title=configs_[config], ylabel=r"$p_{T} / p$", xlabel=r"p [MeV/c]", fout=f"../../Images/PhaseSpace/{stats}/InjectionAngle/px_injection_angle_muons_{config}_MeV_accepted.png") 
   
    else: 
        ut.Plot2DWith1DProj(x=dataEnd["p"], y=dataEnd["pT"]/dataEnd["p"], nBinsX=50, xmin=0, xmax=0.05, nBinsY=100, ymin=0, ymax=1.00, title=configs_[config], ylabel=r"$p_{T} / p$", xlabel=r"p", fout=f"../../Images/PhaseSpace/{stats}/InjectionAngle/px_injection_angle_muons_{config}.png", logZ=True) 

    # ut.PlotHistAndGraph(dataEnd["x"], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config], xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025H_{date}{pionTag}.png")
    # ut.PlotHistAndGraph(dataEnd["y"], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config], xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025V_{date}{pionTag}.png")

    # ut.PlotHistAndGraphWithRatio(dataEnd["x"], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config], xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_ratio_{config}_PWC025H_{date}{pionTag}.png")
    # ut.PlotHistAndGraphWithRatio(dataEnd["y"], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config], xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_ratio_{config}_PWC025V_{date}{pionTag}.png", limitRatio=True)

    # mask = (dataEnd["pz"] <= 0.002) & (dataEnd["pz"] >= -0.002)

    # ut.PlotHistAndGraph(dataEnd["x"][mask], dataHPWC, labels=["Bmad", "PWC025H"], title=configs_[config]+", $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="x [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025H_{date}_pzmask_pion.png")
    # ut.PlotHistAndGraph(dataEnd["y"][mask], dataVPWC, labels=["Bmad", "PWC025V"], title=configs_[config]+", $|\Delta p / p_{0}| \leq 0.2\%$", xlabel="y [mm]", ylabel="$\mu^{+}$ [normalised to max]", fout=f"../../Images/PWCs/{stats}/gr_h1_overlay_{config}_PWC025V_{date}_pzmask_pion.png")

    # Doesn't really belong here...
    # Compare xy vs pz 
    # ut.Plot2D(x=dataEnd["x"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_pz_vs_x_PWC025_pion.png")
    # ut.Plot2D(x=dataEnd["y"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_pz_vs_y_PWC025_pion.png")

    # ut.Plot2DWith1DProj(x=dataEnd["x"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="x [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_wproj_pz_vs_x_PWC025_pion.png")
    # ut.Plot2DWith1DProj(x=dataEnd["y"]*1e3, y=dataEnd["pz"], nBinsX=160, xmin=-40, xmax=40, nBinsY=140, ymin=-0.07, ymax=0.07, title=configs_[config], xlabel="y [mm]", ylabel=r"$\Delta p / p_{0}$", fout=f"../../Images/PWCs/{stats}/h2_wproj_pz_vs_y_PWC025_pion.png")
    return

def RunDispersion(config="0mm", ele="before_wedge", maskMom=False): 

    print("\n---> RunDispersion():")

    # Get data
    finName = f"../../../output/{stats}/{config}/muon_{ele}.dat"

    print(f"---> Analysing:\n{finName}") 

    # Read DataFrames
    data = pd.read_csv(finName, delim_whitespace=True, header=None, names=columns_)

    ut.Plot1DOverlay({"Accepted" : data[(data["pz"] <= 0.002) & (data["pz"] >= -0.002)]*1e3, "All" : data["x"]*1e3},  nbins=96, xmin=-48.0, xmax=48.0, xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h1_overlay_x_{config}_{ele}.png", logY=True)
     

    maskMomTag = ""
    if maskMom: 
        # Mask +/- 0.2% 
        data = data[(data["pz"] <= 0.002) & (data["pz"] >= -0.002)]
        maskMomTag += "_accepted"

    # Plot injection angle
    # dataEnd["pT"] = np.sqrt(pow(dataEnd["px"], 2) + pow(dataEnd["py"], 2))
    # dataEnd["p"] = np.sqrt(pow(dataEnd["px"], 2) + pow(dataEnd["py"], 2) + pow(dataEnd["pz"], 2))

    # def Plot3D(x, y, z, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, zmax=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, fout="3d_plot.png", contours=False, cb=True, NDPI=300):

    # nbins=70, xmin=-0.07, xmax=0.07
    ut.Plot1D(data["x"]*1e3,  nbins=96, xmin=-48.0, xmax=48.0, xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h1_x_{config}_{ele}{maskMomTag}.png", stats=False, logY=True)
    # ut.Plot1DOverlay(data["x"]*1e3,  nbins=96, xmin=-48.0, xmax=48.0, xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h1_x_{config}_{ele}{maskMomTag}.png", stats=False, logY=True)
     
    # ut.Plot2D(x=data["x"]*1e3, y=data["y"]*1e3, nBinsX=96, xmin=-48.0, xmax=48.0, nBinsY=96, ymin=-48.0, ymax=48.0, xlabel="x [mm]", ylabel="y [mm]", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h2_xy_{config}_{ele}{maskMomTag}.png") # , nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/PhaseSpace/{stats}/h1_muons_{ele}_pz_{config}.png") 
    # ut.Plot2D(x=data["x"]*1e3, y=data["pz"], nBinsX=96, xmin=-48.0, xmax=48.0, nBinsY=70, ymin=-0.07, ymax=0.07, xlabel="x [mm]", ylabel="$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h2_xpz_{config}_{ele}{maskMomTag}.png")
    # ut.Plot2D(x=data["y"]*1e3, y=data["pz"], nBinsX=96, xmin=-48.0, xmax=48.0, nBinsY=70, ymin=-0.07, ymax=0.07, xlabel="y [mm]", ylabel="$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h2_ypz_{config}_{ele}{maskMomTag}.png")
    # ut.Plot3D(x=data["x"]*1e3, y=data["y"]*1e3, z=data["pz"], nBinsX=96, xmin=-48.0, xmax=48.0, nBinsY=96, ymin=-48.0, ymax=48.0, zmax=np.max(data["pz"]), xlabel="x [mm]", ylabel="y [mm]", zlabel="$\Delta p / p_{0}$", fout=f"../../Images/PhaseSpace/{stats}/Dispersion/h3_xypz_{config}_{ele}{maskMomTag}.png") # , nbins=28, xmin=-0.07, xmax=0.07, title=alias, xlabel="$\Delta p / p_{0}$", ylabel="Muons / 0.005", fout=f"../../Images/PhaseSpace/{stats}/h1_muons_{ele}_pz_{config}.png") 

    return

# --------------------
# Main
# --------------------

def main():


    # RunSingleOffset(ele="END", offset="NoWedge",  MeV=True)

    # [RunSingleOffset(ele=ele, offset="0mm",  MeV=True) for ele in ["end", "after_wedge"]]

    # [RunOffsetScan(ele) for ele in ["end", "after_wedge"]]
    # RunOffsetScan("end")

    # RunWedgeCooling("0mm") 
    # RunWedgeCooling("Plus10mm")
    # RunWedgeCooling("Minus10mm")

    # RunMuonLosses()

    # RunComparePWCs("NoWedge", "JAN20", pions=True)
    # RunComparePWCs("Minus8mm", "JUN21")
    # RunInjectionAngle("NoWedge")
    # RunInjectionAngle("Minus8mm")

    # RunDispersion(config="", ele="before_wedge", maskMom=True)

    RunDispersion(config="NoWedge", ele="after_wedge", maskMom=False)
    RunDispersion(config="NoWedge", ele="after_wedge", maskMom=True)

    return

if __name__ == "__main__":
    main()