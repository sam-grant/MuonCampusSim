# Samuel Grant 2024

# External libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Internal libraries
# import Utils as ut
import TheoreticalSigmas as ts
import BetheBloch as bb

# Define the colourmap colours
colours = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
    (1.0, 0.4980392156862745, 0.054901960784313725),                # Orange
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),    # Purple
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),   # Cyan
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),   # Pink
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),   # Gray 
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # Yellow
    (255 / 255, 205 / 255, 0 / 255)
]

def RunScattering():

    # Replace 'your_file.csv' with the actual path to your CSV file
    csvFile = "Txt/g4beamline_CopperScatter_100000events_mu+.csv"
    csvFile2 = "Txt/Bmad_LynchDahl_3GeV_antimuon.csv" # really just theory...

    # Get the data from G4beamline
    df = pd.read_csv(csvFile)
    # Get the data from Bmad
    df2 = pd.read_csv(csvFile2)

    # Filter momentum
    momentum = 3000

    # Filter DataFrame for rows with the desired momentum
    df = df[df["p [MeV/c]"] == momentum]

    # print(df)

    # x-values
    x_ = np.linspace(1, 200, 1000)*1e-3 #  np.linspace(df["t [mm]"].min(), df["t [mm]"].max(), 1000)
    
    # Make plot

    # Create figure and axes
    fig, ax = plt.subplots()

    ax.plot(x_*1e3, ts.RossiGreisen(t_0=x_, p=momentum), color=colours[0], linestyle='dashed', linewidth=1.0, label="Rossi-Greisen") # , xerr=xerr, yerr=yerr, fmt='o', color=cmap(i), markersize=2, ecolor=cmap(i), capsize=2, elinewidth=1, linestyle='None', label=label)
    ax.plot(x_*1e3, ts.Highland(t_0=x_, p=momentum), color=colours[1], linestyle='dashed', linewidth=1.0, label="Highland") # , xerr=xerr, yerr=yerr, fmt='o', color=cmap(i), markersize=2, ecolor=cmap(i), capsize=2, elinewidth=1, linestyle='None', label=label)
    ax.plot(x_*1e3, ts.HighlandLynchDahl(t_0=x_, p=momentum), color=colours[5], linestyle='solid', linewidth=1.0, label="Highland-Lynch-Dahl") # , xerr=xerr, yerr=yerr, fmt='o', color=cmap(i), markersize=2, ecolor=cmap(i), capsize=2, elinewidth=1, linestyle='None', label=label)
    ax.plot(x_*1e3, ts.LynchDahl(t_0=x_, p=momentum), color=colours[2], linestyle='solid', linewidth=1.0, label="Lynch-Dahl") # , xerr=xerr, yerr=yerr, fmt='o', color=cmap(i), markersize=2, ecolor=cmap(i), capsize=2, elinewidth=1, linestyle='None', label=label)

    # print(df)
    ax.errorbar(x=np.array(df["t [mm]"]), y=np.array(df["sigma_x [deg]"]), yerr=np.array(df["sigma_x_err [deg]"]), fmt='o', color=colours[10], markersize=3, ecolor=colours[10], capsize=2, elinewidth=1, linestyle='None', label="G4BL (x)")
    ax.errorbar(x=np.array(df["t [mm]"]), y=np.array(df["sigma_y [deg]"]), yerr=np.array(df["sigma_y_err [deg]"]), fmt='o', color=colours[3], markersize=3, ecolor=colours[3], capsize=2, elinewidth=1, linestyle='None', label="G4BL (y)")
    ax.errorbar(x=np.array(df["t [mm]"]), y=np.degrees(np.array(df2["sigma [rad]"])), fmt='o', color=colours[4], markersize=3, ecolor=colours[4], capsize=2, elinewidth=1, linestyle='None', label="Bmad")

    ax.set_title("$\mu^{+}$, "+str(momentum)+" MeV/c, Cu", fontsize=14, pad=10) 
    ax.set_xlabel("Thickness [mm]", fontsize=14, labelpad=10) 
    ax.set_ylabel("$\sigma$ [deg]", fontsize=14, labelpad=10) 


    ax.legend(loc="best", frameon=False, fontsize=13)

    # Save the figure
    fout = "../../Images/FoilAna/gr_scatteringAngle.png"
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    return 

def RunEnergyLoss():

    # # Replace 'your_file.csv' with the actual path to your CSV file
    csvFileG4BL = "Txt/g4beamline_CopperEnergyLoss_100000events_mu+_3.csv" 
    csvFileBmad = "Txt/Bmad_energy_loss_Cu_3000MeV.csv" # 3000 MeV/c, really just theory...

    # Get the data from G4beamline
    df_G4BL = pd.read_csv(csvFileG4BL)
    # Get the data from Bmad
    df_Bmad = pd.read_csv(csvFileBmad)


    # Filter momentum
    momentum = 3000

    # Filter DataFrame for rows with the desired momentum
    df_G4BL = df_G4BL[df_G4BL["p [MeV/c]"] == momentum]

    # print(df)

    # x-values (thickness)
    x_ = np.linspace(1, 200, 1000)*1e-3 #  
    
    # Make plot

    # Create figure and axes
    fig, ax = plt.subplots()

    deltaG4BL = np.array(df_G4BL["delta E [MeV]"])
    deltaG4BLErr = np.array(df_G4BL["delta E error [MeV]"])

    ax.plot(x_*1e3, bb.BetheBloch(t_0=x_, p=momentum), color=colours[2], linestyle='solid', linewidth=1.0, label="Bethe-Bloch") # , xerr=xerr, yerr=yerr, fmt='o', color=cmap(i), markersize=2, ecolor=cmap(i), capsize=2, elinewidth=1, linestyle='None', label=label)
    ax.errorbar(x=np.array(df_G4BL["t [mm]"]), y=deltaG4BL, yerr=deltaG4BLErr, fmt='o', color=colours[3], markersize=3, ecolor=colours[3], capsize=2, elinewidth=1, linestyle='None', label="G4BL")
    ax.errorbar(x=np.array(df_Bmad["t [mm]"]), y=np.array(df_Bmad["delta E [MeV]"]), fmt='o', color=colours[4], markersize=3, ecolor=colours[4], capsize=2, elinewidth=1, linestyle='None', label="Bmad")

    ax.set_title("$\mu^{+}$, "+str(momentum)+" MeV/c, Cu", fontsize=14, pad=10) 
    ax.set_xlabel("Thickness [mm]", fontsize=14, labelpad=10) 
    ax.set_ylabel("$\Delta E$ [MeV]", fontsize=14, labelpad=10) 

    ax.legend(loc="best", frameon=False, fontsize=13)

    # Save the figure
    fout = "../../Images/FoilAna/gr_energyLoss.png"
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    return 

def main():

    RunScattering()
    RunEnergyLoss()

    return 0

if __name__ == "__main__":
    main()
