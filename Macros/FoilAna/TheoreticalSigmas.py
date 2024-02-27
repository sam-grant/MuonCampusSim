# Defaults are for 500 MeV mu+ scattering through 5 cm of copper

import numpy as np

# Globals
c = 299792458 # Speed of light in m/s
e = 1.602176634e-19 # Elementary charge in C
    
def GetBeta(p=500, m_0=105.7):

    # E_tot = gamma * m_0 * c2 = sqrt( (pc)**2 + (m_0c**2)**2 )
    # E_rest = m_0 * c**2 
    # gamma = E_tot / E_rest
    # beta = sqrt( 1 - 1/gamma^2)

    # SI units
    p = (p * 1e6 * e) / c
    m_0 = (m_0 * 1e6 * e) / pow(c, 2)

    E_tot = np.sqrt( pow(p*c, 2) + pow(m_0 * pow(c, 2), 2) )
    E_rest = m_0 * pow(c, 2)
    gamma = E_tot / E_rest
    beta = np.sqrt( 1 - (1/pow(gamma, 2)) )

    return beta

def RossiGreisen(t_0=5e-2, X_0=1.436e-2, p=500, m_0=105.7):
    S1 = 15 # MeV / c
    beta = GetBeta(p, m_0)
    return np.degrees( S1 * np.sqrt(t_0/X_0) / (p*beta) ) 

def Highland(t_0, X_0=1.436e-2, p=500, m_0=105.7):
    S2 = 13.6 # 14.1 # MeV / c 
    beta = GetBeta(p, m_0)
    epsilon = .038 # 1./9 
    return np.degrees( (S2/(p*beta)) * np.sqrt(t_0/X_0) * (1 + epsilon * np.log(t_0/X_0)) )

def HighlandLynchDahl(t_0=5e-2, X_0=1.436e-2, p=500, m_0=105.7, z=1):
    S2 = 13.6 # 14.1 # MeV / c 
    beta = GetBeta(p, m_0)
    epsilon = 0.038 # 1./9 
    return np.degrees( ((S2*z)/(p*beta)) * np.sqrt(t_0/X_0) * (1 + epsilon * np.log( (t_0*pow(z,2)) / (X_0*pow(beta,2)) ) ) )

# I aim to use the same units as specified in the Lynch & Dahl paper 
def LynchDahl(t_0=5e-2, p=500, z=1, m_0=105.7, rho=8.96, Z=29, A=63.546, F=0.98): 

    # Speed factor
    beta = GetBeta(p, m_0)
    # print("beta =", beta)

    # Nuclear factor 
    zza = Z * (Z + 1) / A
    
    # Area density
    m2cm = 1e2
    a_rho = rho * t_0 * m2cm # g/cm^2 

    # The path length in strange units, equivilant to the thickness times the area density?.
    # X = t_0 * a_rho # gm/cm^2 

    # According to David, path length = area density in this case. Don't understand this at all.
    X = a_rho # gm/cm^2 

    # The fine structure constant
    alpha = 1./137

    # The characteristic scattering angle squared
    chi2_c = 0.157 * ( zza * X ) * pow( z / (p*beta) , 2)  

    # print("The characteristic angle =", chi2_c)

    # The screening angle squared
    chi2_a = 2.007e-5 * pow(Z, 2/3) * (1 + 3.34 * pow(Z * z * alpha / beta, 2)) / pow(p, 2)

    # Omega, the mean number of scatters
    # Adjusted according to the Geant3 manual
    omega = chi2_c / (1.167 * chi2_a)

    # Nu 
    nu = 0.5 * omega / (1 - F)

    # Scattering sigma squared 
    sigma2 = (chi2_c / ( 1 + pow(F, 2) ) ) * ( ((1+nu)/nu) * np.log(1+nu) - 1)

    sigma = np.sqrt(sigma2) 

    return np.degrees( sigma )


