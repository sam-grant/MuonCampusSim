import numpy as np

# Constants
c = 299792458.0  # Speed of light in m/s
e = 1.602176634e-19  # Elementary charge in C
m_e = 9.10938356e-31  # Electron mass in kg
N_a = 6.022e23 # Avogadro mol^-1    
epsilon_0 = 8.8541878128e-12 # vacuum permittivity 

def GetBeta(p=3000, m_0=105.7):
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

def BetheBloch(t_0=0.05, p=3000, m_0=105.7, z=1, Z=29, A=63.546, rho=8.96):
    # Speed factor 
    beta = GetBeta(p, m_0)
    
    # Constant
    K = (4 * np.pi) / (m_e * pow(c,2)) 

    # Electron number density
    n = (N_a * Z * rho) / A 
    n = n * (1/0.01**3) # cm^-3 --> m^-3

    # Mean excitation energy 
    I = 11.1 * Z * e # 11.1 eV is an approximation for copper 

    # Bethe-Bloch terms
    S1 = (n * pow(z,2)) / pow(beta, 2)
    S2 = pow( ( pow(e, 2) / (4*np.pi*epsilon_0) ), 2)
    S3 = np.log( ( (2 * m_e * pow(c,2) * pow(beta,2)) / (I * (1-pow(beta,2))) ) ) - pow(beta,2) 

    # Mean stopping power 
    dE_over_dx = - K * S1 * S2 * S3 # J/m 
    dE_over_dx = dE_over_dx / e  # eV/m
    dE_over_dx = dE_over_dx * 1e-6 # MeV

    # Delta E in MeV 
    deltaE = dE_over_dx * t_0 

    # Convert to MeV 
    return deltaE

# Starting parameters for mu+ through 5 cm of copper 
# t_0=0.05 # thickness in m 
# p=3000 # initial momentum in MeV/c
# z=+1 # charge
# m_0=105.7 # m_0=0.5109989461 # electron/positron
# beta = GetBeta(p, m_0)
# rho=8.96
# Z=29 
# A=63.546 # g/mol

# print("---")
# print(f"{p} MeV e+ through {t_0*1e2} cm of Cu")
# print("---")
# print("Energy loss = ", BetheBloch(t_0=t_0, z=z, Z=Z, A=A, rho=rho, beta=beta),"MeV")
# print("---")