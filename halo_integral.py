import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import math


import time
start_time = time.time()

#################################################################################
#### Script to calculate the halo integral for a Maxwell-Boltzmann ##############
#### velocity distribution ######################################################
#################################################################################

####### constant inputs ########################################################
#E_R = 6*10**(-7)  # GeV for XENON1T
E_R = 1.1*10**(-6) # GeV for Lux
m_A = 122 # GeV, xenon

####### expermintal data #######################################################
m_DM = [2021.4120496] # GeV
xsec = [ 3.09882575248*10**(-45)]  #cm^2

"""
# use this to read the curve from the experiment from file
with open("Expdata.txt", "r") as g:
    lines = g.readlines()[0:1]
    mass   = [line.split()[0] for line in lines]
    xsection = [line.split()[1] for line in lines]

g.close()

m_DM = []
xsec = []

for i in range(0, len(mass)):
    m_DM.append(float(mass[i]))
    xsec.append(float(xsection[i]))
"""

v_circ_exp = 220 # km/s
v_esc_exp = 544 # km/s
rho_loc_exp = 0.3 # GeV/cm^3

len_exp = len(m_DM)

####### uncertainty data from analysis group########################################
with open('chi2_vesc.dat', 'r') as f:
    lines = f.readlines()[1:-1]
    f_v0   = [line.split()[0] for line in lines]
    f_rho0 = [line.split()[1] for line in lines]
    f_vesc = [line.split()[2] for line in lines]
    f_chi2 = [line.split()[3] for line in lines]

f.close()

v_0a   = []
rho_0a = []
v_esca = []
chia = []

for i in range(0, len(f_v0)):
    v_0a.append(float(f_v0[i]))
    rho_0a.append(float(f_rho0[i]))
    v_esca.append(float(f_vesc[i]))
    chia.append(float(f_chi2[i]))

v_0   = []
rho_0 = []
v_esc = []
chi   = []

for j in range(0, len(v_0a)):
    if chia[j] <= 50:
        v_0.append(float(v_0a[j]))
        rho_0.append(float(rho_0a[j]))
        v_esc.append(float(v_esca[j]))
        chi.append(float(chia[j]))

chi_np = np.array(chi)
print("first cut", len(chi_np))
#print("max", max(chi_np))
#print(v_esc)

########## Picking the grid point for the minimal chi2
ind_bf = chi.index(min(chi))
chi_bf = min(chi)
v_0_bf = v_0[ind_bf]
rho_0_bf = rho_0[ind_bf]
v_esc_bf = v_esc[ind_bf]

"""
print("ind_bf", ind_bf)
print("chi_bf", chi_bf)
print("v_0_bf", v_0_bf)
print("rho_0_bf", rho_0_bf)
print("v_esc_bf", v_esc_bf)
"""


####### functions ##############################################################
def normalisation(v_esc, v_circ):
    N_esc = special.erf(v_esc/v_circ)-(2/math.sqrt(math.pi))*(v_esc/v_circ)*math.exp(-(v_esc/v_circ)**2)
    return N_esc

def velocityEarth():
    lamb = 2*math.pi*(t-0.218)
    v_earth_x = np.array([-0.0670, 0.4927, -0.8676])*math.sin(lamb)
    v_earth_y = np.array([-0.9931, -0.1170, 0.01032])
    v_earth_vec = 29.8*(v_earth_x-v_earth_y) # 3-vector
    return v_earth_vec

def velmin(m_chi):
    mu = m_chi * m_A / (m_chi + m_A)
    v_min = math.sqrt(m_A*E_R/(2*mu**2)) * 3*10**5 # km/s
    #vmin.append(v_min)
    return v_min

def MaxwellB(N_esc, v, v_earth_1, v_earth_2, v_earth_3, theta, phi, v_circ):
    f_det = 1/(N_esc*(math.pi*v_circ**2)**(3/2)) * math.exp(-(((v*math.sin(theta)*math.cos(phi)+v_earth_1)**2+(v*math.sin(theta)*math.sin(phi)+v_earth_2)**2+(v*math.cos(theta)+v_earth_3)**2)/(v_circ**2)))
    f_MB.append(f_det)
    return f_det

def halo_integral(v_escape, v_circular):
    min_vel = velmin(m_chi)
    theta_min = 0
    theta_max = math.pi
    phi_min = 0
    phi_max = 2*math.pi
    #v_earth = velocityEarth()
    #v_earth_1 = v_earth[0]
    #v_earth_2 = v_earth[1]
    #v_earth_3 = v_earth[2]
    # use instead solar circular velocity as approximation:
    v_earth_1 = 0
    v_earth_2 = 220
    v_earth_3 = 0
    N_esc = normalisation(v_escape, v_circular)
    if min_vel <= v_escape:
    func =  lambda theta, phi, v: MaxwellB(N_esc, v, v_earth_1, v_earth_2, v_earth_3, theta, phi, v_circular)*v*math.sin(theta)
    else:
        func = lambda theta, phi, v: 0
        print(min_vel, v_escape)
        print("mass too small, so v_min larger than v_escape")
    halo_int = integrate.tplquad(func , min_vel, v_escape, lambda v: phi_min, lambda v: phi_max, lambda v, phi: theta_min, lambda v, phi: theta_max)
    return halo_int[0], min_vel

def chi2_profile(rho_eta, chi2, nbins):
    # Bin in (rho0*eta)
    bins      = np.linspace(np.min(rho_eta), np.max(rho_eta), nbins)
    #print("bins", bins)
    chi2_bins = []
    bins_cent = []
    for i in range(len(bins)-1):
        pos = np.where((rho_eta>bins[i]) & (rho_eta<bins[i+1]))
        #print("pos", pos)
        chi2_bins.append(np.min(chi2[pos]))
        bins_cent.append((bins[i+1]+bins[i])/2)
    # return
    #print("bins_cent", bins_cent)
    #print("chi2_bins", chi2_bins)
    return bins_cent, chi2_bins

####### collected data lists ###################################################
f_MB = []

############# calculation ######################################################
print("m_DM", "\t v_min", "\t v_esc", "\t   v_circ", "\t halo integral")
for n in range(0, len_exp):
    m_chi = m_DM[n]
    eta = []
    eta_selected = []   # halo integral from updated upper limit
    eta_exp = []        # halo integral from experimental data
    m = []
    #vmin = []

    ratio_limit = [] # these are the values to multiply the cross section with to get the limits

    factor_band = []
    ratio_band_min = []
    ratio_band_max = []

    rho0_eta = []
    
####### For the updated upper limit ####
    ### halo integral for BF chi^2
    v_escape = v_esc_bf
    v_circular = v_0_bf
    value = halo_integral(v_esc_bf, v_0_bf)
    inte = value[0]
    #print("inte", inte)
    #print("v_min", value[1])
    #print("v_esc from BF", v_escape)
    if inte == 0:       # if this factor is zero, it breaks the loop => you would divide by zero otherwise (where halo MB is zero)
        continue
    else:
        minimum = value[1]
        m.append(m_chi)
        eta_selected.append(inte)
        #print(m_chi, "\t", "%.1f" % minimum, "\t", v_escape, " ", v_circular, "\t", inte)

    ### halo integral  from the experimental data
    value_exp = halo_integral(v_esc_exp, v_circ_exp)
    inte_exp = value_exp[0]
    if inte_exp == 0:
        continue
    else:
        minimum_exp = value_exp[1] # not needed
        eta_exp.append(inte_exp)
        #print("eta_exp", eta_exp[n])

    ### ratio to multiply the cross section to get an updated limit
    factor = inte*rho_0_bf  #eta_selected[n]*rho_0_bf
    ratio =  factor / (inte_exp*rho_loc_exp)
    ratio_limit.append(ratio)
    #print("ratio", ratio)
    
###### making the uncertainty band #####
    ### calculating rho0 times halo integral  (rho0 * eta)
    for b in range(0, len(rho_0)):
        res = halo_integral(v_esc[b], v_0[b])
        res_eta = res[0]
        if res_eta == 0:
            continue
        else:
            #eta.append(res_eta)
            rho0eta = res_eta*rho_0[b]
            rho0_eta.append(rho0eta)
            #print("Integral counter:", len(rho0_eta))
    rho0_eta_np = np.array(rho0_eta)
    print("cut 2", len(rho0_eta_np))
    print("max rho0eta", max(rho0_eta))
    ### making the chi^2 profile for rho0 * eta
    pr = chi2_profile(rho0_eta_np, chi_np,20)
    rhoeta = pr[0]
    chis = pr[1]
    chis_min = min(chis)
    chi_range = []
    rhoeta_range = []
    #print("after chi2", len(rhoeta))
    #print("max rhoeta", max(rhoeta))
    
    
    ### minimum (rho0 * eta) (BF) of curve
    """
    # should be very close to "factor" from above. For consistency, use "factor" from above instead
    rhoeta_BF = min(pr[0])
    index_rhoeta_BF = rhoeta.index(rhoeta_BF)
    chis_BF = chis[index_rhoeta_BF]
    print("rhoeta_BF", rhoeta_BF)
    print("chis_BF", chis_BF)
    """

    rhoeta_BF = factor
    
    ### selecting the chi^2 subset
    for k in range(0, len(chis)):
        if chis[k] <= chis_min + 4: # 4 comes from 2 sigma
            chi_range.append(chis[k])
            rhoeta_range.append(rhoeta[k])

    """
    ### use this instead if you want to skip chi2 profile (should in principle give same result)
    for k in range(0, len(chi)):
        if chi[k] <= chi_bf + 4:
            chi_range.append(chi[k])
            rhoeta_range.append(rho0_eta[k])
    #print("chi_range", chi_range)
    #print("rhoeta_range", rhoeta_range)
    """   

    ### select (rho0 * eta) max and min from the chi^2 subset
    rhoeta_min = min(rhoeta_range)
    rhoeta_max = max(rhoeta_range)
    #print("rhoeta_min", rhoeta_min)
    #print("rhoeta_max", rhoeta_max)
    #print("rhoeta_range", rhoeta_range)
    #print("chi_range", chi_range)

    factor_band_max = rhoeta_max / rhoeta_BF
    ratio_band_max.append(factor_band_max)
    #print("max", factor_band_max)

    factor_band_min = rhoeta_min / rhoeta_BF
    ratio_band_min.append(factor_band_min)
    #print("min", factor_band_min)
    
    updated_lim = xsec[n] * ratio
    upper_lim = updated_lim * factor_band_max
    lower_lim = updated_lim * factor_band_min
    print
    print("Mass", m_chi)
    print("Cross section", xsec[n])
    print("Updated limit", updated_lim)
    print("Upper limit", upper_lim)
    print("Lower limit", lower_lim)
    #print("chi2_bins", chis)
    #print("bins_cent", rhoeta)

print("--- %.2f seconds ---" % (time.time() - start_time))
