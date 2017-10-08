import numpy as np
from math import exp, pi


DEBUG_PRINT = False

H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]
SIGMA = 5.67E-8 #Stefan Boltzmann constant

def blackbody_flux_wavelength(T, wvlngth):
    """
    Return the total photon flux in W m-2 s-1 for a blackbody at temperature
    T for the given wavelength, wvlngth. 

    Parameters:
    T - the temperature of the blackbody [K]
    wvlngth - the frequency at which to calculate the photon flux [nm]

    Returns:
    flux - the photon flux at wvlngth [W m-2 nm-1]
    """

    wvlngth = wvlngth*(1.0E-9) #convert from nm to m
    flux = 2.0*H*C**2.0/wvlngth**5.0*1.0/(exp(H*C/(wvlngth*KB*T))-1.0)
    flux = flux*(1.0E-9) #convert back to nm-1
    return flux*pi

def get_blackbody_flux_for_wavelengths(T, wvlngths, \
        star_rad=1.0, orbital_rad=1.0):
    """
    Get the photon fluxes for an array of wavelength values.

    NOTE: this function can be used for calculating the flux a planet receives
          at an orbital distance of orbital_rad. 

    Inputs:
    T - the temperature of the blackbody [K]
    wvlngths - array of wavelengths to calculate fluxes for [nm]
    star_rad - the radius of the star [m]
    orbital_rad - the radius of the orbit [m]

    Returns:
    fluxes - an array of fluxes for the corresponding wavelength values
             in [W m-2 nm-1]
    """

    fluxes = np.zeros_like(wvlngths)
    for i in range(0,len(wvlngths)):
        fluxes[i] = blackbody_flux_wavelength(T, wvlngths[i])*(star_rad/orbital_rad)**2

    return fluxes


def total_flux(wvlngths, fluxes):
    """
    This function will take an array of wavelengths and an array of fluxes then
    return the total flux of the fluxes array.

    Inputs:
    wvlngths - the array of wavelength values [nm]
    fluxes - the array of flux values [W m-2 s-1 nm-1]

    Returns:
    t_flux - the summed fluxes from the fluxes array [Photons m-2 s-1]
    """

    t_flux = 0.0

    width = 0.0
    for i in range(0,len(wvlngths)-1):
        width = abs(wvlngths[i]-wvlngths[i+1])*(1.0E-9) #convert from nm to m
        t_flux += width*fluxes[i]

    #add the last flux measurement using the width of the previous freq.
    t_flux += width*fluxes[-1]

    return t_flux

def luminosity(T):
    """
    Return the total stellar luminosity.

    Input:
    start_wv - the starting wavelength to consider [nm]
    end_wv - the ending wavelength to consider [nm]
    T - the stellar temperature [K]

    Returns:
    L - the total luminosity of the star [W]
    """
    
    star_rad = star_radius_from_temp(T)
    t_flux = SIGMA*T**4 
    L = 4.0*pi*star_rad**2*t_flux
    return L

def out_habitable_zone_dist(T):
    """
    Find the outer edge of the habitable zone based on stellar temperature. 
    This function implements equations 2 and 3 of  Kopparapu et al. 2013.

    Input:
    T - the stellar temperature [K]

    Returns:
    d - the distance to the edge of the outer habitable zone [AU]
    """

    L_sun = 3.828E26 #solar luminosity [W]
    L = luminosity(T)

    Ts = T - 5780.0
    s_eff = 0.3438+\
            5.8942E-5*Ts+\
            1.6558E-9*Ts**2+\
            -3.0045E-12*Ts**3+\
            -5.2983E-16*Ts**4
    d = (L/L_sun/s_eff)**0.5

    if DEBUG_PRINT:
        print("out_habitable_zone_dist(): for T=%0.0f, d=%0.2f AU\n"%(T,d))
    return d

def in_habitable_zone_dist(T):
    """
    Find the inner edge of the habitable zone based on stellar temperature. 
    This function implements equations 2 and 3 of  Kopparapu et al. 2013.

    Input:
    T - the stellar temperature [K]

    Returns:
    d - the distance to the edge of the inner habitable zone [AU]
    """

    L_sun = 3.828E26 #solar luminosity [W] #luminosity(10.0, 3000.0, 5780.0)
    L = luminosity(T)
    Ts = T - 5780.0
    s_eff = 1.0140+\
            8.1774E-5*Ts+\
            1.7063E-9*Ts**2+\
            -4.3241E-12*Ts**3+\
            -6.6462E-16*Ts**4
    d = (L/L_sun/s_eff)**0.5

    if DEBUG_PRINT:
        print("in_habitable_zone_dist(): for T=%0.0f, d=%0.2f AU\n"%(T,d))
    return d

    

def star_radius_from_temp(T):
    """
    Get the estimated radius from the star. This is found using the relation
    found from temp_rad_comp() below

    Returns the star radius in [m]
    """
    rad = 0
    if T >= 4200.0: #the upper limit for the Mann (2015) equation
        #rad = T*0.00018647 + 0.00825597
        rad = 0.000196203*T - 0.134051
    elif T >= 2729.0: #the limit where the Mann equation is valid
        t_mod = T/3500.0
        rad = 10.554-33.7546*t_mod+35.1909*t_mod**2-11.5928*t_mod**3
    elif T >= 2300.0:
        rad = 0.0001*T-0.1389
    return rad*SUN_RAD



#def temp_rad_comp():
#    """
#    This is just a test function to fit the data from wikipedia because I was
#    too lazy to do it by hand... the fit is to determine the relationship 
#    between stellar temp and radius (turns out it's pretty much linear)
#    """
#    #data from https://en.wikipedia.org/wiki/Stellar_classification table
#    temps = [7500.0,6000.0,5200.0,3700.0]
#    rads = [1.4,1.15,0.96,0.7]
#
#    res = np.polyfit(temps,rads,1)
#    print(res)
#
#    test_x = np.linspace(3700,7500,15)
#    test_r = test_x*res[0] + res[1]
#
#    plt.plot(temps,rads, label="data")
#    plt.plot(test_x, test_r, label="fit")
#    plt.legend()
#    plt.show()



def get_photon_flux_for_wavelengths_in_range(wvs, wv_flux, start, end):
    """
    Get the photon flux in photons m-2 and W m-2 for the given wavelength and 
    flux array between the start and end wavelengths.

    Inputs:
    wvs - an array of wavelengths [nm]
    wv_flux - an array of uniformly spaced wavelength fluxes [W m-2 nm-1]
    start - the starting wavelength to consider [nm]
    end - the ending wavelength to consider [nm]

    Returns:
    t_flux - the total flux between start and end [W m-2]
    p_flux - the total photon flux between start and end [photons m-2 s-1]
    """

    start_ind = (np.abs(wvs-start)).argmin()
    end_ind = (np.abs(wvs-end)).argmin()

    if end_ind == len(wvs)-1:
        end_ind -= 1

    t_flux = 0.0
    p_flux = 0.0
    width = 0.0
    for i in range(start_ind,end_ind):
        width = abs(wvs[i]-wvs[i+1])
        t_flux += width*wv_flux[i]
        photon_energy = H*C/(wvs[i]*(1.0E-9))
        p_flux += width*wv_flux[i]/photon_energy

    return t_flux, p_flux



def read_solar_flux_data():
    """
    Plot the data from http://rredc.nrel.gov/solar/spectra/am1.5/

    See the wikipedia plot of the same data at: https://commons.wikimedia.org/wiki/File:Solar_Spectrum.png

    This function returns the wavelengths array, the array of corresponding 
    fluxes at TOA and the corresponding fluxes at the surface.
    """
    filename = "./ASTMG173.csv"
    data = np.loadtxt(filename,delimiter=",", skiprows=2) 
    wavelengths = data[:,0]
    flux = data[:,1]
    #flux_tilt = data[:,2]
    flux_circ = data[:,3]

    return (wavelengths, flux, flux_circ)

