import numpy as np
from PlanetModel import Planet
from math import exp, pi, sin

H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]


def blackbody_flux(T, wvlngth):
    """
    Return the total photon flux in W m-2 s-1 for a blackbody at temperature
    T for the given wavelength, wvlngth. 

    Parameters:
    T - the temperature of the blackbody [K]
    wvlngth - the frequency at which to calculate the photon flux [nm]

    Returns:
    flux - the photon flux at wvlngth [W m-2 s-1 nm-1]
    """

    wvlngth = wvlngth*(1.0E-9) #convert from nm to m
    flux = 2.0*H*C**2.0/wvlngth**5.0*1.0/(exp(H*C/(wvlngth*KB*T))-1.0)
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
             in [W m-2 s-1 nm-1]
    """

    fluxes = np.zeros_like(wvlngths)
    for i in range(0,len(wvlngths)):
        fluxes[i] = blackbody_flux(T, wvlngths[i])*(star_rad/orbital_rad)**2

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

def luminosity(start_wv, end_wv, T):
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
    wavelengths = np.linspace(start_wv, end_wv, 200)
    fluxes = get_blackbody_flux_for_wavelengths(T, wavelengths)
    t_flux = total_flux(wavelengths, fluxes)
    L = 4.0*pi*star_rad**2*t_flux
    return L


def show_planet_flux(star_temp, orb_dist, planet_radius):
    """
    Generate the data that will be used to show the incident flux on a planet
    """
    plnt = Planet(planet_radius, 50)

    star_radius = star_radius_from_temp(star_temp)

    wavelengths = np.linspace(30,3500, 150)
    fluxes = get_blackbody_flux_for_wavelengths(star_temp, wavelengths,\
            star_rad=star_radius, orbital_rad=orb_dist)

    t_flux = total_flux(wavelengths, fluxes) #total flux in W m-2 

    start_color = (0,0,255)
    end_color = (255,0,0)
    for panel in plnt.panels:
        angle = panel.get_zenith_angle()

        val = -1
        if angle >= 0:
            val = abs(t_flux*sin(pi-angle))

        panel.set_color(linear_color_gradient(start_color, end_color, 0, \
                t_flux, val))

    plnt.generate_planet_data()



def rgb_to_hex(rgb):
    r,g,b = rgb
    return "#%02x%02x%02x"%(r,g,b)

def color_from_ratio(s_col, e_col, ratio):
    rs,gs,bs = s_col
    re,ge,be = e_col

    r = rs*(1.0-ratio) + re*ratio
    g = gs*(1.0-ratio) + ge*ratio
    b = bs*(1.0-ratio) + be*ratio

    return (r,g,b)

def linear_color_gradient(start_color, end_color, min_val, max_val, val):
    """
    calculate a linear color gradient between the start and end colors.

    Inputs:
    start_color - the color associated with the min_val [hex color]
    end_color - the color associated with the max_val [hex color]
    min_val - the minimum value [float]
    max_val - the maximum value [float]
    val - the value a color is needed for [float]
    
    Returns:
    color - the hex color of for the value, val [hex color string]
    """

    color = 0x000000

    if val >= max_val:
        color = end_color
    elif val <= min_val:
        color = start_color
    else:
        ratio = (val - min_val)/(max_val - min_val)
        color = color_from_ratio(start_color, end_color, ratio)

    return rgb_to_hex(color)

def out_habitable_zone_dist(T):
    """
    Find the outer edge of the habitable zone based on stellar temperature. 
    This function implements equations 2 and 3 of  Kopparapu et al. 2013.

    Input:
    T - the stellar temperature [K]

    Returns:
    d - the distance to the edge of the outer habitable zone [AU]
    """

    L_sun = luminosity(10.0, 3000.0, 5780.0)
    L = luminosity(10.0,3000.0,T)
    Ts = T - 5780.0
    s_eff = 0.3179+\
            5.413E-5*Ts+\
            1.5313E-9*Ts**2+\
            -2.7786E-12*Ts**3+\
            -4.8997E-16*Ts**4
    d = (L/L_sun/s_eff)**0.5

    print("at d S/S_0 is: %0.2f"%(L/(4.0*pi*(d*AU)**2)/1366.0))
    return d


    

def star_radius_from_temp(T):
    """
    Get the estimated radius from the star. This is found using the relation
    found from temp_rad_comp() below
    """
    return SUN_RAD*T*0.00018647 + 0.00825597

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

def test():
    wavelengths = np.linspace(10,5000,150)
    fluxes = get_blackbody_flux_for_wavelengths(5800.0, wavelengths, star_rad=SUN_RAD,\
            orbital_rad=AU)
    tot = total_flux(wavelengths, fluxes)
    print("Total flux is: %0.2f"%(tot))

#show_planet_flux(3000.0, AU, 20)
