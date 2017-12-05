import Fluxes
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fmin
from random import uniform


DEBUG_PRINT = False

H = 6.626E-34 #Planck constant in [J s]
C = 2.99792458E8 #Speed of light [m s-1]                                          
KB = 1.38E-23 #Boltzmann constant in [m2 kg s-2 K-1]
AU = 1.496E11 #1 AU in [m]
SUN_RAD = 6.957E8 #solar radius [m]
SIGMA = 5.67E-8 #Stefan Boltzmann constant


def get_earth_surface_flux(start,end):
    """
    Get the total incident flux in between the wavelengths start and end
    at the surface of the Earth.
    
    Input:
    start - the beginning wavelength to consider [nm]
    end - the ending wavelength to consider [nm]

    Output:
    t_flux - the total flux at the surface in [W m-2]
    p_flux - the total flux at the surface in [photons m-2 s-1]
    """

    #just get the 1st and 3rd values from read_solar_flux_data()
    wavelengths, flux_circ = Fluxes.read_solar_flux_data()[::2]
    t_flux, p_flux = Fluxes.get_photon_flux_for_wavelengths_in_range(\
            wavelengths, flux_circ, start, end)

    if DEBUG_PRINT:
        print("get_earth_surface_flux(): total flux: %0.2f [W m-2],"\
                " or %2.2e [photons m-2 s-1] between %0.0f and %0.0f nm\n"%\
                (t_flux,p_flux,start,end))

    return t_flux, p_flux


def blackbody_flux(T, R, start, end):
    """
    Get the flux and the photon flux for a blackbody at orbital distance r 
    between the wavelengths start and end.

    Inputs:
    T - the temperature of the star [K]
    R - the radius of the orbit
    start - the start wavelength to consider [nm]
    end - the ending wavelength to consider [nm]

    Output:
    t_flux - the total flux at the surface in [W m-2]
    p_flux - the total flux at the surface in [photons m-2 s-1]
    """

    star_radius = Fluxes.star_radius_from_temp(T)
    wavelengths = np.linspace(start,end,end-start)

    fluxes = Fluxes.get_blackbody_flux_for_wavelengths(T, wavelengths, \
            star_rad=star_radius, orbital_rad=R)

    t_flux, p_flux = Fluxes.get_photon_flux_for_wavelengths_in_range(\
            wavelengths, fluxes, start,end)

    if DEBUG_PRINT:
        print("blackbody_flux(): total flux: %0.2f [W m-2],"\
                " or %2.2e [photons m-2 s-1] between %0.0f and %0.0f nm\n"%\
                (t_flux,p_flux,start,end))

    return t_flux, p_flux


def get_dist_from_flux(flux, T):
    rad = Fluxes.star_radius_from_temp(T)
    SIMGA = 5.67E-8 #Stefan boltzmann constant
    dist = (rad**2*SIMGA*T**4/flux)**0.5
    return dist

def get_outer_HZ_in_flux(T):
    rad = Fluxes.star_radius_from_temp(T)
    dist = Fluxes.out_habitable_zone_dist(T)*AU
    flux = rad**2/dist**2*SIGMA*T**4
    return flux

def get_inner_HZ_in_flux(T):
    rad = Fluxes.star_radius_from_temp(T)
    dist = Fluxes.in_habitable_zone_dist(T)*AU
    flux = rad**2/dist**2*SIGMA*T**4
    return flux

def plot_750nm_limit_contours(CS, ax):
    p_7 = CS.collections[0].get_paths()[0]
    v_7 = p_7.vertices
    xs_7 = v_7[:,0]
    ys_7 = v_7[:,1]
    #fill above the 7% contour first
    ax.fill_between(xs_7,ys_7,4200, facecolor="blue", alpha=0.3)
    #now fill below the contour
    ax.fill_between(xs_7,ys_7,2300, facecolor="red", alpha=0.3)

    p_31 = CS.collections[1].get_paths()[0]
    v_31 = p_31.vertices
    xs_31 = v_31[:,0]
    ys_31 = v_31[:,1]
    ax.fill_between(xs_31,ys_31,4200, facecolor="white", alpha=1.0)

    ax.plot(xs_7[3:-3],ys_7[3:-3],"k")


def plot_900nm_limit_contours(CS, temps, outer_HZ, ax):
    p_31 = CS.collections[1].get_paths()[0]
    v_31 = p_31.vertices
    xs_31 = v_31[:,0]
    ys_31 = v_31[:,1]
    ax.fill_betweenx(temps, outer_HZ, xs_31[-1], facecolor="blue",alpha=0.3, edgecolor="none")
    ax.fill_between(xs_31,ys_31,4200, facecolor="white", alpha=1.0, edgecolor="none")

    p_7 = CS.collections[0].get_paths()[0]
    v_7 = p_7.vertices
    xs_7 = v_7[:,0]
    ys_7 = v_7[:,1]
    #now fill below the contour
    ax.fill_between(xs_7,ys_7,2300, facecolor="white", alpha=1.0, edgecolor="none")
    ax.fill_between(xs_7,ys_7,2300, facecolor="red", alpha=0.3, edgecolor="none")

def plot_1300nm_limit_contours(CS, ax):
    p_31 = CS.collections[1].get_paths()[0]
    v_31 = p_31.vertices
    xs_31 = v_31[:,0]
    ys_31 = v_31[:,1]
    ax.fill_between(xs_31,ys_31,2300, facecolor="blue", alpha=0.3, edgecolor="none")


    
def get_photo_scale_factor(wv, wv_lim):
    """
    Determine how many photons an optimized photosystem will need for the 
    given wavelength limit. Return the scaled efficiency. Example: on Earth
    we have a two photon system with PS1 and PS2, beyond 750nm to 1040nm a 3
    step system is needed (66% as efficient for same number of photons).

    Inputs:
    wv - the optimal pigment wavelength [nm]
    wv_lim - the limit in useable photons for photosynthesis [nm]

    Returns:
    scale_factor - the fraction of effectiveness compare to Earth
    """
    scale_factor = 1.0 
    lim = wv_lim
    if wv < wv_lim:
        lim = wv

    if lim > 1400:
        #above 1400 nm 6 photons are used
        scale_factor = 0.333
    elif lim > 1040:
        #between 1040 and 1400 nm 4 photons are used
        scale_factor = 0.5
    elif lim > 750:
        #between 750 and 1040nm 3 photons are used
        scale_factor = 0.6666

    return scale_factor

def generate_single_plot(ax, temps, fluxes, results, \
        inner_HZ, outer_HZ, earth_flux, axnum):

    contours = [0.07, 0.31]

    CS = ax.contour(fluxes/earth_flux,temps,results,contours, alpha=0)

    fs = 14
    if axnum==1:
        ax.text(0.85,4000, "A", fontsize=fs)
        #plot_750nm_limit_contours(CS, ax)
        plot_900nm_limit_contours(CS, temps, outer_HZ/earth_flux, ax)
    elif axnum==2:
        ax.text(0.85,4000, "B", fontsize=fs)
        plot_900nm_limit_contours(CS, temps, outer_HZ/earth_flux, ax)
    elif axnum==3:
        ax.text(0.85,4000, "C", fontsize=fs)
        plot_900nm_limit_contours(CS, temps, outer_HZ/earth_flux, ax)
    elif axnum==4:
        ax.text(0.85,4000, "D", fontsize=fs)
        plot_1300nm_limit_contours(CS, ax)



    ax.fill_betweenx(temps, inner_HZ/earth_flux, 0.9, facecolor="white")
    ax.fill_betweenx(temps, outer_HZ/earth_flux, 0.2, facecolor="white")

    
    ax.plot(outer_HZ/earth_flux, temps, "k", linewidth="2")
    ax.plot(inner_HZ/earth_flux, temps, "k", linewidth="2")

    ax.plot([0.662],[2559],"ko") #TRAPPIST-1e
    ax.plot([0.382],[2559],"ko") #TRAPPIST-1f
    ax.plot([0.258],[2559],"ko") #TRAPPIST-1g
    ax.text(0.46, 2409, "TRAPPIST-1e,f,g", color="black", horizontalalignment="center")
    #ax.text(0.258, 2659, "g", color="black", horizontalalignment="center")
    #ax.text(0.382, 2659, "f", color="black", horizontalalignment="center")
    #ax.text(0.662, 2659, "e", color="black", horizontalalignment="center")

    ax.plot([0.39],[3131],"ro") #LHS 1140b
    ax.text(0.39, 2981, "LHS 1140b", color="red", horizontalalignment="center")

    ax.plot([0.65],[3050],"bo") #Proxima b
    ax.text(0.65,2900,"Proxima b",color="blue", horizontalalignment="center")


def get_net_oxygen(sink_flux, burial_rate, photon_fraction):
    """
    The calculated outgassing rate (see paper) is ~5 Tmoles/yr on the modern
    Earth. Half of that is done on land, half in the ocean.

    """

    
    land_fraction = 1.0 if photon_fraction > 0.31 else photon_fraction/0.31
    ocean_fraction = 1.0 if photon_fraction > 0.07 else photon_fraction/0.07

    land_contribution = 2.5*land_fraction
    ocean_contribution = 2.5*ocean_fraction



def plot_oxic_vs_anoxic():
    """
    Plot the potential redox state of the atmosphere based on Earth's sources
    and sinks for oxygen.
    """


    earth_p_flux = get_earth_surface_flux(400,700)[1]
    print("Earth photon flux (400-700nm): %2.3e"%(earth_p_flux))
    earth_flux = 1361.0
    albedo = 0.3
    photon_limit = 750.0
    sink_min = 4.5 #the minimum rate in Tera moles of outgased reductant
    sink_max = 6.9 #the max outgased rate in Tera moles of reductants
    burial_min = 0.001 #the min fraction of organic carbon buried
    burial_max = 0.002 #the max fraction of organic carbon buried
    

    temps = np.linspace(2300,4200,30)
    fluxes = np.linspace(0.2*earth_flux,0.9*earth_flux,30) #fluxes in terms of Earth flux
    results = np.zeros((len(fluxes),len(temps)))

    outer_HZ = np.zeros_like(temps)
    inner_HZ = np.zeros_like(temps)

    for i in range(0,len(temps)):
        outer_HZ[i] = get_outer_HZ_in_flux(temps[i])
        inner_HZ[i] = get_inner_HZ_in_flux(temps[i])
        star_rad = Fluxes.star_radius_from_temp(temps[i])

        for j in range(0,len(fluxes)):
            orb = get_dist_from_flux(fluxes[j],temps[i])
            wv = bjorn_opt_pigment(temps[i], star_rad, orb)

            p_flux = blackbody_flux(temps[i],orb,400.0,photon_limit)[1]
            scale_factor = get_photo_scale_factor(wv, photon_limit)
            useable_photon_flux = p_flux/earth_p_flux*(1.0-albedo)*scale_factor

            b_rate = uniform(burial_min, burial_max)
            sink_rate = uniform(sink_min, sink_max)

 

def plot_photo_limited_regions():
    """
    Plot the orbital distance vs temperature with contours showing the percent
    of Earth's photon flux.
    """

    earth_p_flux = get_earth_surface_flux(400,700)[1]
    print("Earth photon flux (400-700nm): %2.3e"%(earth_p_flux))
    earth_flux = 1366.0
    albedo = 0.3
    

    temps = np.linspace(2300,4200,30)
    fluxes = np.linspace(0.2*earth_flux,0.9*earth_flux,30) #fluxes in terms of Earth flux
    results_750nm = np.zeros((len(fluxes),len(temps)))
    results_900nm = np.zeros((len(fluxes),len(temps)))
    results_1100nm = np.zeros((len(fluxes),len(temps)))
    results_1500nm = np.zeros((len(fluxes),len(temps)))

    outer_HZ = np.zeros_like(temps)
    inner_HZ = np.zeros_like(temps)

    for i in range(0,len(temps)):
        outer_HZ[i] = get_outer_HZ_in_flux(temps[i])
        inner_HZ[i] = get_inner_HZ_in_flux(temps[i])
        star_rad = Fluxes.star_radius_from_temp(temps[i])
        for j in range(0,len(fluxes)):
            orb = get_dist_from_flux(fluxes[j],temps[i])
            wv = bjorn_opt_pigment(temps[i], star_rad, orb)

            p_flux = blackbody_flux(temps[i],orb,400.0,750.0)[1]
            scale_factor = get_photo_scale_factor(wv, 750.0)
            results_750nm[i][j] = p_flux/earth_p_flux*(1.0-albedo)*scale_factor
            
            p_flux = blackbody_flux(temps[i],orb,400.0,900.0)[1]
            scale_factor = get_photo_scale_factor(wv, 900.0)
            results_900nm[i][j] = p_flux/earth_p_flux*(1.0-albedo)*scale_factor

            p_flux = blackbody_flux(temps[i],orb,400.0,1100.0)[1]
            scale_factor = get_photo_scale_factor(wv, 1100.0)
            results_1100nm[i][j] = p_flux/earth_p_flux*(1.0-albedo)*scale_factor

            p_flux = blackbody_flux(temps[i],orb,400.0,1500.0)[1]
            scale_factor = get_photo_scale_factor(wv, 1500.0)
            results_1500nm[i][j] = p_flux/earth_p_flux*(1.0-albedo)*scale_factor


    ####################
    # This code is the code for making the isobar plot Tori asked for
    """
    def cont_string(num):
        string = "%2.0f%%"%(num*100)
        return string
    contours = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.4, 0.50, 0.75, 1.0] #ORL TD

    plt.gca().invert_xaxis()

    CS = plt.contour(fluxes/earth_flux,temps,results_750nm,contours) #ORL TD
    plt.clabel(CS, inline=1, fontsize=10, fmt=cont_string, manual=True) #ORL TD

    
    plt.fill_betweenx(temps, inner_HZ/earth_flux, 0.9, facecolor="white")
    plt.fill_betweenx(temps, outer_HZ/earth_flux, 0.2, facecolor="white")

    plt.plot(outer_HZ/earth_flux, temps, "k", linewidth="2")
    plt.plot(inner_HZ/earth_flux, temps, "k", linewidth="2")

    plt.plot([0.662],[2559],"ko") #TRAPPIST-1e
    plt.plot([0.382],[2559],"ko") #TRAPPIST-1f
    plt.plot([0.258],[2559],"ko") #TRAPPIST-1g
    plt.text(0.46, 2409, "TRAPPIST-1e,f,g", color="black", horizontalalignment="center")

    plt.plot([0.39],[3131],"ro") #LHS 1140b
    plt.text(0.39, 2981, "LHS 1140b", color="red", horizontalalignment="center")

    plt.plot([0.65],[3050],"bo") #Proxima b
    plt.text(0.65,2900,"Proxima b",color="blue", horizontalalignment="center")


    plt.xlabel(r"Incident Flux [$S/S_{\oplus}$]")

    plt.ylabel("Stellar Temperature [K]")

    

    plt.show()
    return
    """
    ####################

    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, sharex='col', sharey='row')
    f.subplots_adjust(hspace=0.05, wspace=0.12)
    """
    CS = plt.contour(fluxes/earth_flux,temps,results,contours, alpha=0)

    if photo_limit == 750:
        plot_750nm_limit_contours(CS)
    elif photo_limit == 900:
        plot_900nm_limit_contours(CS, temps, outer_HZ/earth_flux)
    elif photo_limit == 1100:
        plot_900nm_limit_contours(CS, temps, outer_HZ/earth_flux)
    elif photo_limit > 1300:
        plot_1300nm_limit_contours(CS)
    else:
        CS = plt.contour(fluxes/earth_flux,temps,results)
        plt.clabel(CS, inline=1, fontize=12)
    """
    
    generate_single_plot(ax1, temps, fluxes, results_750nm, \
        inner_HZ, outer_HZ, earth_flux, 1)

    generate_single_plot(ax2, temps, fluxes, results_900nm, \
        inner_HZ, outer_HZ, earth_flux, 2)

    generate_single_plot(ax3, temps, fluxes, results_1100nm, \
        inner_HZ, outer_HZ, earth_flux, 3)

    generate_single_plot(ax4, temps, fluxes, results_1500nm, \
        inner_HZ, outer_HZ, earth_flux, 4)





    ax1.invert_xaxis()
    ax2.invert_xaxis()

    ax4.set_xlabel(r"Incident Flux [$S/S_{\oplus}$]")
    ax3.set_xlabel(r"Incident Flux [$S/S_{\oplus}$]")

    ax1.set_ylabel("Stellar Temperature [K]")
    ax3.set_ylabel("Stellar Temperature [K]")

    

    plt.show()


def bjorn_opt_pigment(T_star, star_rad, orb_rad):
    """
    Calculate the optimal pigment following Bjorn (1976).

    Inputs:
    T_star - the stellar temperature [K]
    star_rad - the radius of the star [m]
    orb_rad - the orbital radius of the planet [m]

    Output:
    opt_wv - the optimized absorption wavelength [nm]
    """

    phi = 0.33 #from Bjorn (1976)
    T = 300.0 # the temperature of the plants

    def power_solver_eqn(nu):
        if nu < 1.0E14 or nu > 8.0E14: #no solution is outside these bounds
            return 10 #just to get the solver back on track
        u_0 = KB*T*log(phi*star_rad**2/(4.0*orb_rad**2))+H*nu*\
                (1.0-T/T_star)
        u = u_0 + KB*T*log(KB*T/(u_0+KB*T))
        u_eff = u/(1.0+KB*T/u)

        power = u_eff*nu**2*exp(-H*nu/(KB*T_star))
        return -power

    opt_nu = fmin(power_solver_eqn, 6.7E14, disp=False)
    return C/opt_nu*(1.0E9)


def bjorn_pigment_model_over_temp():
    """
    Calculate the optimal pigment color for a planet with Earth-like flux
    around another star
    """

    temps = np.linspace(2500,7500,50)
    pigs = np.zeros_like(temps)

    i, star_rad, orb_rad = (0,0,0) #initialize them to make the warning go away

    for i in range(0,len(temps)):
        #star_rad = (0.00018647*temps[i]+0.00825597)*SUN_RAD
        star_rad = Fluxes.star_radius_from_temp(temps[i])
        orb_rad = (star_rad**2*SIGMA*temps[i]**4/1366.0)**0.5

        #convert the pigment from nm to microns and save it
        pigs[i] = bjorn_opt_pigment(temps[i], star_rad, orb_rad)/(1.0E3)

    plot_pigments_over_temp(temps,pigs)

def plot_pigments_over_temp(temps, pigs):
    height = 0.06
    base = np.min(pigs)-height

    plt.gca().add_patch(patches.Rectangle((2500, base),1200,height, alpha=0.5,\
            edgecolor="none", facecolor="red"))
    plt.gca().add_patch(patches.Rectangle((3700, base),1500,height, alpha=0.5,\
            edgecolor="none", facecolor="orange"))
    plt.gca().add_patch(patches.Rectangle((5200, base),800,height, alpha=0.5,\
            edgecolor="none", facecolor="yellow"))
    plt.gca().add_patch(patches.Rectangle((6000, base),1500,height, alpha=0.5,\
            edgecolor="none", facecolor="#ffff99"))

    plt.gca().text(3100,base+height/2.0,"M", verticalalignment="center", \
            horizontalalignment="center")
    plt.gca().text(4450,base+height/2.0,"K", verticalalignment="center", \
            horizontalalignment="center")
    plt.gca().text(5600,base+height/2.0,"G", verticalalignment="center", \
            horizontalalignment="center")
    plt.gca().text(6750,base+height/2.0,"F", verticalalignment="center", \
            horizontalalignment="center")

    plt.plot((2550,2550),(base+height,np.max(pigs)), "k:", linewidth=2, \
            label="TRAPPIST-1")
    plt.plot((3042,3042),(base+height,np.max(pigs)), "k--", linewidth=2, \
            label="Proxima Centauri")
    plt.plot((5778,5778),(base+height,np.max(pigs)), "k-.", linewidth=2, \
            label="Sun")


    plt.plot(temps,pigs, "k", linewidth=2)
    plt.xlim(np.min(temps),np.max(temps))
    plt.ylim(base,np.max(pigs))
    plt.xlabel("Stellar Temperature [K]")
    plt.ylabel(r"Optimal Pigment Absorption [$\mathrm{\mu}$m]")
    plt.legend()
    plt.show()

def test_rad():
    temps = np.linspace(2300,7500,100)
    rad = np.zeros_like(temps)

    for i in range(0,len(temps)):
        r = Fluxes.star_radius_from_temp(temps[i])
        rad[i] = r

    plt.plot(temps,rad/SUN_RAD,label="rad", linewidth=2)
    plt.xlabel("Temperature [K]")
    plt.ylabel(r"Radius [$R_{\odot}$]")
    plt.xlim(2300,7500)
    plt.show()


#ORL - these functions generated plots for the paper
plot_photo_limited_regions()
#bjorn_pigment_model_over_temp()
plot_oxic_vs_anoxic()

#this function plots our stellar temperature radius relationship
#test_rad()

Temp = 2300.0
print(Fluxes.star_radius_from_temp(Temp))
d = Fluxes.out_habitable_zone_dist(Temp)

a_t, a_p = get_earth_surface_flux(400,750)
b_t, b_p = blackbody_flux(Temp, d*AU, 400,1100)

print("planet has: %0.2f%% of Earth photon flux"%(b_p*0.7/a_p*100.0))
