import Fluxes
import numpy as np

DEBUG_PRINT = True
AU = 1.496E11 #1 AU in [m]

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

    wavelengths, flux, flux_circ = Fluxes.read_solar_flux_data()
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


T = 2500.0
d = Fluxes.out_habitable_zone_dist(T)

get_earth_surface_flux(400,700)
blackbody_flux(T, d*AU, 400,900)
