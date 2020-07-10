import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, chi2
from scipy.linalg import lstsq, inv

from astropy import constants
from astropy import units
import corner

from plotstyles import useagab, apply_tufte
from pygaia.astrometry.coordinates import CoordinateTransformation, Transformations
from pygaia.astrometry import constants as pygcst
from pygaia.astrometry.vectorastrometry import astrometryToPhaseSpace

def ephemeris_earth_simple(t):
    """
    Calculate a very simple ephemeris for "earth" (or "Gaia") in the BCRS. Assume a perfectly
    circular orbit of exactly 1 AU radius with a 1 yr period. The orbit is calculated in Ecliptic
    coordinates and the result is then transformed to the ICRS.
    
    Parameters
    ----------
    
    t : float array
        Times at which to calculate the ephemeris in Julian years.
        
    Returns
    -------
    
    Array of shape (3,t.size) representing the xyz components of the ephemeris at times t.
    """
    ecltoicrs = CoordinateTransformation(Transformations.ECL2ICRS)
    orbitalperiod = 1.0 #(Julian yr)
    orbitalradius = 1.0 #(AU)
    (b_xecl, b_yecl, b_zecl) = (orbitalradius*np.cos(2*np.pi/orbitalperiod*t), 
                                orbitalradius*np.sin(2*np.pi/orbitalperiod*t), t*0)
    b_xbcrs, b_ybcrs, b_zbcrs = ecltoicrs.transformCartesianCoordinates(b_xecl, b_yecl, b_zecl)
    return np.vstack([b_xbcrs, b_ybcrs, b_zbcrs])
    
def calc_epochpos_topocentric(alpha, delta, parallax, mura, mudec, vrad, t, refepoch, ephem, eqMat=False):
    """
    For each observation epoch calculate the topocentric positions delta_alpha*cos(delta) and 
    delta_delta given the astrometric parameters of a source, the observation times, and the 
    ephemeris (in the BCRS) for the observer.
    
    Parameters
    ----------
    
    alpha : float
        Right ascension at reference epoch (radians)
    delta : float
        Declination at reference epoch (radians)
    parallax : float
        Parallax (mas), negative values allowed
    mura : float
        Proper motion in right ascension, including cos(delta) factor (mas/yr)
    mudec : float
        Proper motion in declination (mas/yr)
    vrad : float
        Radial velocity (km/s)
    t : float array
        Observation times (Julian year)
    refepoch : float
        Reference epoch (Julian year)
    ephem : function
        Funtion providing the observer's ephemeris in BCRS at times t (units of AU)
        
    Keywords
    --------
    
    eqMat : boolean
        If True return the equations matrix for solving for proper motion and parallax from a
        set of observations (simplified: ignores radial motion effect and assumes perfectly known
        observation times and a perfectly known position at the reference epoch).
        
    Returns
    -------
    
    Arrays delta_alpha* and delta_delta (local plane coordinates with respect to (alpha, delta)).
    """
    mastorad = np.pi/(180*3600*1000)
    
    if parallax<0:
        signparallax=1
    else:
        signparallax=-1

    # Phase space coordinates at reference epoch. Ignore light travel time from source to observer and take the
    # absolute value of the parallax in order to handle negative parallaxes according to the interpretation that
    # the observer's orbit is then going the other way around the sun.
    bS_x, bS_y, bS_z, vS_x, vS_y, vS_z = astrometryToPhaseSpace(alpha, delta, np.abs(parallax), mura, mudec, vrad)
    
    # Normal triad, defined at the reference epoch.
    p = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    q = np.array([-np.sin(delta)*np.cos(alpha), -np.sin(delta)*np.sin(alpha), np.cos(delta)])
    r = np.array([np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha), np.sin(delta)])

    # Calculate observer's ephemeris.
    bO_bcrs = ephem(t)

    # Include the Roemer delay, take units into account.
    tB = t + np.dot(r, bO_bcrs)*pygcst.auInMeter/constants.c.value/pygcst.julianYearSeconds
    
    # Phase space coordinates for the observation times.
    bS_bcrs = np.zeros((3,t.size))
    bS_bcrs[0,:] = bS_x+vS_x*(tB-refepoch)*(1000*pygcst.julianYearSeconds/pygcst.parsec)
    bS_bcrs[1,:] = bS_y+vS_y*(tB-refepoch)*(1000*pygcst.julianYearSeconds/pygcst.parsec)
    bS_bcrs[2,:] = bS_z+vS_z*(tB-refepoch)*(1000*pygcst.julianYearSeconds/pygcst.parsec)
    
    uO = bS_bcrs + signparallax * bO_bcrs*pygcst.auInMeter/pygcst.parsec
    alpha_obs = np.arctan2(uO[1,:], uO[0,:])
    indices = (alpha_obs<0)
    alpha_obs[indices] = alpha_obs[indices]+2*np.pi
    delta_obs = np.arctan2(uO[2,:], np.sqrt(uO[0,:]**2+uO[1,:]**2))
                 
    # Calculate the difference between the observed direction to the source at time t and the coordinate direction
    # with respect to the solar system barycentre at the reference epoch (alpha, delta).
    delta_alpha = (alpha_obs - alpha)*np.cos(delta_obs)/mastorad
    delta_delta = (delta_obs - delta)/mastorad
    
    if eqMat:
        nobs = t.size
        A = np.zeros((2*nobs,3))
        A[0:nobs,0] = -np.dot(p,bO_bcrs)
        A[nobs:2*nobs,0] = -np.dot(q,bO_bcrs)
        A[0:nobs,1] = (tB-refepoch)
        A[nobs:2*nobs,2] = (tB-refepoch)
        return delta_alpha, delta_delta, A
    else:
        return delta_alpha, delta_delta
    
# Time range and observation time sampling in a separate block to keep observation times constant in the
# subsequent code block.
refepoch = 2015.5
startepoch = 2014.5
endepoch = 2019.5
nobs = 10
    
time = np.linspace(startepoch, endepoch, 1000)
tobs_sample = uniform.rvs(loc=startepoch, scale=endepoch-startepoch, size=nobs)

# source astrometric parameters at reference epoch

# Barnard's star
# For this star, which is close and has a large radial motion, the simplified astrometric solution will fail
# because perspective acceleration effects are not accounted for. This is seen in particular from the distribution
# of the reduced chi^2 values (all too high).
#
#alpha = 4.7028598776 #(66.0*units.degree).to(units.rad).value
#delta = 0.0814769927 #(16*units.degree).to(units.rad).value
#parallax = 548.31 # mas
#mura = -798.58          # mas/yr
#mudec = 10328.12        # mas/yr
#vrad = -110.51           # km/s

# Star at small parallax
alphadeg = 200.0
deltadeg= 80.0
alpha = (alphadeg*units.degree).to(units.rad).value
delta = (deltadeg*units.degree).to(units.rad).value
parallax = 0.4   # mas
mura =  1        # mas/yr
mudec = -2       # mas/yr
vrad = 40           # km/s

delta_alpha, delta_delta = calc_epochpos_topocentric(alpha, delta, parallax, mura, mudec,
                                                  vrad, time, refepoch, ephemeris_earth_simple)
delta_alpha_sample, delta_delta_sample, A = calc_epochpos_topocentric(alpha, delta, parallax, mura, mudec,
                                                                   vrad, tobs_sample, refepoch, 
                                                                   ephemeris_earth_simple, eqMat=True)

useagab(usetex=False, sroncolours=False, fontfam='sans')
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(121)
apply_tufte(ax)

ax.plot(delta_alpha, delta_delta, 
        label=r'Source path', lw=2)
ax.plot(delta_alpha_sample, delta_delta_sample, 'o', 
        label=r'Observation times'.
        format(parallax, mura, mudec), lw=2)
ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
ax.set_ylabel(r'$\Delta\delta$ [mas]')
ax.axhline(y=0, c='gray', lw=1)
ax.axvline(x=0, c='gray', lw=1)
ax.legend(loc='upper right', fontsize=14,  facecolor='#000000', framealpha=0.1,
         labelspacing=1)
ax.set_title(r'$\alpha={0:.0f}^\circ$, $\delta={1:.0f}^\circ$, $\varpi={2:.2f}$, $\mu_{{\alpha*}}={3:.2f}$, $\mu_\delta={4:.2f}$'.
        format(alphadeg, deltadeg, parallax, mura, mudec), fontsize=16)

ax1dra = fig.add_subplot(222)
apply_tufte(ax1dra)
ax1dra.spines['bottom'].set_visible(False)
ax1dra.xaxis.set_ticks([])
ax1dra.plot(time, delta_alpha)
ax1dra.plot(tobs_sample, delta_alpha_sample, 'o')
ax1dra.set_ylabel(r'$\Delta\alpha*$ [mas]')

ax1ddec = fig.add_subplot(224)
apply_tufte(ax1ddec)
ax1ddec.plot(time, delta_delta)
ax1ddec.plot(tobs_sample, delta_delta_sample, 'o')
ax1ddec.set_xlabel(r'Time [yr]')
ax1ddec.set_ylabel(r'$\Delta\delta$ [mas]')

plt.tight_layout()
plt.savefig('source_motion.pdf')
plt.show()
