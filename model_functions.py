'''
model_functions.py

Creator: Paul Zivick
Last edited: 6/10/2022

Notes/Description: This is a collection of functions
that are needed in order to run the Make_predictions
Jupyter notebook. These functions are adopted from earlier
versions written for the Zivick+2021 analysis, in particular
updating to use column names in manipulations rather than
explicitly stated indice variables.
'''

from numpy import deg2rad as d2r
import numpy as np
from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord

################################################################
####

def predict_pm(inputs, model, usevdM02=False):

    data = Table(data=inputs, names=inputs.colnames) #Create a copy of the Table

    center = [d2r(model['RA_0']), d2r(model['Dec_0'])]

    rho, phi = wcs2ang(center[0], center[1], \
     d2r(data['RA']), d2r(data['Dec']))

    data.add_column(rho, name='rho')
    data.add_column(phi, name='phi')

    dist0, vtran, thtran = prep_model_params(model)

    galxyz = ang2xyz(rho, phi, data['Dist'], \
    dist0,theta=d2r(model['theta']),incl=d2r(model['incl']))

    data.add_column(galxyz[:,0], name='x0')
    data.add_column(galxyz[:,1], name='y0')
    data.add_column(galxyz[:,2], name='z0')

##

    velcm = np.zeros((len(data),3))

    velcm[:,0], velcm[:,1], velcm[:,2] = make_cm_angvec(vtran, thtran, model['vsys_0'], data['rho'], data['phi'])

    components = Table([velcm[:,0], velcm[:,1], velcm[:,2]], names=['cm1', 'cm2', 'cm3'])

##

    velpn = np.zeros((len(data),3))

    velpn[:,0], velpn[:,1], velpn[:,2] = pn_comp(d2r(model['theta']), d2r(model['incl']), \
    dist0, data['phi'], data['rho'], model['didt'], \
    model['dtdt'])

    components.add_column(velpn[:,0], name='pn1')
    components.add_column(velpn[:,1], name='pn2')
    components.add_column(velpn[:,2], name='pn3')

##

    velint = np.zeros((len(data),3))

    if (usevdM02):
        velint[:,0], velint[:,1], velint[:,2] = make_int_angvec_plane(model['rad0'], \
        model['Vrot'],model['rotdir'],d2r(model['theta']), \
        d2r(model['incl']),dist0, data['phi'], data['rho'], \
        data['Dist'], usevdM02=usevdM02)

    else:
        velint[:,0], velint[:,1], velint[:,2] = make_int_angvec_plane(model['rad0'], \
        model['Vrot'], model['rotdir'],d2r(model['theta']), \
        d2r(model['incl']),dist0, data['phi'], data['rho'], \
        data['Dist'])

    components.add_column(velint[:,0], name='rot1')
    components.add_column(velint[:,1], name='rot2')
    components.add_column(velint[:,2], name='rot3')

##
    veltidal = np.zeros((len(data),3))

    newtidal = tidal_linear(center, dist0, d2r(data['RA']), \
      d2r(data['Dec']), data['Dist'], model['relvel0'][0], \
      model['relvel0'][1], model['relvel0'][2], \
      model['vsys_0'], model['tidalScale'])

    veltidal[:,0], veltidal[:,1], veltidal[:,2] = vel_xyz2sph(newtidal, 0.0, 0.0, data['phi'], data['rho'])

    components.add_column(veltidal[:,0], name='tidal1')
    components.add_column(veltidal[:,1], name='tidal2')
    components.add_column(veltidal[:,2], name='tidal3')

##

    data.add_column(components['cm1'] + components['pn1'] + components['rot1'] + components['tidal1'], name='v1')
    data.add_column(components['cm2'] + components['pn2'] + components['rot2'] + components['tidal2'], name='v2')
    data.add_column(components['cm3'] + components['pn3'] + components['rot3'] + components['tidal3'], name='v3')

    cosg, sing = calc_gamma(center[0], center[1], \
     d2r(data['RA']), d2r(data['Dec']), data['rho'])
    data.add_column(cosg, name='cosG')
    data.add_column(sing, name='sinG')

    pmra, pmdec = ang2wcs_vec(dist0, data['v2'], \
     data['v3'], data['cosG'], data['sinG'], \
     data['rho'], data['phi'])

    pmra = -1.0 * pmra

    data.add_column(pmra, name='pmRA')
    data.add_column(pmdec, name='pmDec')

    return data, components

################################################################

################################################################
#### Function to convert from WCS to angular coordinates

def wcs2ang(ra0, dec0, ra, dec):

    rho = np.arccos( np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0) + np.sin(dec) * np.sin(dec0) )

    phi = np.arccos( (-1.0*np.cos(dec) * np.sin(ra - ra0)) / (np.sin(rho)) )
  #phi = np.arcsin( ((np.sin(dec)*np.cos(dec0)) - (np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0))) / (np.sin(rho)) )

# Calculate Cartesian coordinates to test for correcting the angle

    testx, testy = wcs2gaiaxy(ra, dec, np.asarray([ra0, dec0]))

    for i in range(len(testy)):
        if (testy[i] < 0.0):
            phi[i] = -1.0*phi[i]

    return rho, phi

################################################################

################################################################
#### Function to convert angular coordinates to cartesian in
#### the frame of the galaxy rotation

def ang2xyz(rho, phi, dist, dist0, theta=0.00001, incl=0.00001):

    new = np.zeros((len(phi),3))

    new[:,0] = dist * np.sin(rho) * np.cos(phi - theta)

    new[:,1] = dist * (np.sin(rho) * np.cos(incl) * np.sin(phi-theta) + np.cos(rho) * np.sin(incl)) - (dist0 * np.sin(incl))

    new[:,2] = dist * (np.sin(rho) * np.sin(incl) * np.sin(phi-theta) - np.cos(rho) * np.cos(incl)) + (dist0 * np.cos(incl))

    return new

################################################################

################################################################
####
def prep_model_params(modelparams):

    # Calculate the distance from the distance modulus
    dist = (10**((modelparams['m_M']/5.0)+1)) / 1000.0

    # Calculate velocity parameters based on input PMs
    # total CM proper motion
    mutran = (modelparams['pmRA_0']**2 + modelparams['pmDec_0']**2)**(0.5)

    # angle of motion for the CM PM
    thtran = d2r(np.rad2deg(np.arctan2(modelparams['pmRA_0'],modelparams['pmDec_0']) + 90.0))

    # total CM velocity
    vtran = 4.7403885 * mutran * dist

    return dist, vtran, thtran

################################################################

################################################################
#### Function to calculate the tidal expansion assuming
#### increased velocity only in direction of relative velocity

def tidal_linear(center, dist0, allra, alldec, alldist, relpmra, relpmdec, relrv, vsys, tidalscale):

  #SMC Center in astropy structure for transformation
  center0 = coord.ICRS(ra=center[0]*u.rad, dec=center[1]*u.rad, \
                       distance=dist0*u.kpc, pm_ra_cosdec=relpmra*u.mas/u.yr,\
                       pm_dec=relpmdec*u.mas/u.yr, \
                       radial_velocity=relrv*u.km/u.s)

  #SMC Center in Galactocentric coordinates
  center1 = center0.transform_to(coord.Galactocentric)

  #Relative velocity in Galactocentric coordinates
  relvg = np.asarray([center1.v_x.value,center1.v_y.value, center1.v_z.value])

  #All stellar positions in astropy structure
  all0 = coord.ICRS(ra=allra*u.rad, dec=alldec*u.rad, \
                     distance=alldist*u.kpc)

  #Stellar positions into Galactocentric coordinates
  all1 = all0.transform_to(coord.Galactocentric)

  #Calculate the distance between a star and SMC center along axis
  #of relative velocity
  newdist=((all1.x.value-center1.x.value)*(relvg[0]/np.linalg.norm(relvg)))+\
          ((all1.y.value-center1.y.value)*(relvg[1]/np.linalg.norm(relvg)))+\
          ((all1.z.value-center1.z.value)*(relvg[2]/np.linalg.norm(relvg)))


  # Transform the WCS relative motion vector into vdM02 Cartesian coordinates
  vtran_rel = 4.7403885 * dist0 * (relpmra**2 + relpmdec**2)**(0.5)
  thtran_rel = d2r(np.rad2deg(np.arctan2(relpmra, relpmdec))+90.0)

  relvc = np.asarray([vtran_rel*np.cos(thtran_rel), vtran_rel*np.sin(thtran_rel), -1.0*relrv])


  tidalcomp = np.zeros((len(allra), 3))
  tidalcomp[:,0] = newdist * tidalscale * (relvc[0] / np.linalg.norm(relvc))
  tidalcomp[:,1] = newdist * tidalscale * (relvc[1] / np.linalg.norm(relvc))
  tidalcomp[:,2] = newdist * tidalscale * (relvc[2] / np.linalg.norm(relvc))

  return tidalcomp

################################################################

################################################################
#### Function to create vectors (v1, v2, v3) for the center of
#### mass motion

def make_cm_angvec(vtran, thtran, vsys, rho, phi):

  v1 = vtran*np.sin(rho)*np.cos(phi - thtran) + vsys*np.cos(rho)
  v2 = vtran*np.cos(rho)*np.cos(phi - thtran) - vsys*np.sin(rho)
  v3 = -1.0 * vtran * np.sin(phi - thtran)

  return v1, v2, v3

################################################################

################################################################
#### Function to calculate the contribution from prcession
#### and nutation

def pn_comp(thet, incl, dist0, phi, rho, didt, dtdt):

  const = (dist0 * np.sin(rho)) / ((np.cos(incl) * np.cos(rho)) - \
                                   (np.sin(incl) * np.sin(rho) * \
                                    (np.sin(rho - thet))))

  v1 = const * ( (didt * np.sin(phi - thet)) * \
                ((np.cos(incl) * np.cos(rho)) - np.sin(incl) * \
                 np.sin(rho) * np.sin(phi - thet)))

  v2 = const * ( (didt * np.sin(phi - thet)) * \
                ((-1.0 * np.cos(incl) * np.sin(rho)) - np.sin(incl) * \
                 np.cos(rho) * np.sin(phi - thet)))

  v3 = const * ( (didt * np.sin(phi - thet)) * \
                 (-1.0 * np.sin(incl) * np.cos(phi - thet)) + \
                 (dtdt * np.cos(incl)))

  return v1, v2, v3

################################################################

################################################################
#### Function to create vectors (v1, v2, v3) for the internal
#### motions of the galaxy by first creating (vx', vy', vz')
#### given a rotating velocity field

def make_int_angvec_plane(rad0, vel0, sign, thet, incl, dist0, phi, rho, dist, checkcurve=False, checkfile="", usevdM02=False, n0=0.0):

# Calculates the (vx', vy', vz') vector in the frame of the galaxy

  newcoord = ang2xyz(phi, rho, dist, dist0, theta=thet, incl=incl)
  x1, y1, z1 = 0, 1, 2

  vframe = np.zeros((len(rho),3))
  vx, vy, vz = 0, 1, 2

  if (checkcurve):
    if (usevdM02):
      rotval = rotcurve_vdM02(newcoord[:,x1], newcoord[:,y1], rad0, vel0, checkcurve=checkcurve, checkfile=checkfile, n0=n0)
    else:
      rotval = rotcurve(newcoord[:,x1], newcoord[:,y1], rad0, vel0, checkcurve=checkcurve, checkfile=checkfile)
  else:
    if (usevdM02):
      rotval = rotcurve_vdM02(newcoord[:,x1], newcoord[:,y1], rad0, vel0, n0=n0)
    else:
      rotval = rotcurve(newcoord[:,x1], newcoord[:,y1], rad0, vel0)

  rotval.shape = (len(rho),)



  vframe[:,vx] = sign * rotval * (newcoord[:,y1]/(newcoord[:,x1]**2 + newcoord[:,y1]**2)**(0.5))
  vframe[:,vy] = -1.0 * sign * rotval * (newcoord[:,x1]/(newcoord[:,x1]**2 + newcoord[:,y1]**2)**(0.5))

# Calculate the components of the transformation matrix from v' to v

  v1, v2, v3 = vel_xyz2sph(vframe, thet, incl, phi, rho)

#

  return v1, v2, v3

################################################################

################################################################
#### Convert to Gaia cartesian coordinates

def wcs2gaiaxy(ra, dec, center):
  x = np.cos(dec) * np.sin(ra-center[0])
  y = np.sin(dec) * np.cos(center[1]) - np.cos(dec) * np.sin(center[1]) * np.cos(ra - center[0])

  x = np.rad2deg(x)
  y = np.rad2deg(y)

  return x,y

################################################################

################################################################
#### Function to create a rotation curve in the frame of the
#### galaxy

def rotcurve(x, y, r0, v0, checkcurve=False, checkfile=""):
  rad = (x**2 + y**2)**(0.5)
  vrot = np.zeros((len(rad),1))

  for ii in range(len(rad)):
    if (rad[ii] < r0):
      vrot[ii] = (rad[ii]/r0) * v0
    else:
      vrot[ii] = v0

  if (checkcurve):
    plt.clf()
    plt.scatter(rad, vrot, s=4, marker="+", color="gray", alpha=0.8)
    plt.xlabel(r'Radius (kpc)')
    plt.ylabel(r'Velocity (km/s)')
    plt.ylim(0.0, 60.0)
    plt.tight_layout()
    plt.savefig(checkfile)


  return vrot

################################################################

################################################################
#### Function to create a rotation curve in the frame of the
#### galaxy

def rotcurve_vdM02(x, y, r0, v0, checkcurve=False, checkfile="", n0=0.0):
  rad = (x**2 + y**2)**(0.5)
  vrot = np.zeros((len(rad),1))

  for ii in range(len(rad)):
    vrot[ii] = v0 * ((rad[ii]**n0) / ((rad[ii]**n0) + (r0**n0)))

  if (checkcurve):
    plt.clf()
    plt.scatter(rad, vrot, s=4, marker="+", color="gray", alpha=0.8)
    plt.xlabel(r'Radius (kpc)')
    plt.ylabel(r'Velocity (km/s)')
    plt.ylim(0.0, 60.0)
    plt.tight_layout()
    plt.savefig(checkfile)


  return vrot

################################################################

################################################################
#### Function to convert velocities in the vdM02 vx, vy, vz
## space into the more spherical v1, v2, v3 space

def vel_xyz2sph(vel, thet, incl, phi, rho):

# Calculate the components of the transformation matrix from v' to v

  la = np.sin(rho) * np.cos(phi-thet)
  lb = np.cos(rho) * np.cos(phi-thet)
  lc = -1.0 * np.sin(phi-thet)
  ld = np.sin(rho) * np.cos(incl) * np.sin(phi-thet) + np.cos(rho) * np.sin(incl)
  le = np.cos(rho) * np.cos(incl) * np.sin(phi-thet) - np.sin(rho) * np.sin(incl)
  lf = np.cos(incl) * np.cos(phi-thet)
  lg = np.sin(rho) * np.sin(incl) * np.sin(phi-thet) - np.cos(rho) * np.cos(incl)
  lh = np.cos(rho) * np.sin(incl) * np.sin(phi-thet) + np.sin(rho) * np.cos(incl)
  li = np.sin(incl) * np.cos(phi-thet)

  vx, vy, vz = 0, 1, 2

  v1 = la * vel[:,vx] + ld * vel[:,vy] + lg * vel[:,vz]
  v2 = lb * vel[:,vx] + le * vel[:,vy] + lh * vel[:,vz]
  v3 = lc * vel[:,vx] + lf * vel[:,vy] + li * vel[:,vz]

#

  return v1, v2, v3


################################################################

##########################################################################
#### Function to calculate the Gamma factor for vector transformations

def calc_gamma(ra0, dec0, ra, dec, rho):

    cosG = (np.sin(dec) * np.cos(dec0) * np.cos(ra-ra0) - np.cos(dec)*np.sin(dec0)) / np.sin(rho)

    sinG = (np.cos(dec0) * np.sin(ra - ra0)) / np.sin(rho)

    return cosG, sinG

##########################################################################

##########################################################################
#### Function to convert an angular vector to wcs

def ang2wcs_vec(dist0, v2, v3, cosG, sinG, rho, phi, theta=np.deg2rad(0.00001), incl=np.deg2rad(0.00001)):

# Calculates the scaling quantity for the proper motion

    propercon = (np.cos(incl) * np.cos(rho) - np.sin(incl)*np.sin(rho)*np.sin(phi-theta)) \
    / (dist0 * np.cos(incl))

# Use the scaling quantity and the vector components in the skyplane
# to calculate the observable proper motions

    muwe = (propercon * (-1.0*sinG*(v2) - cosG*(v3))) / (4.7403895)
    muno = (propercon * (cosG*(v2) - sinG*(v3))) / (4.7403895)

    return muwe, muno

##########################################################################
