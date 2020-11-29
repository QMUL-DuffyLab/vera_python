# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:32:49 2020

Short conversion functions

@author: Chris
"""

import numpy as np

#scipy constants
from scipy.constants import hbar as hbar
from scipy.constants import c as c_vac #speed of lightin vaccum
from scipy.constants import pi as pi 
from scipy.constants import e as elec_q #electronic charge

"""
(1) Convert a time (in ps) into a lengthscale (in 1/cm)
    
"""

def ps_to_cm(time_ps):
    if time_ps==0.0:
        return(np.inf)
    else:
        return(1.0/(200.0*pi*c_vac*(time_ps*1E-12)))

"""
(2) Convert a lengthscale (in 1/cm) into a rate (in 1/ps)
    
"""

def rate_cm_to_rate_ps(rate_cm):
    return((200.0E-12*pi*c_vac*rate_cm))
    

"""
(3) Convert an energy (in eV) into a lengthscale (in 1/cm)
"""    

def eV_to_cm(energy_eV):
    E=energy_eV*elec_q
    return(E/(100.0*hbar*2.0*pi*c_vac))

"""
(4) Convert wavelength (in nm) into a lengthscale (in 1/cm)
"""

def nm_to_cm(wavelength_nm):
    return(1.0/(wavelength_nm*1.0E-7))