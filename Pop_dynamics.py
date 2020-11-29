# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:19:27 2020

A Collection of sub-routines needed to simulate energy relaxation within a 
manifold of vibronic states. Two phenomena are explicitly characterized

(1) Inramolecular vibrational redistribution on a single electronic state.
(2) Interconversion between manifolds of vibronic states on different 
electronic states.

This contains several functions necessary for computing populaiton relzation 
within a second order purturbative treatment of the system-bath interaction

@author: Chris
"""

import numpy as np
import cm_conversion as cm_con  #does tricky conversions involving 1/cm

#scipy constants
from scipy.constants import hbar as hbar
from scipy.constants import c as c_vac #speed of lightin vaccum
from scipy.constants import pi as pi 

from scipy.integrate import odeint

import matplotlib.pyplot as plt


"""
(1) FC_overlap: This computes a full set of FC overlaps for a pair of electronc 
state. We assume that each state has the same set of optically-coupled modes 
and that the PES of equivalent modes are identical but for a displacement along
the normal coordinate. 

Input: (i) disp=dimensionless displacement for equivalent modes on two electronic states.
       (ii) a_max=highest vibrational quantum number on lower electronic state.
       (iii) b_max=highest vibrational quantum number on upper electronic state.

Output: a_max*ap_max array of FC_overlap. These are dimensionless
"""

def FC_overlap(disp,a_max,b_max):
    
    #Efficient recursive method for computing FC_overlap as presented by May and Khun (2.8.2)
 
    #Declare array of FC coefiicients. The array is square with a dimension determined
    #by the size of the largest of the two vibrational manifolds
    nu_max=max(a_max,b_max)
    
    FC=np.zeros((nu_max+1,nu_max+1)) #accounts for a=0, 1, 2, ...
    
    #Vibrationless overlap
    FC[0][0]=np.exp(-0.5*disp*disp)
    
    #outer edges
    for a in range(1,nu_max+1):
        FC[0][a]=-disp/np.sqrt(float(a))*FC[0][a-1]
        FC[a][0]=((-1.0)**a)*FC[0][a]
        
    #remaining elements
    for a in range(1, nu_max+1):
        for b in range(a, nu_max+1):
            FC[a][b]=-disp/np.sqrt(float(b))*FC[a][b-1]+np.sqrt((float(a))/(float(b)))*FC[a-1][b-1]
            FC[b][a]=((-1.0)**(b-a))*FC[a][b]
 
            
    return(FC)

"""
(2) Spectral density of an ODO: 
    
Input: (i) w=frequency (in 1/cm)
       (ii) L=reorganization energy (in 1/cm)
       (iii) G=damping term (in 1/cm)
       
Output: C''(w)
"""
    
def C_ODO(w,L,G):
    if G!=np.inf:
        #Since G is often quoted as a time but used in calculations as 1/cm
        #stray zero times (from a partially-filled array of values) can be transformed
        #into infinite energies. This just excludes them
        
        C=(2.0*L*G*w)/((w*w)+(G*G))
    else:
        C=0.0        
    return(C)

"""
(3) VR rates k_alpha: Computed using Redfield theory in the secular and Markov 
approximations. Rate constants which are given by the Fourier Transform of 
the bath correlation function which is itself related to the spectral density.

Input: (i) w_alpha=natural frequency of mode alpha (in 1/cm)
       (ii) beta=inverse thermodynamic temperature (in cm)
       (iii) L, G=paramters for the spectral density (in 1/cm)
       
Output: The upward (k_up) and downward (k_down) rate constants for transfer to
adjacent vibrational levels of the same mode. This will be in units of 1/cm 
"""
    
def k_alpha(w_alpha,beta,L,G):
    #note this function
    k_up=C_ODO(-w_alpha,L,G)*((1.0/np.tanh(-0.5*beta*w_alpha))+1)
    k_down=C_ODO(w_alpha,L,G)*((1.0/np.tanh(0.5*beta*w_alpha))+1)
    
    return(k_up, k_down)
    
"""
(4) IC rates between a vibronic state belonging to one manifold
(|ia>=|i>|a1(i)>|a2(i)>...|aN(i)>) and a vibronic state on a second manifold
(|ja'>=|j>|a'1(j)>|a'2(j)>...|a'N(j)>). 

Input: (i) e_ij = 2-Tuple of indices i and j elecronic states
       (ii) a = N-Tuple of vibrational quantum numbers for one electroic state.
       (iii) b= N-Tuple of vibrational quantum numbers for the other electronic state.
       (iv) w_e= 2-Tuple of the electronic transition frequencies wi and wj (in 1/cm)
       (v) w_alpha= N-Tuple of natural frequencies of the modes (in 1/cm)
       (vi) beta, L,G=parameters for the spectral density
       
Output: A single rate constant (in 1/cm)
"""

def k_inter(e_ij,a,b,w_e,w_alpha,beta,L,G):
    #We are calculating the rate constant k^{e_ij[0],e_ij[1]}_{a,b}
    #(1) Calculate the vibrational energy gap
    w_ab=[]
    
    for i, item in enumerate(a):
        w_ab.append(w_alpha[i]*(a[i]-b[i]))

    w_ab=sum(w_ab)
    
    
    #(2) Correlation function
    if e_ij[0]>e_ij[1]:
        #This is an 'uphill rate'

        #(3) Calculate the electronic energy gap
        w_ij=w_e[0]-w_e[1]
    
        #(4) Total energy gap
        Delta_ij_ab=w_ij+w_ab
        
        k=C_ODO(-Delta_ij_ab,L,G)
        k=k*((1.0/np.tanh(-0.5*beta*Delta_ij_ab))+1)
    else:
        #(3) Calculate the electronic energy gap
        w_ij=w_e[1]-w_e[0]
    
        #(4) Total energy gap
        Delta_ij_ab=w_ij+w_ab

        k=C_ODO(Delta_ij_ab,L,G)
        k=k*((1.0/np.tanh(0.5*beta*Delta_ij_ab))+1)
            
    return k

"""
(5) Normalized Gaussian function 

Input: (i) x=variable
       (ii) mu=centre point
       (iii) width=FWHM of the Gaussian (=2 sqrt(2 ln(2))sig )

Output: value in either inverse time or inverse frequency
       
"""
    
def gauss_profile(x,mu,width):
    A=10.0#=(2.0*np.sqrt(2.0*np.log(2)))/(width*np.sqrt(2.0*np.pi))
    gauss=-np.square(x-mu)/(2.0*np.square(width/(2.0*np.sqrt(2.0*np.log(2.0)))))
    return(A*np.exp(gauss))

"""
(6) Equilibrium populations of a thermal oscillator: This is used to initialize
    the populations of the vibrational states on the electronic ground state
    

Input: (i) a=N-tuple of vibrational quantum numbers for each of the N normal modes 
       (ii) w_alpha=N-tuple of natural frequencies (in cm^-1) of the N normal modes
       (iii) beta=inverse thermodynamic temperature (in cm)

Output: population (ie occupation probability) of the specific vibrational state 
        defined by the tuple of quanutum numbers, a. 
"""

def thermal_osc(a,w_alpha,beta):
    epsilon_a=0.0 #vibrational energy
    for i, nu in enumerate(a):
        epsilon_a=epsilon_a+(w_alpha[i]*(nu+0.5))
        
    na=np.exp(-beta*epsilon_a)
    
    #compute partition function
    Z=1.0
    for i, w in enumerate(w_alpha):
        Z=Z*(np.exp(-0.5*beta*w)/(1-np.exp(-beta*w)))

    na=na/Z
        
    return(na)
    
"""
(7) The coupled linear differential equations that describe the population 
    dynamics across the vibronic manifold. This is where everything is assembled
    and then numerically integrated
    

Input: (i) n_ia = a flattend array of current/initial population values
       (ii) t = time (in ps) used to define the pumping term 
       (iii) n_ia_shape is needed to reshape the flattend data input. this is 
             to allow compatability with scipy.integrate.odeint
       (iv) w_i=tuple of pure electronic energies
       (v)  w_mode=tuple of mode frequencies
       (vi) k_alpha= an array of IVR rates
       (vii) FC = an array of FC overlap factors
       (viii) k_IC =an array of IC ragtes
       (ix) pump_param = Tuple of parameters for the pump. In order this contains
            (a) elec_excite=number indicating the electronic state(s) selected by the laser
            (b) A0x = The effective intensity of the laser
            (c) w_pump_centre=frequency of the laser
            (d) w_pump_width=frequency width
            (e) t_pump0=arrival time of the pulse
            (f) t_pump_duration=pump duration

Output: (i) dndt = a flattend array of time derivatives of the vibronic populations. 
 
"""

def dndt(n_ia,t,n_ia_shape,w_i,w_mode,k_alpha,FC,k_IC,pump_param):

    #reshape n_ia since it had to be flattened to be compatible with 
    #scipy.integrate.odeint input requirements
    n_ia=n_ia.reshape((n_ia_shape))

    #use shape of n_ia to determine shape of dndt
    N_elec=n_ia_shape[0] #number of electronic states
    N_mode=len(n_ia_shape)-1 #the first index is the electronic state, the remainder 
                             #count vibrational levels in each mode
    N_vib_level=n_ia_shape[1]-1 #Number of EXCITED vibrational levels considered for each mode
                                #assumes all modes have same number of vibrational levels                       
    
    dndt_IVR=np.zeros(n_ia_shape) #3 components of the dynamics (IVR, IC, pumping)
    dndt_IC=np.zeros(n_ia_shape) 
    dndt_pump=np.zeros(n_ia_shape)
        
    '''************************************************************************
    Intramolecular Vibrational Redistribution
    ************************************************************************'''
    mode_iter=np.nditer(dndt_IVR,flags=['multi_index'])
    for element in mode_iter: #iterating through all dndt_IVR (the LHS of the equation)
        ia=mode_iter.multi_index #the full list of indices for the vibronic level

        #The left hand side of the equation
        #interate through the modes
        for alpha in range(N_mode):

            #terms involving the vibronic level below
            if ia[alpha+1]>0: #prevet counting below the zero point
                dndt_IVR[ia]-=ia[alpha+1]*k_alpha[ia[0]][alpha][0]*n_ia[ia] #loss of population to the lower state
                
                ia_lower=list(ia)#index for the lower vibronic level. Has to be mutable
                ia_lower[alpha+1]-=1 #lower the index of the current mode by 1
                ia_lower=tuple(ia_lower) #convert to a tuple so it can be used as an index
                dndt_IVR[ia]+=ia[alpha+1]*k_alpha[ia[0]][alpha][1]*n_ia[ia_lower] #gain of population from the level below

            #terms involving the vibronic level above
            if ia[alpha+1]<N_vib_level: #prevent counting levels above the maximum level
                dndt_IVR[ia]-=(ia[alpha+1]+1.0)*k_alpha[ia[0]][alpha][1]*n_ia[ia] #loss of population to the upper state
                
                ia_upper=list(ia)
                ia_upper[alpha+1]+=1 #raise the index of the current mode by 1
                ia_upper=tuple(ia_upper)
                dndt_IVR[ia]+=(ia[alpha+1]+1.0)*k_alpha[ia[0]][alpha][0]*n_ia[ia_upper] #gain in population due to transfer from level above

    '''************************************************************************
    Interconversion
    ************************************************************************'''
    
    for i in range(N_elec): #need to separate electronic and vibration indices
        mode_iter=np.nditer(dndt_IC[0],flags=['multi_index'])
        for element in mode_iter: #iterating through all dndt_IVR (the LHS of the equation)
            a=mode_iter.multi_index #the vibrational indices of the 'current' vibronic state
                
            #To construct the right hand side of the equation we need to iterate 
            #over the vibrational indices of all 'other' vibronic state
            mode_iter2=np.nditer(dndt_IC[0],flags=['multi_index'])
            for element2 in mode_iter2:
                b=mode_iter2.multi_index

                #Construct the index tuples we need to define rates and populations
                #(1) the population indices
                n_index=[i] #index for the current state
                n_plus_index=[i+1] #index for the upper electronic state
                n_minus_index=[i-1] #index for the lower electronic state
                
                for a_nu in a: n_index.append(a_nu) #add vibrational indices
                for b_nu in b: 
                    n_plus_index.append(b_nu)
                    n_minus_index.append(b_nu)
                
                n_index=tuple(n_index)
                n_plus_index=tuple(n_plus_index)
                n_minus_index=tuple(n_minus_index)
               
                #(2) the indices for the interconversion rates
                k_ba_ji_index=[i+1,i]
                k_ba_ij_index=[i,i+1]
                k_ab_ji_index=[i-1,i]
                k_ab_ij_index=[i,i-1]
                
                for a_nu in a:
                    k_ab_ji_index.append(a_nu)
                    k_ab_ij_index.append(a_nu)
                for b_nu in b:
                    k_ba_ji_index.append(b_nu)
                    k_ba_ij_index.append(b_nu)
                for a_nu in a:
                    k_ba_ji_index.append(a_nu)
                    k_ba_ij_index.append(a_nu)
                for b_nu in b:
                    k_ab_ji_index.append(b_nu)
                    k_ab_ij_index.append(b_nu)
                
                k_ba_ji_index=tuple(k_ba_ji_index)
                k_ba_ij_index=tuple(k_ba_ij_index)
                k_ab_ji_index=tuple(k_ab_ji_index) 
                k_ab_ij_index=tuple(k_ab_ij_index)
                
                #the electronic ground state. Only interacts with state above
                if i==0:
                    FC_i_plus=1.0                                    
                    for alpha in range(N_mode):
                        FC_i_plus*=FC[i][i+1][alpha][a[alpha]][b[alpha]]*FC[i][i+1][alpha][a[alpha]][b[alpha]]
                                 
                    dndt_IC[n_index]-=FC_i_plus*k_IC[k_ba_ji_index]*n_ia[n_index]
                    dndt_IC[n_index]+=FC_i_plus*k_IC[k_ba_ij_index]*n_ia[n_plus_index]
                #the upper most electronic state (excluding Sn)    
                elif i==N_elec-1:                               
                    FC_i_minus=1.0
                    for alpha in range(N_mode):
                        FC_i_minus*=FC[i-1][i][alpha][b[alpha]][a[alpha]]*FC[i-1][i][alpha][b[alpha]][a[alpha]]
 
                    dndt_IC[n_index]-=FC_i_minus*k_IC[k_ab_ji_index]*n_ia[n_index]
                    dndt_IC[n_index]+=FC_i_minus*k_IC[k_ab_ij_index]*n_ia[n_minus_index]
                #intermediate electronic states
                else:
                    FC_i_plus=1.0                                    
                    FC_i_minus=1.0
                    for alpha in range(N_mode):
                        FC_i_plus*=FC[i][i+1][alpha][a[alpha]][b[alpha]]*FC[i][i+1][alpha][a[alpha]][b[alpha]]
                        FC_i_minus*=FC[i-1][i][alpha][b[alpha]][a[alpha]]*FC[i-1][i][alpha][b[alpha]][a[alpha]]
                                 
                    dndt_IC[n_index]-=FC_i_plus*k_IC[k_ba_ji_index]*n_ia[n_index]
                    dndt_IC[n_index]+=FC_i_plus*k_IC[k_ba_ij_index]*n_ia[n_plus_index]
                    dndt_IC[n_index]-=FC_i_minus*k_IC[k_ab_ji_index]*n_ia[n_index]
                    dndt_IC[n_index]+=FC_i_minus*k_IC[k_ab_ij_index]*n_ia[n_minus_index]

    '''************************************************************************
    Pumping term
    
    Currently this characterizes excitation but a 'sech squared' laser pulse of 
    finite Gaussian width. It is assumed that the pusle is sufficiently narrow
    to select a particular electronic state although it may cover several 
    vibrational levels on that state. 
    
    This part could be modified to include white excitation or transfer of 
    energy from a different chromophore. The IC and IVR terms will remain 
    unchanged
    ************************************************************************'''

    #depopulate the ground state
    
    mode_iter=np.nditer(dndt_pump[0],flags=['multi_index'])
    for element in mode_iter: #Interate through the vibronic levels on the ground state
        a=mode_iter.multi_index #the vibrational indices of the 'current' vibronic state

        mode_iter2=np.nditer(dndt_pump[0],flags=['multi_index'])
        for element2 in mode_iter2:
            b=mode_iter2.multi_index

            #Calculate the full vibronic transition frequency
            Delta_x0_ba=w_i[pump_param[0]]-w_i[0] #the pure electronic gap
            for alpha in range(N_mode):
                Delta_x0_ba+=(w_mode[alpha]*(b[alpha]-a[alpha]))

            #Calculate the product of FC overlap
            FC_square=1.0
            for alpha in range(N_mode):
                FC_square*=FC[0][pump_param[0]][alpha][a[alpha]][b[alpha]]*FC[0][pump_param[0]][alpha][a[alpha]][b[alpha]]

            #Construct the frequency distribution in the pulse
            #This is a non-normalized guassian
            sig=pump_param[3]/(2.0*np.sqrt(2.0*np.log(2.0))) #standard deviation from width
            gauss=np.exp(-np.square(Delta_x0_ba-pump_param[2])/(2.0*np.square(sig)))
            gauss=gauss*(1.0/(sig*np.sqrt(2.0*np.pi)))

            #pulse duration (this is a normalized squared secant distribution)
            #Essentially it empties the ground state over the duration of the pulse
            Gamma=(1.0/(2.0*pump_param[5]))\
            *(np.square(1.0/np.cosh((t-pump_param[4])/pump_param[5])))
            
            #Depopulate the ground state
            dndt_pump[0][a]-=A0x*FC_square*gauss*Gamma*n_ia[0][a]
            
            #populate the excited state
            dndt_pump[pump_param[0]][b]+=A0x*FC_square*gauss*Gamma*n_ia[0][a]
            #(elec_excite,A0x,w_pump_centre,w_pump_width,t_pump0, t_pump_duration)

    dndt=dndt_IVR+dndt_IC+dndt_pump #combined dynamics
    dndt=dndt.flatten() #flatten the output for compatibility with odeint output requirements
    return(dndt)

"""    

        
                
            
            

            

            #Depopulation of the electronic ground state
            dndt_pump[0][a]-=sig*Gamma
                        
            #population of the resonantly selected electronic state
            #dndt_pump[pump_param[0]][a]+=FC_square*sig*Gamma

    ************************************************************************
    Output
    ************************************************************************"""
    


    


"""****************************************************************************
Input block (to be rewritten later for file input)
****************************************************************************"""

N_elec=3 #number of electronic states (not including the 'Sn' state for ESA)

beta=1/200.0 #thermodynamics inverse temperature

#electronic transtion frequncy for each state relative to w0 (+1 includes transition to Sn)
wi=np.zeros(N_elec+1)

wi[0]=0.0 #ground state
wi[1]=10000.0 #first singlet 
wi[2]=20000.0 #second singlet
wi[3]=30000.0 #nth singlet (manifests only as acceptor for ESA)

N_mode=2 #number of optically coupled normal modes
N_vib_level=3 #number of excited vibrational levels considered for each mode

#Optical mode frequencies (in 1/cm)
w_mode=np.zeros(N_mode)

w_mode[0]=1100.0
w_mode[1]=1500.0

#Reorganization energy for Interconversion from electronic state j -> i 
#This does not need to include transitions from Sn.
L_IC_ij=np.zeros((N_elec,N_elec))

L_IC_ij[1][2]=1000.0
L_IC_ij[0][1]=200.0

#Reorganization energy for Intramolecular Vibrartional Redistribution on a
#single electronic state. This does not need to include Sn
L_IVR_i=np.zeros(N_elec)

L_IVR_i[0]=50.0
L_IVR_i[1]=100.0
L_IVR_i[2]=300.0

#Damping times for IC and IVR (in fs) for IC from electronic state j -> i. 
#These are then converted to 1/cm.
#This is assumed to be identical for each normal mode
#This does not need to include Sn
G_IC_ij=np.zeros((N_elec,N_elec))
G_IC_ij[1][2]=163.6
G_IC_ij[0][1]=163.6

G_IVR_i=np.zeros((N_elec))
G_IVR_i[0]=163.6
G_IVR_i[1]=163.6
G_IVR_i[2]=163.6

#Convert to 1/cm
for i, vec in enumerate(G_IC_ij):
    for j, item in enumerate(vec):
        G_IC_ij[i][j]=cm_con.ps_to_cm(G_IC_ij[i][j]*1E-3)

#Dimensionless displacements of equivalent modes on different electronic states
#We need to include Sn 
#The indexing is disp[mode][elec_lower][elec_upper]
disp=np.zeros((N_mode,N_elec+1,N_elec+1))

disp[0][0][1]=1.0
disp[1][0][1]=1.0
disp[0][1][2]=1.0
disp[1][1][2]=1.0
disp[0][0][2]=0.7
disp[1][0][2]=0.7
disp[0][1][3]=0.4
disp[1][1][3]=0.4
    
#Pumping parameters


elec_excite=2 #The electronic state targeted by the excitation laser
              #This is just to spare looping through of resonant states  

A0x=100.0 #The intesity constant of the pusle (needed to completely empty the ground state)

w_pump_centre=20000.0 #Centre wavelength of the pulse (in 1/cm)
w_pump_width=10.0 #pulse width (in nm)

t_pump0=0.1 #peak time (in ps) of the pump pulse
t_pump_duration=70.0E-3 #pulse duraation (in ps)



"""****************************************************************************
Initial conditions
****************************************************************************"""

#Need to declare an array representing the initial vibronic populations. 
#this will have N_N_mode+1 indices
#The first index counts the electronic states (no need to include Sn) 
#The remnainig indices count the vibrational levels of each normal mode

#Create a list of dimensions
dim_nia=[] #dimensionality of the array of populations (i=electronic states, a=vibrational tuple)
dim_nia.append(N_elec) #first index will count the electronic states
for mode in range(N_mode): #loop of distinct vibrartional modes
    dim_nia.append(N_vib_level+1) #append number of vibrational levels for each mode
                              #must also include the zero point  

n_init=np.zeros(dim_nia) #initial initial population array. 

#Thermally populate the ground state
it = np.nditer(n_init[0], flags=['multi_index']) #this flattens the multi-dimensional list
                                                 #for element wise iteration while keeping track of 
                                                 #the multi-dimensional index
for element in it:
    a=it.multi_index #store the tuple of vibrational quantum numbers
    pop_index=[0] #the list of indices for the populations
    
    for i in a:
        pop_index.append(i)
    pop_index=tuple(pop_index) #convert into a tuple so we can use to refer to an element in n_init

    n_init[pop_index]=thermal_osc(a,w_mode,beta)

"""****************************************************************************
Calculate and assign IVR rates
****************************************************************************"""

#There is a forward and backward rate for each normal mode, on each electronic state
k_IVR=np.zeros((N_elec, N_mode, 2))

for i in range(N_elec): #interate over the electronic states
    for alpha in range(N_mode): #iterate over normal modes
        k_IVR[i][alpha][1], k_IVR[i][alpha][0]=k_alpha(w_mode[alpha],beta,L_IVR_i[i],G_IVR_i[i])

#Finally, we should turn these rates into 1/ps
k_IVR=cm_con.rate_cm_to_rate_ps(k_IVR)

"""****************************************************************************
Calculate and assign IC rates
****************************************************************************"""

#These rates possess several indices.
#The first 2 are the acceptor and donor electronic states (in that order)
#N_mode indicies representing the vibrational quantum numbers of the acceptor vibronic state
#N_mode indicies representing the vibrational quantum numbers of the donor vibronic state
#There is no need to include Sn

dim_IC=[N_elec,N_elec] #work out the dimensionality of the container for the rates
for mode in range(N_mode): #add the vibrational modes and levels of the acceptor state
    dim_IC.append(N_vib_level+1)
for mode in range(N_mode): #do it again for the donor state
    dim_IC.append(N_vib_level+1)

k_IC=np.zeros((dim_IC))

for i in range(N_elec):    #pairwise iteration over electronic manifolds
    for j in range(N_elec):
        
        if i!=j: #Can't have IC within a single state
            e_ij=(i,j) #indexes of the current pair of electronic states
            w_e=(wi[i],wi[j]) #transition energies of the two states
            w_alpha=tuple(w_mode) #tuple of vibrational frequencies of optical modes
            
            #Flatten and iterate remainder (due to non-fixed number of vibrational modes)
            it_vib = np.nditer(k_IC[i][j], flags=['multi_index'])
            for vib_element in it_vib:
                #the tuple it_vib.multi_index contains the vibrational quantum numbers 
                #for both electronic states. 
                #We first need to combine this with the electronic indicies for internal referencing of k_IC
                vib_all=it_vib.multi_index #all vibrational indices
                a=vib_all[:len(vib_all)//2] #vibrational indices for electronic state i (first half)
                b=vib_all[len(vib_all)//2:] #vibrational indices for electronic state j (second half)
     
                index_all=[i,j] #add the vibrational indexes to the electronic indexes
                for nu in vib_all:
                    index_all.append(nu)
                
                index_all=tuple(index_all) #convert the indices to a tuple to all specification of an element of k_IC

                #calculate the rates
                k_IC[index_all]=k_inter(e_ij,a,b,w_e,w_alpha,beta,L_IC_ij[i][j],G_IC_ij[i][j])

#Finally, we should turn these rates into 1/ps
k_IC=cm_con.rate_cm_to_rate_ps(k_IC)



"""****************************************************************************
Calculate and assign The FC overlap integrals
****************************************************************************"""

#The FC overlaps will be contained within a 5D array.
#Indices 1 and 2 will specify electronic states i and j. We nedd to include Sn 
#Index 3 specifies the normal mode
#Index 4 specifies the vibrational quantum number of that mode on state i (inlcudes the vib ground state)
#Index 5 specifies the vibrational quantum number of that mode on state j 

dim_FC=(N_elec+1,N_elec+1,N_mode,N_vib_level+1,N_vib_level+1)
FC=np.zeros((dim_FC))

for i in range(N_elec+1):
    for j in range(N_elec+1):
        if i!=j:
            for alpha in range(N_mode):
                FC_ij_alpha=FC_overlap(disp[alpha][i][j],N_vib_level,N_vib_level)
                
                for a in range(N_vib_level):
                    for b in range(N_vib_level):
                        FC[i][j][alpha][a][b]=FC_ij_alpha[a][b]
 


"""****************************************************************************
Population evolution
****************************************************************************"""

#timesteps for integration
ts=np.linspace(0,10,10000) #time in ps



#scipy.integrate.odeint requires a 1D input. We therefore have to flatten it,
#store the orignal shape and then rehsape the output. 
n_init_shape=np.shape(n_init)
n_init_flat=n_init.flatten()

nt_flat=odeint(dndt,n_init_flat,ts,\
               args=(n_init_shape,wi,w_mode,\
                     k_IVR,FC,k_IC,\
                     (elec_excite,A0x,w_pump_centre,w_pump_width,\
                      t_pump0, t_pump_duration)))

#restore the original shape of the population data for each timestep
nt=[] #empty container
for n_ts in nt_flat: #iternate through time-steps
    n_ts_reshape=n_ts.reshape((n_init_shape))
    nt.append(n_ts_reshape)

nt=np.array(nt)


#some test plotting
n_S0=[]
n_S1=[]
n_S2=[]
for i, time in enumerate(ts):
    n_S0.append(nt[i][0][0][0])
    n_S1.append(nt[i][1][0][0])
    n_S2.append(nt[i][2][0][0])



plt.plot(ts,n_S0,label='S0')
plt.plot(ts,n_S1,label='S1')
plt.plot(ts,n_S2,label='S2')
plt.show()

#some test plotting
n0_01=[]
n0_10=[]
for i, time in enumerate(ts):
    n0_01.append(nt[i][0][0][1])
    n0_10.append(nt[i][0][1][0])

plt.plot(ts,n_S1,label='S1')
plt.plot(ts,n0_01,label='S0_01')
plt.plot(ts,n0_10,label='S0_10')
plt.show()

"""****************************************************************************
File output
****************************************************************************"""

#out_file=open("populartion_evolution_full.txt","w+")
