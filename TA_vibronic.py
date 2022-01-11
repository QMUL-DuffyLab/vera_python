# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:49:57 2021

Code computes the TA spectrum of a molecular system described by the VERA
formalism. 

Input:
    (1) The TA spectrum in matrix form (intensity as a function of 1/cm)
    (2) The desired time intervals considered for the fit
    (3) a large number of parameters characterizing the vibronic system
    
Output:
    (1) Fit data in matrix form
    (2) png files of data fits
    (3) Text file containing the parameter set used to generate fit
    
Note: This currently uses a second order perturbative model of the population
dynamics of the vibronic system. Coherent effects are neglected and so dynamics 
at very short times (i.e. inside the pump pulse) are likely not meaningful.
 

@author: Chris Duffy
"""

import numpy as np
from scipy import interpolate

import Pop_dynamics1_2 as pop

import matplotlib.pyplot as plt

"""
(1) Lorentzian lineshape function

Input: (i) w=frequency/energy (in 1/cm). The variable. 
       (ii) Delta_j0_ba=vibronic energy gap (in 1/cm). The peak position essentially. 
       (iii) Delta_w=The FWHM of the line (in 1/cm)

Output: Single number amplitude of the spectral line at point w. 
"""

def Lorentz(w,Delta_j0_ba,Delta_w):
    return(((1.0/np.pi)*0.5*Delta_w)/(np.square(w-Delta_j0_ba)+np.square(0.5*Delta_w)))

"""
(2) Normalized Gaussian lineshape function

Input: (i) w=frequency/energy (in 1/cm). The variable. 
       (ii) Delta_j0_ba=vibronic energy gap (in 1/cm). The peak position essentially. 
       (iii) Delta_w=The FWHM of the line (in 1/cm)

Output: Single number amplitude of the spectral line at point w. 
"""

def Gaussian(w,Delta_j0_ba,Delta_w):
    #first convert the FHWM into a variance
    sig=Delta_w/(2.0*np.sqrt(2.0*np.log(2.0)))
    
    #normalization constant
    Norm=1.0/(sig*np.sqrt(2.0*np.pi))
    
    #shape function
    G=np.exp(-np.square(w-Delta_j0_ba)/(2.0*sig*sig))
    
    return(Norm*G)

"""****************************************************************************
*******************************************************************************
Input block (to be rewritten later for file input)
*******************************************************************************
****************************************************************************"""

"""****************************************************************************
Control parameters
****************************************************************************"""

#tuple of times we wish to display and fit (in ps)


#trace_times=(0.3,0.6,1.0,2.0)
trace_times=(1.0,5.0,10.0,20.0)
#trace_times=(10.0,20.0,42.0)
#trace_times=(1.4,2.4,4.0,4.7,7)

steps_per_ps=1000 #temporal resolution for the population dynamics

"""****************************************************************************
Parameters for the vibronic system 
****************************************************************************"""

beta=1/207 #1/207.0 #thermodynamics inverse temperature
swi_thermal='Y' #thermally-populate ground state?
                #Selecting other than 'Y' or 'y' will assume absolute ground state. 

#Parameters related to electronic states

N_elec=3 #number of electronic states (not including the 'Sn' state for ESA)
wi=[0.0,14050.0,20300.0,31950.0] #Electronic energies (in 1/cm) in ascending order 


#Reorganization energies (L in 1/cm) and damping times (G in fs) for Interconversion 
#from electronic state j -> i. The damping times are converted to 1/cm susequenctly
#They are assumed to be identical for each normal mode
#They do not need to be defined for the upper electronic state as this is merely
#an acceptor for ESA
#The definition is cumbersome but easier to manually alter

L_IC_ij=np.zeros((N_elec,N_elec)) 
L_IC_ij[0][1]=31.0#65.0 #easier to assign in this manner
L_IC_ij[1][2]=860.0#1100.0

G_IC_ij=np.zeros((N_elec,N_elec))
G_IC_ij[1][2]=34.46  #this is equivalent to 163.6 fs
G_IC_ij[0][1]=34.46

#Parameters associated with vibratonal levels

N_mode=2 #number of optically coupled normal modes
N_vib_level=3 #number of excited vibrational levels considered for each mode
w_mode=[1156.0, 1523.0]
#w_mode=[1156.0, 1522.5] #frequencies for the optically coupled modes (in 1/cm)
#1520
#Reorganization energies (L in 1/cm) and damping times (G in fs) for Intramolecular 
#Vibrartional Redistribution on a single electronic state. This does not need 
#to include upper electronic state as this is merely an acceptor of ESA

L_IVR_i=[15.0, 100.0, 150.0]

#L_IVR_i=[28.0, 110.0, 150.0]

G_IVR_i=[34.46, 34.46, 34.46] #this is equivalent to 163.6 fs


#Dimensionless displacements of equivalent modes on different electronic states
#for different modes. The indexing is disp[mode][elec_lower][elec_upper] and
#the frequency of the modes are defined in w_mode 
#We need to include the uppermost electronic state as the FC orverlap determines the ESA. 
disp=np.zeros((N_mode,N_elec+1,N_elec+1))

disp[0][0][1]=0.82#fix at 74 2PE
disp[0][1][2]=0.8 #approximate values from beta carotene
disp[0][0][2]=0.70# fix at 0.70 from steady state absorption
disp[0][1][3]=0.55#0.5

disp[1][0][1]=0.82#fix at 0.82 2PE
disp[1][1][2]=0.8#0.76#0.9 #approximate values from beta carotene 
disp[1][0][2]=0.71# fix at 0.84 from steady state absorption
disp[1][1][3]=0.0#0.0

#Testing

disp[0][1][0]=0.82#fix at 74 2PE
disp[0][2][1]=0.8 #approximate values from beta carotene
disp[0][2][0]=0.70# fix at 0.70 from steady state absorption
disp[0][3][1]=0.55#0.5

disp[1][1][0]=0.82#fix at 0.82 2PE
disp[1][2][1]=0.8#0.76#0.9 #approximate values from beta carotene 
disp[1][2][0]=0.71# fix at 0.84 from steady state absorption
disp[1][3][1]=0.0#0.0



"""****************************************************************************
Parameters for spectroscopic simulation 
****************************************************************************"""

#switches to indicate Gaussian ('gauss') or Lorentzian ('lorentz' or default)
#lineshapes for transitions (pairs of states). 
#This is because linear absorption is often Gaussian while other processes are 
#closer to lorentzian
swi_lineshapes=[]
for i in range(N_elec+1):
    line=[]
    for j in range(N_elec+1):
        line.append('lorentz')
    swi_lineshapes.append(line)
    
swi_lineshapes[0][2]='gauss' #linear absorption was found to be guassian
swi_lineshapes[1][3]='gauss'

osc_ij=np.zeros((N_elec+1,N_elec+1)) #amplutudes/oscillator strength for electronic transitions

osc_ij[1][3]=40.0
osc_ij[0][2]=osc_ij[1][3]*0.82#40.0 #ratio was found to be 1.4

D_wij=np.zeros((N_elec+1,N_elec+1)) #line widths for 
D_wij[0][2]=1190.0 #from steady_state absorption
D_wij[1][3]=1090.0#1200.0#1050.0 #from excited state absorption

D_stokes=150.0#Stokes shift for SE due to the rest of the bath. 

"""****************************************************************************
Parameters for the optical pump 
****************************************************************************"""

elec_excite=2 #The electronic state targeted by the excitation laser

A0x=200.0 #The intesity constant of the pusle (needed to completely empty the ground state)

w_pump_centre=wi[2] #Centre wavelength of the pulse (in 1/cm)
w_pump_width=10.0 #pulse width (in 1/cm)

t_pump0=0.0 #peak time (in ps) of the pump pulse
t_pump_duration=110.0E-3 #FWHM pulse duraation (in ps)

#wrap into a tuple to pass to the Pop function
pump_param=(elec_excite, A0x, w_pump_centre, w_pump_width, t_pump0, t_pump_duration)

"""****************************************************************************
TA data input (from file) 
****************************************************************************"""

t_TA=[] #list of timepoints (in ps) for the TA data
TA_matrix=[] #The TA matrix (columns=Energy, rows=times)

file_in=open("Lut_pyridine_TA.txt", 'r')
for i, line in enumerate(file_in):
    line=line.rstrip().split('\t')

    if i==0: #get the energy axis
        E_cm=line #list of Energies (in 1/cm) for the TA data
        E_cm.remove(E_cm[0]) #remove the first element which is blank (due to 2D nature of data)
    else:
        t_TA.append(line[0]) #time is the first entry in each row
        line.remove(line[0]) #having stored it, strip it out
        TA_matrix.append(line)
        
t_TA=np.array(t_TA)
TA_matrix=np.array(TA_matrix)
E_cm=np.array(E_cm)

#convert to floats
t_TA=t_TA.astype(float)
TA_matrix=TA_matrix.astype(float)
E_cm=E_cm.astype(float)


#select the plotting/fitting traces based on the tuple TA_times
t_TA_plot=[] #time points to be plotted
TA_matrix_plot=[]

for trace in trace_times:
    t_diff=abs(t_TA-trace) #finds the closest value
    trace_index=np.argmin(t_diff)      

    t_TA_plot.append(t_TA[trace_index])
    TA_matrix_plot.append(TA_matrix[trace_index])    

file_in.close()

"""****************************************************************************
*******************************************************************************
Calculation block
*******************************************************************************
****************************************************************************"""

"""****************************************************************************
Population evolution and interpolation
****************************************************************************"""

#timesteps for integration of population dynamics (in ps)
start_time=round(t_TA[0]) #earliest time in the trace (likely negative due to 0 being defined as pulse peak)
finish_time=round(trace_times[-1]) #the last time point we wish to fit
step_num=int(finish_time-start_time)*steps_per_ps
ts=np.linspace(start_time,finish_time,step_num) 

#numerical calculation of the population evolution
nt=pop.evol(beta,swi_thermal,N_elec,wi,L_IC_ij,G_IC_ij,N_mode,N_vib_level,w_mode,L_IVR_i,G_IVR_i,disp,pump_param,ts)

#interpolate to desired data points
nt_interp=interpolate.interp1d(ts,nt,axis=0) #interpolate along first axisof nt (time)
nt_fit=nt_interp(t_TA_plot)

"""****************************************************************************
Lineshapes
****************************************************************************"""

#FC overlaps
dim_FC=(N_elec+1,N_elec+1,N_mode,N_vib_level+1,N_vib_level+1)
FC=np.zeros((dim_FC))

for i in range(N_elec+1):
    for j in range(N_elec+1):
        if i!=j:
            for alpha in range(N_mode):
                FC_ij_alpha=pop.FC_overlap(disp[alpha][i][j],N_vib_level,N_vib_level)
                
                for a in range(N_vib_level+1):
                    for b in range(N_vib_level+1):
                        FC[i][j][alpha][a][b]=FC_ij_alpha[a][b]


#We create an of pure lineshape functions for all pairs of (optically allowed)
#vibronic transitions.
#These are not weighted by the populations of the vibronic levels
#dimensionality of the array of I

dim_I_ab_ij=[len(E_cm),N_elec+1,N_elec+1] #Freq index and then two electronic state i and j

for mode in range(N_mode): #add vibronic tuple (a) for each mode on state i 
    dim_I_ab_ij.append(N_vib_level+1) #include the zero point for each

for mode in range(N_mode): #add vibronic tuple (b) for each mode on state j 
    dim_I_ab_ij.append(N_vib_level+1) #include the zero point for each

I_ab_ij=np.zeros(dim_I_ab_ij)

I_ab_ij_SE=np.zeros(dim_I_ab_ij) #This a hack to introduce Stokes shifts (due
                                 #relaxation of the bath) in the stimulated emission

for j in range(N_elec+1): #loop over upper electronic states
    for i in range(j): #loop over lower electronic states
        if osc_ij[i][j]!=0.0: #only consider optically allowed transitions
            for w_bin, w in enumerate(E_cm): #loop over frequencies (spectrum)

                #loop over vibrational states on i            
                it_low=np.nditer(nt[0][0], flags=['multi_index']) #this flattens the multi-dimensional list
                for vib_low in it_low:
                    a=it_low.multi_index #vibrational indics on state i 
                                         #nt[0][0] is used only to obtain shape and indexes
                                         #we dont use the populations here
 
                    #loop over vibrational states on j                   
                    it_up=np.nditer(nt[0][0], flags=['multi_index']) #this flattens the multi-dimensional list
                    for vib_up in it_up:
                        b=it_up.multi_index #vibrational 

                        I_ab_ij_w=0.0 #container for ammplitude at current value of w

                        I_ab_ij_w_SE=0.0 #container for shifted ammplitude at current value of w
                            
                        #product of FC overlaps
                        FC_sq=1.0
                        for alpha in range(N_mode):
                            FC_sq=FC_sq*FC[i][j][alpha][a[alpha]][b[alpha]]*FC[i][j][alpha][a[alpha]][b[alpha]]

                        #true energy gap
                        Delta_ji_ba=wi[j]-wi[i] #pure electronic gap   

                        w_ba=[] #vibrational gap
                        for k, b_num in enumerate(b):
                            w_ba.append(w_mode[k]*(b[k]-a[k]))
                        w_ba=sum(w_ba)#

                        Delta_ji_ba=Delta_ji_ba+w_ba #full vbironic energy gap

                        if swi_lineshapes[i][j]=='gauss' or swi_lineshapes[j][i]=='gauss':
                            I_ab_ij_w=I_ab_ij_w+(osc_ij[i][j]*FC_sq*Gaussian(w,Delta_ji_ba,D_wij[i][j]))
                            
                            #The shifted lineshapes for the SE
                            I_ab_ij_w_SE=I_ab_ij_w_SE+(osc_ij[i][j]*FC_sq*Gaussian(w,Delta_ji_ba-D_stokes,D_wij[i][j]))
                        else:
                            I_ab_ij_w=I_ab_ij_w+(osc_ij[i][j]*FC_sq*Lorentz(w,Delta_ji_ba,D_wij[i][j]))

                            #The shifted lineshapes for the SE
                            I_ab_ij_w_SE=I_ab_ij_w_SE+(osc_ij[i][j]*FC_sq*Lorentz(w,Delta_ji_ba-D_stokes,D_wij[i][j]))
                        #Need to reassemble the complete index
                        I_index=[w_bin,i,j]
                        for a_num in a:
                            I_index.append(a_num)
                        for b_num in b:
                            I_index.append(b_num)
                        
                        I_index=tuple(I_index)

                        I_ab_ij[I_index]=I_ab_ij_w #bin the pure lineshape data

                        I_ab_ij_SE[I_index]=I_ab_ij_w_SE #bin the pure, shifted lineshape data

"""****************************************************************************
TA_spectral components
****************************************************************************"""

##Generate true line shapes functions via population weighting                        
##These are then dependent on both frequency and time
A_GSB=np.zeros((len(t_TA_plot),len(E_cm))) #Ground state bleach. 
                                           #Recovered ground state absorption minus
                                           #steady state absorption

A_ESA=np.zeros((len(t_TA_plot),len(E_cm))) #excited state absorption
A_SE=np.zeros((len(t_TA_plot),len(E_cm))) #stimulated emission

A_total=np.zeros((len(t_TA_plot),len(E_cm)))

#reminder
#nt(time, state, mode1 level, mode2 level)
#I_ab_ij=(w,lower state i,upper state j, mode1 on i, mode2 on i, mode1 on j, mode1 on j)
        

for t_step, time in enumerate(t_TA_plot): #loop over time
    for E_step, E in enumerate(E_cm): #lop over frequency/energy
                
        for j in range(1,N_elec+1): #loop over upper states
            for i in range(j): #loop over lower electronic states
                    #loop over vibrational states on i=0 (ground state)
                    
                    it_low=np.nditer(nt[0][0], flags=['multi_index']) #this flattens the multi-dimensional list
                    for vib_low in it_low:
                        a=it_low.multi_index #vibrational indics on state i 
    
                        #loop over vibrational states on j>0                   
                        it_up=np.nditer(nt[0][0], flags=['multi_index']) #this flattens the multi-dimensional list
                        for vib_up in it_up:
                            b=it_up.multi_index #vibrational index on j. 
                                             
                            #Assemble indexes
                            I_index=[E_step,i,j] #lineshape index
                            nt_index=[t_step,i] #population index
    
                            for a_num in a:
                                I_index.append(a_num)
                                nt_index.append(a_num)
                            for b_num in b:
                                I_index.append(b_num)
    
                            I_index=tuple(I_index)
                            nt_index=tuple(nt_index)
                        
                            if i==0: #i.e. the ground state

                                n0_index=[0,i]
                                for a_num in a:
                                    n0_index.append(a_num)
                                n0_index=tuple(n0_index)

                                #ground state absorptions
                                A_GSB[t_step,E_step]=A_GSB[t_step,E_step]+(nt_fit[nt_index]*I_ab_ij[I_index])
                                #steady state
                                A_GSB[t_step,E_step]=A_GSB[t_step,E_step]-(nt[n0_index]*I_ab_ij[I_index])

                            if i>0: #i.e. excited state effects
                                A_ESA[t_step,E_step]=A_ESA[t_step,E_step]+(nt_fit[nt_index]*I_ab_ij[I_index])


                            #stimulated emission
                            if j<N_elec:                            
                                nt_SE_index=[t_step,j]
                                for b_num in b:
                                    nt_SE_index.append(b_num)
                                nt_SE_index=tuple(nt_SE_index)    
                                
                                A_SE[t_step,E_step]=A_SE[t_step,E_step]-(nt_fit[nt_SE_index]*I_ab_ij_SE[I_index])
    

A_total=A_GSB+A_ESA+A_SE

"""****************************************************************************
*******************************************************************************
Plotting and output block
*******************************************************************************
****************************************************************************"""


#Diagnostic plotting

#generate data for the laser pulse. 
tau=pump_param[5]/1.76
pump_trace=(1.0/(2.0*tau))*(np.square(1.0/np.cosh((ts-pump_param[4])/tau)))
pump_trace=pump_trace/8.0

n_S0=[]
n_S1=[]
n_S2=[]
for i, time in enumerate(ts):
    n_S0.append(np.sum(nt[i][0]))
    n_S1.append(np.sum(nt[i][1]))
    n_S2.append(np.sum(nt[i][2]))

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ts,n_S0,label='S0 (total)')
plt.plot(ts,n_S1,label='S1 (total)')
plt.plot(ts,n_S2,label='S2 (total)')
plt.plot(ts,pump_trace,label='pump')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('t (ps)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Populations',fontsize=14)
plt.show()

#some test plotting
n1_00=[]
n1_01=[]
n1_10=[]
n1_20=[]
n1_11=[]
for i, time in enumerate(ts):
    n1_00.append(nt[i][1][0][0])
    n1_01.append(nt[i][1][0][1])
    n1_10.append(nt[i][1][1][0])
    n1_20.append(nt[i][1][2][0])
    n1_11.append(nt[i][1][1][1])

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ts,n1_00,label='S1(00)')
plt.plot(ts,n1_10,label='S1(10)')
plt.plot(ts,n1_01,label='S1(01)')
plt.plot(ts,n1_20,label='S1(20)')
plt.plot(ts,n1_11,label='S1(11)')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('t (ps)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Populations',fontsize=14)

plt.show()

#some test plotting
n0_00=[]
n0_01=[]
n0_10=[]
for i, time in enumerate(ts):
    n0_00.append(nt[i][0][0][0])
    n0_01.append(nt[i][0][0][1])
    n0_10.append(nt[i][0][1][0])

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(ts,n0_00,label='S0(00)')
plt.plot(ts,n0_01,label='S0(01)')
plt.plot(ts,n0_10,label='S0(10)')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('t (ps)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Populations',fontsize=14)

plt.show()

"""****************************************************************************
Fitting
****************************************************************************"""

#plot components
#GSB
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
for i, trace in enumerate(TA_matrix_plot):
    plt.plot(E_cm,A_GSB[i],label=str(trace_times[i])+' ps')
plt.title('Ground State Bleach')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('w (1/cm)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Intensity (arb.)',fontsize=14)
plt.show()

#ESA
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
for i, trace in enumerate(TA_matrix_plot):
    plt.plot(E_cm,A_ESA[i],label=str(trace_times[i])+' ps')

plt.title('Excited state Absorption')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('w (1/cm)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Intensity (arb.)',fontsize=14)
plt.show()

#SE
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
for i, trace in enumerate(TA_matrix_plot):
    plt.plot(E_cm,A_SE[i],label=str(trace_times[i])+' ps')
plt.title('Stimulated Emission')
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('w (1/cm)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Intensity (arb.)',fontsize=14)
plt.show()


#define colour pallete
colour_palate=[]
for i, trace in enumerate(TA_matrix_plot):
    colour_palate.append('C'+str(i))

#plot experimental traces and fits
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
for i, trace in enumerate(TA_matrix_plot):
    plt.plot(E_cm,trace,label=str(trace_times[i])+' ps',linewidth=5,alpha=0.6,color=colour_palate[i])

for i, trace in enumerate(TA_matrix_plot):
    if i==0:
        plt.plot(E_cm,A_total[i],label='fit',linestyle='dashed',color='k')
    else:
        plt.plot(E_cm,A_total[i],linestyle='dashed',color='k')
        
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.xlabel('w (1/cm)',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Intensity (arb.)',fontsize=14)
plt.show()


#pop.evol(beta,N_elec,wi,L_IC_ij,G_IC_ij,N_mode,N_vib_level,w_mode,L_IVR_i,G_IVR_i,disp,pump_param,ts)