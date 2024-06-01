#from numba import jit
import numpy as np
#from joblib import Parallel, delayed, parallel_backend
#from joblib import load, dump
#import tempfile
#import shutil
#import os
#
#import sys
#sys.path.append('pyunicorn_timeseries')
#from pyunicorn_timeseries.surrogates import Surrogates

def set_model_constants(xx=50.E3,nx=100,va=10.,tmax=60*360*24*3600.,avep=24*3600.,dt=3600.,period=3600*24*360*1,B=2.,T0=273.15+6,dT=2.,Cs=1.E-3,Cp=1030.,ra=1.5,ro=1030.,ri=900.,Cpo=4.E3,Cpi=2.9E3,H=200.,vo=0.2,Hb=1.E3,Li=3.3E6,Tf=273.15-1.8,SW0=50.,SW_anom=100.,emissivity=0.99,Da=1.E6,Do=5.E2,tau_entrainment=30*24*3600.,**args):
    '''Setup model constants. All of the constants have fixed values, but one can pass in own values or even some arbitrary values via **args.'''
    #
    C={}
    C['xx']    = xx #grid size in [m]
    C['nx']    = nx #number of grid cell - the total width of the domain is xx*nx long
    C['va']    = va #wind in m/s
    #
    C['tmax']  = tmax #tmax seconds
    C['dt']    = dt #timestep
    #
    C['avep']   = avep #averaging period in seconds 
    #
    C['period'] = period #period of boundary restoring
    C['Cs']     = Cs #exchange coefficient for bulk formula
    C['Cp']     = Cp #air heat capacity
    C['ra']     = ra #density of air [kg/m3]
    C['ro']     = ro #density of sea water [kg/m3]
    C['ri']     = ri #density of sea ice [kg/m3]
    C['Cpo']    = Cpo #sea water heat capacity
    C['T0']     = T0 #initial temp in degC
    C['dT']     = dT #initial temp perturbationHb=2E3
    C['H']      = H #mixed layer depth in ocean [m]
    C['vo']     = vo #ocean current speed [m/s]
    C['Hb']     = Hb #boundary layer height in the atmosphere [m]
    C['Cpi']    = Cpi #sea ice heat capacity [J/ Kg K]
    C['Li']     = Li #Latent heat of fusion of sea water [J / kg K]
    C['Tf']     = Tf #Freezing point of sea water [C]
    C['B']      = B # long-wave radiation constant [W/m2]
    C['emissivity'] = emissivity #surface emissivity
    C['SW0']    = SW0 # background net downwelling SW radiation
    C['SW_anom']= SW_anom # amplitude of annual cycle in SW radiation
    C['Da']     = Da # atmospheric diffusion [m2/s]
    C['Do']     = Do # ocean diffusion [m2/s]
    C['tau_entrainment'] = tau_entrainment # ocean entrainment/damping timescale
    
    for var in args.keys():
        C[var]=args[var]
    #
    return C

        
def CoupledChannel(C,forcing, T_boundary=None, dt_f=30*24*3600, restoring=False,ice_model=True,atm_adv=True,spatial_pattern=None,atm_DA_tendencies=None,ocn_DA_tendencies=None, return_coupled_fluxes=False,random_amp=0.1):
    '''
    This is the main function for the coupled ocean--atm channel model.
    
    ## INPUT VARIABLES ##
    
    tmax: running time in seconds
    avep: averaging period for the ouput
    T0: initial temperature
    forcing: dimensionless scaling for the heat flux forcing - default strength is 5 W/m2
    dt_f: timestep of the forcing
    atm_adv: boolean, advective atmosphere
    atm_ocn: boolean, advective ocean
    '''
    #
    # number of simulation timesteps and output timesteps
    nt   = int(C['tmax']/C['dt'])   #simulation
    nt1  = int(C['tmax']/C['avep']) #output
    # rtas = np.random.rand(C['nx'])
    # intitialize the model variables, first dimension is due to 2 timesteps deep scheme
    sst  = C['T0']*np.ones((2,C['nx']))
    tas  = C['T0']*np.ones((2,C['nx'])) #+rtas
    hice = np.zeros((2,C['nx']))
    # INCOMING SHORTWAVE RADIATION
    SW0   = np.tile(C['SW0'][:,np.newaxis],(1,nt))
    naxis = np.tile(np.arange(nt)[np.newaxis,],(C['nx'],1))
    SW_warming = np.max(np.concatenate([(SW0-C['SW_anom']*np.cos(2*np.pi*(naxis*C['dt'])/(360*24*3600)))[np.newaxis,],np.zeros((C['nx'],nt))[np.newaxis,]],axis=0),0)
    # If boundary conditions are not defined, then set initially to T0
    if np.all(T_boundary==None):
        T_boundary=C['T0']*np.ones(nt)
    #
    sst_boundary=T_boundary[0]*np.ones((2)) #nt+1
    #    evolve_boundary=True
    #else:
    #    sst_boundary=np.concatenate((sst_boundary[np.newaxis,],sst_boundary[np.newaxis,]),axis=0)
    #    evolve_boundary=False
    #
    # interpolate forcing to the new timescale
    if np.all(forcing!=None):
        forcing = np.interp(np.arange(0,len(forcing)*dt_f,C['dt']),np.arange(0,len(forcing)*dt_f,dt_f),forcing)
    else:
        forcing = np.zeros(nt+1)
    #
    # initialize outputs
    sst_out    = np.zeros((nt1,C['nx']))
    tas_out    = np.zeros((nt1,C['nx']))
    hice_out   = np.zeros((nt1,C['nx']))
    sflx_f_out = np.zeros((nt1,C['nx'])) #forcing
    sflx_out   = np.zeros((nt1,C['nx']))
    # spatial pattern of the forcing - assume a sine wave
    if np.all(spatial_pattern==None):
          spatial_pattern=np.ones(C['nx'])
    #
    if np.all(atm_DA_tendencies!=None):
        use_atm_tendencies=True
    else:
        use_atm_tendencies=False
    if np.all(ocn_DA_tendencies!=None):
        use_ocn_tendencies=True
    else:
        use_ocn_tendencies=False
    #
    if return_coupled_fluxes:
        atm_DA_tendencies = np.zeros((nt,C['nx']))
        ocn_DA_tendencies = np.zeros((nt,C['nx']))

    # initialize counters
    c=0; c2=0; c3=0; n=1
    #####################
    #  --- TIME LOOP ---
    #####################
    for nn in range(nt):
        #
        # FORCING - WILL BE ZERO IF NOT SPECIFIED, no spatial pattern if not specified
        sflx=forcing[nn]*spatial_pattern #+ forcing[nn]*random_amp*np.random.rand(C['nx'])
        #
        # save the forcing component
        #
        sflx_f_out[c,:]=sflx_f_out[c,:]+sflx
        #
        # SURFACE HEAT FLUXES
        # Add sensible heat flux to the total surface flux in W/m**-2
        sflx=sflx+C['ra']*C['Cp']*C['va']*C['Cs']*(sst[n-1,:]-tas[n-1,:])
        # RADIATIVE FLUXES - LW will cool the atmosphere, SW will warm the ocean
        LW_cooling = C['emissivity']*5.67E-8*(tas[n-1,:]**4)
        #
        # OCEAN BOUNDARY CONDITION
        #if evolve_boundary:
        sst_boundary_tendency=SW_warming[0,nn]*C['dt']/(C['H']*C['Cpo']*C['ro'])-C['emissivity']*5.67E-8*(sst_boundary[n-1]**4)*C['dt']/(C['H']*C['Cpo']*C['ro'])+(T_boundary[nn]-sst_boundary[n-1])*C['dt']/C['period']
        
        ############################################
        # 
        #              ATMOSPHERE
        #
        ############################################
        #
        # ADVECTION
        #
        # set atm_adv=False is no atmospheric advection - note that we still need to know the wind speed to resolve heat fluxes
        if atm_adv:
            a_adv = np.concatenate([sst_boundary[n-1]-tas[n-1,:1],tas[n-1,:-1]-tas[n-1,1:]],axis=0)*(C['va']*C['dt']/C['xx'])
        else:
            a_adv = 0 
        #
        # DIFFUSION
        # 
        a_diff  = (tas[n-1,2:]+tas[n-1,:-2]-2*tas[n-1,1:-1])*(C['Da']*C['dt']/(C['xx']**2))
        a_diff0 = (tas[n-1,1]+sst_boundary[n-1]-2*tas[n-1,0])*(C['Da']*C['dt']/(C['xx']**2))
        a_diff  = np.concatenate([np.array([a_diff0]),a_diff,a_diff[-1:]],axis=0)
        #
        # SURFACE FLUXES
        #
        a_netsflx = (sflx*C['dt'])/(C['Hb']*C['Cp']*C['ra']) - LW_cooling*C['dt']/(C['Hb']*C['Cp']*C['ra'])
        #
        #
        if return_coupled_fluxes:
            atm_DA_tendencies[nn,:] = a_adv + a_diff
        #
        # ATM UPDATE
        #
        if use_atm_tendencies:
            tas[n,:] = tas[n-1,:] + a_netsflx + atm_DA_tendencies[c3,:]
        else:
            tas[n,:] = tas[n-1,:] + a_netsflx + a_adv + a_diff
        #
        ################################################
        # 
        #                  OCEAN 
        #
        ################################################
        #  AND DIFFUSION + ENTRAINMENT
        # ocean advection
        # 
        # ADVECTION set vo=0 for stagnant ocean (slab)
        #
        o_adv = np.concatenate([sst_boundary[n-1]-sst[n-1,:1],sst[n-1,:-1]-sst[n-1,1:]],axis=0)*(C['vo']*C['dt']/C['xx'])
        #
        # DIFFUSION
        #
        o_diff = (sst[n-1,2:]+sst[n-1,:-2]-2*sst[n-1,1:-1])*(C['Do']*C['dt']/(C['xx']**2))
        o_diff0 = (sst[n-1,1]+sst_boundary[n-1]-2*sst[n-1,0])*(C['Do']*C['dt']/(C['xx']**2))
        o_diff = np.concatenate([np.array([o_diff0]),o_diff,o_diff[-1:]],axis=0)
        #
        # ENTRAINMENT - RESTORING TO AN AMBIENT WATER MASS (CAN BE SEEN AS LATERAL OR VERTICAL MIXING)
        # set tau_entrainment=0 for no entrainment
        if C['tau_entrainment']>0:
            o_entrain = (C['T0']-sst[n-1,:])*C['dt']/C['tau_entrainment']
        else:
            o_entrain = 0
        # 
        # SURFACE FLUXES        
        #
        o_netsflx = -sflx*C['dt']/(C['H']*C['Cpo']*C['ro'])+SW_warming[:,nn]*C['dt']/(C['H']*C['Cpo']*C['ro'])
        #
        if return_coupled_fluxes:
            ocn_DA_tendencies[nn,:] = o_adv + o_diff + o_entrain
        #
        # OCN update
        if use_ocn_tendencies:
            sst[n,:] = sst[n-1,:] + o_netsflx + ocn_DA_tendencies[c3,:]
        else:
            sst[n,:] = sst[n-1,:] + o_netsflx + o_adv + o_diff + o_entrain
        #
        if ice_model:
            # THIS IS A DIAGNOSTIC SEA ICE MODEL
            # 
            # SST is first allowed to cool below freezing and then we form sea ice from the excess_freeze 
            # i.e the amount that heat that is used to cool SST below freezing is converted to ice instead.
            # Similarly, SST is allowed to warm above Tf even if there is ice, and then excess_melt, 
            # i.e. the amount of heat that is used to warm the water is first used to melt ice, 
            # and then the rest can warm the water.
            #
            # This scheme conserves energy - it simply switches it between ocean and ice storages
            #
            # advection
            #hice[n-1,1:]=hice[n-1,1:]-(hice[n-1,:-1]-hice[n-1,1:])*(C['vo']*C['dt']/C['xx'])
            #dhice         = (hice[n-1,:-1]-hice[n-1,1:])*(C['vo']*C['dt']/C['xx'])
            #hice[n-1,:-1] = hice[n-1,:-1] -dhice
            #hice[n-1,-1]  = hice[n-1,-1] + dhice[-1]
            #
            ice_mask = (hice[n-1,:]>0).astype(np.float) #cells where there is ice to melt
            freezing_mask = (sst[n,:]<C['Tf']).astype(np.float) #cells where freezing will happen
            # change in energy
            dEdt = C['H']*C['ro']*C['Cpo']*(sst[n,:]-sst[n-1,:])/C['dt']
            # negative change in energy will produce ice whenver the water would otherwise cool below freezing
            excess_freeze = freezing_mask*np.max([-dEdt,np.zeros(C['nx'])],axis=0)
            # positive change will melt ice where there is ice
            excess_melt   = ice_mask*np.max([dEdt,np.zeros(C['nx'])],axis=0)
            # note that freezing and melting will never happen at the same time in the same cell
            # freezing
            dhice_freeze  = C['dt']*excess_freeze/(C['Li']*C['ri'])
            # melting
            dhice_melt= C['dt']*excess_melt/(C['Li']*C['ri'])
            # update
            hice[n,:]     = hice[n-1,:] + dhice_freeze - dhice_melt
            # check how much energy was used for melting sea ice - remove this energy from ocean
            hice_melt = (dhice_melt>0).astype(np.float)*np.min([dhice_melt,hice[n-1,:]],axis=0)
            # Do not allow ice to be negative - that energy is kept in the ocean all the time. 
            # The line above ensures that not more energy than is needed to melt the whole ice cover
            # is removed from the ocean at any given time
            hice[n,:] = np.max([hice[n,:],np.zeros(C['nx'])],axis=0)
            #
            # Update SST
            # Give back the energy that was used for freezing (will keep the water temperature above freezing)
            sst[n,:] = sst[n,:] + C['dt']*excess_freeze/(C['H']*C['Cpo']*C['ro']) 
            # take out the heat that was used to melt ice 
            # (need to cap to hice, the extra heat is never used and will stay in the ocean)
            sst[n,:] = sst[n,:] - hice_melt*(C['Li']*C['ri'])/(C['ro']*C['Cpo']*C['H'])
        #
        #############################
        # ---   PREPARE OUTPUT ----
        #############################
        # accumulate output
        tas_out[c,:]  = tas_out[c,:]+tas[n,:]
        sst_out[c,:]  = sst_out[c,:]+sst[n,:]
        hice_out[c,:] = hice_out[c,:]+hice[n,:]
        sflx_out[c,:] = sflx_out[c,:]+sflx
        # accumulate averaging counter
        c2=c2+1
        c3=c3+1
        if ((nn+1)*C['dt'])%(360*24*3600)==0:
            #print(nn)
            c3=0
        #calculate the average for the output
        if (((nn+1)*C['dt'])%C['avep']==0 and nn>0):
            tas_out[c,:]    = tas_out[c,:]/c2
            sst_out[c,:]    = sst_out[c,:]/c2
            sflx_out[c,:]   = sflx_out[c,:]/c2
            sflx_f_out[c,:] = sflx_f_out[c,:]/c2
            hice_out[c,:]   = hice_out[c,:]/c2
            # update counters
            c  = c+1
            c2 = 0
            if ((nn+1)*C['dt'])%(360*24*3600)==0:
                print('Year ', (nn+1)*C['dt']/(360*24*3600), sst[1,int(C['nx']/4)], sst[1,int(3*C['nx']/4)])
        #update the variables
        tas[0,:]  = tas[1,:].copy()
        sst[0,:]  = sst[1,:].copy()
        hice[0,:] = hice[1,:].copy()
        # SST at the boundary
        sst_boundary[n-1]=sst_boundary[n-1]+sst_boundary_tendency
        #
        #
    # if there is no ice, set to nan
    hice_out[np.where(hice_out==0)]=np.nan
    #
    if return_coupled_fluxes:
        return tas_out, sst_out, hice_out, sflx_out, sflx_f_out, nt1, nt, atm_DA_tendencies, ocn_DA_tendencies
    else:
        return tas_out, sst_out, hice_out, sflx_out, sflx_f_out, nt1, nt


#@jit(nopython=True)
def CoupledChannel_time(nt,nx,xx,dt,avep,sst,tas,hice,sst_boundary,sst_out,tas_out,hice_out,sflx_f_out,sflx_out,forcing,spatial_pattern,ra,Cp,va,vo,Da,Do,Cs,T0,Tf,emissivity,SW0,SW_anom,H,Hb,Cpo,ro,tau_entrainment,Li,ri,use_ocn_tendencies,use_atm_tendencies, atm_DA_tendencies, ocn_DA_tendencies,ice_model,atm_adv,return_coupled_fluxes):
    '''
    Separate time loop to enable numba
    '''
    #initialize counters
    c=0; c2=0; c3=0; n=1
    #####################
    #  --- TIME LOOP ---
    #####################
    for nn in range(nt):
        #
        # FORCING - WILL BE ZERO IF NOT SPECIFIED, no spatial pattern if not specified
        sflx=forcing[nn]*spatial_pattern #+ forcing[nn]*random_amp*np.random.rand(C['nx'])
        #
        # save the forcing component
        #
        sflx_f_out[c,:]=sflx_f_out[c,:]+sflx
        #
        # SURFACE HEAT FLUXES
        # Add sensible heat flux to the total surface flux in W/m**-2
        sflx=sflx+ra*Cp*va*Cs*(sst[n-1,:]-tas[n-1,:])
        # RADIATIVE FLUXES - LW will cool the atmosphere, SW will warm the ocean
        LW_cooling = emissivity*5.67E-8*(tas[n-1,:]**4)
        SW_warming = SW0+max(SW_anom*np.sin(2*float(nn)*dt*np.pi/(360*24*3600)),0.0)
        #net_radiation = SW_warming-LW_cooling 
        net_radiation = -LW_cooling
        #
        # OCEAN BOUNDARY CONDITION - SET dT to zero to suppress the sin
        sst_boundary[n]=sst_boundary[n-1]+SW_warming[0]*dt/(H*Cpo*ro)-emissivity*5.67E-8*(sst_boundary[n-1]**4)*dt/(H*Cpo*ro)+(T0-sst_boundary[n-1])*dt/(360*24*3600) #C['T0']+C['dT']*np.sin(nn*C['dt']*np.pi/C['period']) + 
        #
        # ATMOSPHERE - ADVECTION AND DIFFUSION
        # set atm_adv=False is no atmospheric advection - note that we need to know the wind speed to resolve heat fluxes
        if atm_adv:
            a_adv = np.concatenate((sst_boundary[n-1]-tas[n-1,:1],tas[n-1,:-1]-tas[n-1,1:]),axis=0)*(va*dt/xx)
            #tas[n,0]=tas[n-1,0]+(C['T0']-tas[n-1,0])*(C['va']*C['dt']/C['xx']) #always constant temperature blowing over the ocean from land
            #tas[n,0]=tas[n-1,0]+(sst[n,0]-tas[n-1,0])*(C['va']*C['dt']/C['xx']) #atmospheric temperature at the boundary is in equilibrium with the ocean
            #tas[n,1:]=tas[n-1,1:]+(tas[n-1,:-1]-tas[n-1,1:])*(C['va']*C['dt']/C['xx'])
        else:
            #tas[n,:]  = tas[n-1,0]
            a_adv = np.zeros(nx)
        #
        # DIFFUSION
        # 
        #tas[n,1:-1] = tas[n,1:-1] + (tas[n-1,2:]+tas[n-1,:-2]-2*tas[n-1,1:-1])*(C['Da']*C['dt']/(C['xx']**2))
        a_diff  = (tas[n-1,2:]+tas[n-1,:-2]-2*tas[n-1,1:-1])*(Da*dt/(xx**2))
        a_diff0 = (tas[n-1,1]+sst_boundary[n-1]-2*tas[n-1,0])*(Da*dt/(xx**2))
        a_diff  = np.concatenate((np.array([a_diff0]),a_diff,a_diff[-1:]),axis=0)
        #
        # ATMOSPHERE - SURFACE FLUXES
        #
        a_netsflx = (sflx*dt)/(Hb*Cp*ra) + net_radiation*dt/(Hb*Cp*ra)
        #
        # full update
        #
        #
        if return_coupled_fluxes:
            atm_DA_tendencies[nn,:]=np.sum((a_adv,a_diff),axis=0)
        #
        if use_atm_tendencies:
            tas[n,:] = tas[n-1,:] + a_netsflx + atm_DA_tendencies[c3,:]
        else:
            tas[n,:] = tas[n-1,:] + a_netsflx + a_adv + a_diff
        #
        # OCEAN - ADVECTION AND DIFFUSION + ENTRAINMENT
        # ocean advection
        # set vo=0 for stagnant ocean (slab)
        #
        #sst[n,1:] = sst[n-1,1:]+(sst[n-1,:-1]-sst[n-1,1:])*(1-ocn_mixing_ratio)*(C['vo']*C['dt']/C['xx'])+(C['T0']-sst[n-1,1:])*ocn_mixing_ratio*(C['vo']*C['dt']/C['xx'])
        o_adv = np.concatenate((sst_boundary[n-1]-sst[n-1,:1],sst[n-1,:-1]-sst[n-1,1:]),axis=0)*(vo*dt/xx)
        # DIFFUSION
        #sst[n,1:-1] = sst[n,1:-1] + (sst[n-1,2:]+sst[n-1,:-2]-2*sst[n-1,1:-1])*(C['Do']*C['dt']/(C['xx']**2))
        o_diff = (sst[n-1,2:]+sst[n-1,:-2]-2*sst[n-1,1:-1])*(Do*dt/(xx**2))
        o_diff0 = (sst[n-1,1]+sst_boundary[n-1]-2*sst[n-1,0])*(Do*dt/(xx**2))
        o_diff = np.concatenate((np.array([o_diff0]),o_diff,o_diff[-1:]),axis=0)
        # ENTRAINMENT (damping by a lower layer)
        o_entrain = (T0-sst[n-1,:])*dt/tau_entrainment
        #sst[n,1:]=sst[n,1:]+(C['T0']-sst[n-1,1:])*C['dt']/C['tau_entrainment']
        # 
        # OCEAN - SURFACE FLUXES        
        #
        o_netsflx = -sflx*dt/(H*Cpo*ro)+SW_warming*dt/(H*Cpo*ro)
        #sst[n,:]=sst[n,:]-(sflx*C['dt'])/(C['H']*C['Cpo']*C['ro'])
        if return_coupled_fluxes:
            ocn_DA_tendencies[nn,:] = o_adv + o_diff + o_entrain
        # OCN update
        if use_ocn_tendencies:
            sst[n,:] = sst[n-1,:] + o_netsflx + ocn_DA_tendencies[c3,:]
        else:
            sst[n,:] = sst[n-1,:] + o_netsflx + o_adv + o_diff + o_entrain
        #
        if ice_model:
            # THIS IS A DIAGNOSTIC SEA ICE MODEL
            # 
            # sst is first allowed to cool below freezing and then we forM sea ice from the excess_freeze 
            # i.e the amount that heat that is used to cool sst below freezing is converted to ice instead
            # similarly sst is allowed to warm above Tf even if there is ice, and then excess_melt, 
            # i.e. the amount of heat that is used to warm the water is first used to melt ice, 
            # and then the rest can warm water. This scheme conserves energy - it simply switches it between ocean and ice
            #
            ice_mask = (hice[n-1,:]>0).astype(np.float) #cells where there is ice to melt
            freezing_mask = (sst[n,:]<Tf).astype(np.float) #cells where freezing will happen
            # change in energy
            dEdt = H*ro*Cpo*(sst[n,:]-sst[n-1,:])/dt
            # negative change in energy will produce ice whenver the water would otherwise cool below freezing
            excess_freeze = freezing_mask*np.max([-dEdt,np.zeros(nx)],axis=0)
            # positive change will melt ice where there is ice
            excess_melt   = ice_mask*np.max([dEdt,np.zeros(nx)],axis=0)
            # note that freezing and melting will never happen at the same time in the same cell
            # freezing
            dhice_freeze  = dt*excess_freeze/(Li*ri)
            # melting
            dhice_melt= dt*excess_melt/(Li*ri)
            # update
            hice[n,:]     = hice[n-1,:] + dhice_freeze - dhice_melt
            # check how much energy was used for melting sea ice - remove this energy from ocean
            hice_melt = (dhice_melt>0).astype(np.float)*np.min([dhice_melt,hice[n-1,:]],axis=0)
            # Do not allow ice to be negative - that energy is kept in the ocean all the time. 
            # The line above ensures that not more energy than is needed to melt the whole ice cover
            # is removed from the ocean at any given time
            hice[n,:] = np.max([hice[n,:],np.zeros(nx)],axis=0)
            #
            # Update SST
            # Give back the energy that was used for freezing (will keep the water temperature above freezing)
            sst[n,:] = sst[n,:] + dt*excess_freeze/(H*Cpo*ro) 
            # take out the heat that was used to melt ice 
            # (need to cap to hice, the extra heat is never used and will stay in the ocean)
            sst[n,:] = sst[n,:] - hice_melt*(Li*ri)/(ro*Cpo*H)
        #
        #############################
        # ---   PREPARE OUTPUT ----
        #############################
        #accumulate
        tas_out[c,:]  = tas_out[c,:]+tas[n,:]
        sst_out[c,:]  = sst_out[c,:]+sst[n,:]
        hice_out[c,:] = hice_out[c,:]+hice[n,:]
        sflx_out[c,:] = sflx_out[c,:]+sflx
        # accumulate averaging counter
        c2=c2+1
        c3=c3+1
        if ((nn+1)*dt)%(360*24*3600)==0:
            #print(nn)
            c3=0
        #calculate the average for the output
        if (((nn+1)*dt)%avep==0 and nn>0):
            tas_out[c,:]    = tas_out[c,:]/c2
            sst_out[c,:]    = sst_out[c,:]/c2
            sflx_out[c,:]   = sflx_out[c,:]/c2
            sflx_f_out[c,:] = sflx_f_out[c,:]/c2
            hice_out[c,:]   = hice_out[c,:]/c2
            # update counters
            c  = c+1
            c2 = 0
            #if ((nn+1)*C['dt'])%(360*24*3600)==0:
            #    print('Year ', (nn+1)*C['dt']/(360*24*3600), sst[1,int(C['nx']/4)], sst[1,int(3*C['nx']/4)])
        #update the variables
        tas[0,:]  = tas[1,:].copy()
        sst[0,:]  = sst[1,:].copy()
        hice[0,:] = hice[1,:].copy()
        sst_boundary[0]=sst_boundary[1].copy()
    #
    hice_out[np.where(hice_out==0)]=np.nan
    #
    return tas_out, sst_out, hice_out, sflx_out, sflx_f_out, atm_DA_tendencies, ocn_DA_tendencies

def CoupledChannel2(C,forcing, dt_f=30*24*3600, ocn_mixing_ratio=0, restoring=False,ice_model=True,atm_adv=True,spatial_pattern=None,atm_DA_tendencies=None,ocn_DA_tendencies=None, return_coupled_fluxes=False,random_amp=0.1):
    '''
    This is the main function for the coupled ocean--atm channel model.
    
    ## INPUT VARIABLES ##
    
    tmax: running time in seconds
    avep: averaging period for the ouput
    T0: initial temperature
    forcing: dimensionless scaling for the heat flux forcing - default strength is 5 W/m2
    dt_f: timestep of the forcing
    atm_adv: boolean, advective atmosphere
    atm_ocn: boolean, advective ocean
    ocn_mixing: add non-local mixing to ocean
    ocn_mixing_ratio: 0-1 ratio between advection and mixing (0 only advection; 1 only mixing)
    
    '''
    #
    #print(C)
    #print(C['T0'],C['SW0'],C['Da'],C['xx'])
    #
    nt=int(C['tmax']/C['dt']) #steps
    nt1=int(C['tmax']/C['avep'])
    tau=float(C['period'])/float(C['dt']) #this is period/dt, previously nt/8 
    rtas=np.random.rand(C['nx'])
    #print(rtas.max())
    #intitialize the model variables, only 2 timesteps deep scheme
    sst=C['T0']*np.ones((2,C['nx']))
    tas=C['T0']*np.ones((2,C['nx']))+rtas
    hice=np.zeros((2,C['nx']))
    sst_boundary=C['T0']*np.ones((2))
    #
    #print(sst.max(),tas.max())
    #interpolate forcing to the new timescale
    if np.all(forcing!=None):
        forcing = np.interp(np.arange(0,len(forcing)*dt_f,C['dt']),np.arange(0,len(forcing)*dt_f,dt_f),forcing)
    else:
        forcing = np.zeros(nt+1)
    #
    #initialize outputs
    sst_out    = np.zeros((nt1,C['nx']))
    tas_out    = np.zeros((nt1,C['nx']))
    hice_out   = np.zeros((nt1,C['nx']))
    sflx_f_out = np.zeros((nt1,C['nx'])) #forcing
    sflx_out   = np.zeros((nt1,C['nx']))
    #spatial pattern of the forcing - assume a sine wave
    if np.all(spatial_pattern==None):
          spatial_pattern=np.ones(C['nx'])
    #
    if np.all(atm_DA_tendencies!=None):
        use_atm_tendencies=True
    else:
        use_atm_tendencies=False
    if np.all(ocn_DA_tendencies!=None):
        use_ocn_tendencies=True
    else:
        use_ocn_tendencies=False
    #
    atm_DA_tendencies = np.zeros((nt,C['nx']))
    ocn_DA_tendencies = np.zeros((nt,C['nx']))
    #
    tas_out, sst_out, hice_out, sflx_out, sflx_f_out, atm_DA_tendencies, ocn_DA_tendencies=CoupledChannel_time(nt,C['nx'],C['xx'],C['dt'],C['avep'],sst,tas,hice,sst_boundary,sst_out,tas_out,hice_out,sflx_f_out,sflx_out,forcing,spatial_pattern,C['ra'],C['Cp'],C['va'],C['vo'],C['Da'],C['Do'],C['Cs'],C['T0'],C['Tf'],C['emissivity'],C['SW0'],C['SW_anom'],C['H'],C['Hb'],C['Cpo'],C['ro'],C['tau_entrainment'],C['Li'],C['ri'],use_ocn_tendencies,use_atm_tendencies, atm_DA_tendencies, ocn_DA_tendencies,ice_model,atm_adv,return_coupled_fluxes)
    #
    if return_coupled_fluxes:
        return tas_out, sst_out, hice_out, sflx_out, sflx_f_out, nt1, nt, atm_DA_tendencies, ocn_DA_tendencies
    else:
        return tas_out, sst_out, hice_out, sflx_out, sflx_f_out, nt1, nt    
    