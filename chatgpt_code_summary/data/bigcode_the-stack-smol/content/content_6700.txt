"""Euler explicit time advancement routine"""

from .projection import predictor, corrector, divergence
from .stats import stats

def advance_euler(gridc, gridx, gridy, scalars, grid_var_list, predcorr):

    """
    Subroutine for the fractional step euler explicit time advancement of Navier Stokes equations
 
    Arguments
    ---------
    gridc : object
          Grid object for cell centered variables

    gridx : object
          Grid object for x-face variables

    gridy : object
          Grid object for y-face variables

    scalars: object
           Scalars object to access time-step and Reynold number

    grid_var_list : list
           List containing variable names for velocity, RHS term from the previous time-step, divergence and pressure

    predcorr : string
           Flag for the fractional step method equations - 'predictor', 'divergence', 'corrector'

    """

    velc = grid_var_list[0]
    hvar = grid_var_list[1]
    divv = grid_var_list[2]
    pres = grid_var_list[3]

    if(predcorr == 'predictor'):

        # Calculate predicted velocity: u* = dt*H(u^n)
        predictor(gridx, gridy, velc, hvar, scalars.variable['Re'], scalars.variable['dt'])


    if(predcorr == 'divergence'):    
        # Calculate RHS for the pressure Poission solver div(u)/dt
        divergence(gridc, gridx, gridy, velc, divv, ifac = scalars.variable['dt'])


    elif(predcorr == 'corrector'):

        # Calculate corrected velocity u^n+1 = u* - dt * grad(P) 
        corrector(gridc, gridx, gridy, velc, pres, scalars.variable['dt'])
    
        # Calculate divergence of the corrected velocity to display stats
        divergence(gridc, gridx, gridy, velc, divv)
    
        # Calculate stats
        scalars.stats.update(stats(gridc, gridx, gridy, velc, pres, divv))
