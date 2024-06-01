import numpy as np
import numpy.ma as npma
from scipy import stats
import matplotlib.pyplot as plt
import baspy as bp
import fnmatch

"""
Created on Wed Nov 27 18:34 2019

@author: Christine McKenna

========================================================================
Purpose: Plots Supp Fig 2, a pdf of all possible 20-year trends in gsat 
         for CMIP6 piControl simulations for each model. First detrends
         the raw gsat time series to remove any long term drift,
         which could bias 20-year trends (e.g. if positive drift,
         pdf of 20-year trends likely biased positive).
         Saves pdf of 20-year trends for models used in Supp Fig 8. 
========================================================================
"""


# Required directories
loaddir_CMIP = 'Priestley-Centre/Near_term_warming/analysis_figure_code/'+\
               'SuppFig2/saved_arrays'
savedir = 'Priestley-Centre/Near_term_warming/analysis_figure_code/'+\
          'SuppFig8/saved_data'


### ------ Load in CMIP6 data ------

# Load models
models = np.load(loaddir_CMIP+'/models_gtas_CMIP6_piControl.npy')

# Load catalogue so can extract runids
var = 'tas'
cat_PI = bp.catalogue(dataset='cmip6',Var=var,Experiment='piControl',\
                      CMOR='Amon')
years = np.linspace(1,20,20)

### Process data, one model and RunID at a time
i = 0
fig,axs = plt.subplots(7,7,sharex=True,sharey=True,\
                       figsize=(15,12))
fig.suptitle('PDFs of rolling GSAT trends for 20-year segments of CMIP6 '+\
             'piControl runs',fontsize=20)
axs = axs.ravel()

for model in models:

    ## Get data for model
    filtmod_PI = cat_PI[cat_PI['Model'] == model]

    ## Only keep r1i1p1f?
    runids_PI = np.unique(filtmod_PI['RunID'])
    runids_PI = fnmatch.filter(runids_PI,'r1i1p1f?')

    ## Get data for each RunID
    for runid in runids_PI: 

        ## Load gsat data
        gsat_tmp = np.load(loaddir_CMIP+'/gtas_'+model+'_'+runid+\
                           '_CMIP6_piControl.npy')
        ny = len(gsat_tmp)

        ## Remove any drift
        [m,c,_,_,_] = stats.linregress(np.linspace(0,ny-1,ny),gsat_tmp)
        gsat_lin = m*np.linspace(0,ny-1,ny)+c
        gsat = gsat_tmp - gsat_lin

        ## Calculate trends
        gsat_trends = np.zeros([ny-20])
        for y in xrange(0,ny-20):
            [m,_,_,_,_] = stats.linregress(years,gsat[y:y+20])
            gsat_trends[y] = m*10

        ## If model used in Supp Fig 8 save pdf of 20y trends
        if (model == 'BCC-CSM2-MR') or (model == 'MIROC-ES2L'):
            np.save(savedir+'/gsat_20ytrends_CMIP6_piControl_'+\
                    model+'.npy',gsat_trends) 

        
        ### ------ Plot results ------

        ### Plot individual models      
        axs[i].hist(gsat_trends,density=True)
        axs[i].set_title(model,fontsize=13) 
        axs[i].plot(np.zeros([2]),[0,11],'grey',linewidth=1)
        axs[i].plot(np.ones([2])*(-0.075),[0,11],'black',\
                    linewidth=1,linestyle='--')
        axs[i].plot(np.ones([2])*(0.072),[0,11],'black',\
                    linewidth=1,linestyle='--')
        axs[i].plot(np.ones([2])*(-0.084),[0,11],'black',\
                    linewidth=1,linestyle='--')
        axs[i].plot(np.ones([2])*(0.094),[0,11],'black',\
                    linewidth=1,linestyle='--')
        axs[i].tick_params(labelsize=13)
        i += 1


fig.text(0.5,0.02,'$^{\circ}$C / decade',ha='center',\
         va='center',fontsize=18)
fig.text(0.02,0.5,'Probability density',ha='center',va='center',\
         rotation='vertical',fontsize=18)
axs[i-1].set_xlim([-0.3,0.3])
axs[i-1].set_ylim([0,11])
axs[i].axis('off')
plt.subplots_adjust(top=0.9,bottom=0.07,left=0.07,right=0.97,\
                    wspace=0.17,hspace=0.27)
plt.show()

