from interp_marcs_alpha_v6 import interp_marcs
import numpy as np
import time

input_model_path='/project2/alexji/MARCS'
output_model_path='test-MARCS'

teff_arr = [3200,3300,3400,3500,3600,3700,3800,3900,4000,4250,4500,4750,
            5000]
logg_arr = np.arange(0., 5.5, 0.5)
feh_arr = np.arange(-4., 1.5, 0.5)
alphafe_arr = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

teff_arr = [3400,4000,4281,5000]
logg_arr = [0.0, 0.47, 0.88,2.0]
feh_arr = [-3.77, -2.85, -1.23,-1.5]
alphafe_arr = [-0.77, -0.12, 0.23, 0.66,0.4]


if __name__=="__main__":
        start = time.time()
        for teff in teff_arr:
                for logg in logg_arr:
                        for feh in feh_arr:
                                for alphafe in alphafe_arr:
        
                                        print(teff, logg, feh, alphafe)
        
                                        interp_marcs(teff, logg, feh, alphafe,
                                        output_model_path=output_model_path,
                                        input_model_path=input_model_path,
                                        check_file_exists=True, extrapol=True,
                                        geometry='sph')
        print("took",time.time()-start,"s")
