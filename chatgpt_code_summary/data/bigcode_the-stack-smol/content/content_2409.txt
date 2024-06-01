import numpy as np
import h5py as py
import matplotlib.pyplot as plt
import sys

hdf5_file = py.File("..\\Build\\TestsWithGL\\t2d_mpm_chm_t_bar_conference_restart.hdf5", "r")
frame_id = 0

th_grp = hdf5_file['TimeHistory']['penetration']
pcl_dset = th_grp['frame_%d' % frame_id]['ParticleData']
pcl_num = pcl_dset.attrs['pcl_num']
print(pcl_num)

pcl_stress = np.zeros([pcl_num, 4])
p_min_id = 0
p_min =  sys.float_info.min
p_max_id = 0
p_max = -sys.float_info.max
for pcl_id in range(pcl_num):
    pcl_data = pcl_dset[pcl_id]
    pcl_stress[pcl_id][0] = pcl_data['s11']
    pcl_stress[pcl_id][1] = pcl_data['s22']
    pcl_stress[pcl_id][2] = pcl_data['s12']
    pcl_stress[pcl_id][3] = pcl_data['p']
    #p = pcl_stress[pcl_id][3]
    p = (pcl_stress[pcl_id][0] + pcl_stress[pcl_id][1] + pcl_stress[pcl_id][2]) / 3.0
    if (p < p_min):
        p_min = p
        p_min_id = pcl_id
    if (p > p_max):
        p_max = p
        p_max_id = pcl_id

print("p min: %f pcl %d\np max: %f pcl %d" % (p_min, p_min_id, p_max, p_max_id))
hdf5_file.close()
