import numpy as np
import pandas as pd
import scipy.stats

### Constants
mass_p = 0.93827
min_p_energy = mass_p + 0.04
min_e_energy = 0.020
data_samples = ['on', 'off']
gr = 1.618

phi_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
phi_labs = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
theta_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
theta_labs = [r"$0$", r"$\pi/4$", r"$\pi$/2", r"$3\pi$/4", r"$\pi$"]

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
fid_vol = np.array([[10,10,20], [10,10,50]])
contain_vol = np.array([[5,6,10], [5,6,10]])
fid_box = np.array([lower+fid_vol[0], upper-fid_vol[1]]).T
contain_box = np.array([lower+contain_vol[0], upper-contain_vol[1]]).T
tpc_box = np.array([lower, upper]).T

def is_in_box(x,y,z,box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z
def is_fid(x,y,z):
    return is_in_box(x,y,z,fid_box)
def is_contain(x,y,z):
    return is_in_box(x,y,z,contain_box)
def is_tpc(x,y,z):
    return is_in_box(x,y,z,tpc_box)
       
### Get the pitch
def get_pitch(dir_y, dir_z, plane):
    if plane == 0:
        cos = dir_y * (-np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 1:
        cos = dir_y * (np.sqrt(3)/2) + dir_z * (1/2)
    if plane == 2:
        cos = dir_z
    return 0.3 / cos

def effErr(num_w, den_w, symm=True):
    conf_level = 0.682689492137
    num_h = len(num_w)
    num_w_h = sum(num_w)
    num_w2_h = sum(num_w ** 2)
    den = len(den_w)
    den_w_h = sum(den_w)
    den_w2_h = sum(den_w ** 2)

    eff = num_w_h / den_w_h

    variance = (num_w2_h * (1.0 - 2 * eff) + den_w2_h * eff * eff) / (den_w_h * den_w_h)
    sigma = np.sqrt(variance)
    prob = 0.5 * (1.0 - conf_level)
    delta = -scipy.stats.norm.ppf(prob) * sigma
    if symm:
        return eff, delta
    else:
        if eff - delta < 0:
            unc_low = eff
        else:
            unc_low = delta
        if eff_i + delta_i > 1:
            unc_up = 1.0 - eff
        else:
            unc_up = delta
        return eff, unc_low, unc_up