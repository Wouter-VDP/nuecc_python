import numpy as np
import pandas as pd
import uproot
import awkward
import col_load
import time
import glob
import pickle
import os

from enum_sample import Sample_dict

### Constants
root_dir = "nuselection"
main_tree = "NeutrinoSelectionFilter"
dir_path = "/uboone/data/users/wvdp/searchingfornues/March2020/combined/"
exclude_samples = ['beam_on', 'beam_off' , "train", "dirt", "nu", "filter", 'nue']

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
# fid_vol = np.array([[5,6,20], [5,6,50]])
fid_vol = np.array([[10, 10, 20], [10, 10, 50]])
tpc_box = np.array([lower, upper]).T
fid_box = np.array([lower + fid_vol[0], upper - fid_vol[1]]).T


def is_in_box(x, y, z, box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z


def is_fid(x, y, z):
    return is_in_box(x, y, z, fid_box)


def is_tpc(x, y, z):
    return is_in_box(x, y, z, tpc_box)


def load_truth_event(tree, run_number, sample_enum):   
    mc_arrays = tree.arrays(
        col_load.table_cols | col_load.filter_cols, namedecode="utf-8"
    )
    # Fix weighting
    mc_arrays["weightSplineTimesTune"] = np.clip(np.nan_to_num(mc_arrays["weightSplineTimesTune"], nan=1, posinf=1, neginf=1), 0, 20)
    
    has_fiducial_vtx = is_fid(
        mc_arrays["true_nu_vtx_x"],
        mc_arrays["true_nu_vtx_y"],
        mc_arrays["true_nu_vtx_z"],
    )
    has_electron = mc_arrays["nelec"] > 0
    signal_mask = has_fiducial_vtx & has_electron
    mc_arrays['true_fid_vol'] = has_fiducial_vtx
    mc_arrays["nueccinc"] = signal_mask
    mc_arrays["Run"] = np.repeat(run_number, len(signal_mask))
    mc_arrays["sample"] = np.repeat(sample_enum, len(signal_mask)) 

    # Define categories of enriched filters:
    in_tpc = is_tpc(
        mc_arrays["true_nu_vtx_x"],
        mc_arrays["true_nu_vtx_y"],
        mc_arrays["true_nu_vtx_z"],
    )
    nuecc = (abs(mc_arrays["nu_pdg"]) == 12) & (mc_arrays["ccnc"] == 0) & in_tpc

    cc_pi0 = mc_arrays["mcf_pass_ccpi0"] == 1
    nc_pi0 = (
        (mc_arrays["mcf_np0"] == 1)
        & (mc_arrays["mcf_nmp"] == 0)
        & (mc_arrays["mcf_nmm"] == 0)
        & (mc_arrays["mcf_nem"] == 0)
        & (mc_arrays["mcf_nep"] == 0)
    )
    cc_nopi = (
        (mc_arrays["mcf_pass_ccnopi"] == 1)
        & (mc_arrays["n_pfps"] != 0)
        & (mc_arrays["slnunhits"] / mc_arrays["slnhits"] > 0.1)
    )

    cc_cpi = (
        (mc_arrays["mcf_pass_cccpi"] == 1)
        & (mc_arrays["n_pfps"] != 0)
        & (mc_arrays["slnunhits"] / mc_arrays["slnhits"] > 0.1)
    )
    nc_cpi = (
        (mc_arrays["mcf_pass_nccpi"] == 1)
        & (mc_arrays["n_pfps"] != 0)
        & (mc_arrays["slnunhits"] / mc_arrays["slnhits"] > 0.1)
    )
    nc_nopi = (
        (mc_arrays["mcf_pass_ncnopi"] == 1)
        & (mc_arrays["n_pfps"] != 0)
        & (mc_arrays["slnunhits"] / mc_arrays["slnhits"] > 0.1)
    )
    filter_cat = (
        4 * nuecc
        + 61 * cc_cpi
        + 62 * cc_pi0
        + 63 * cc_nopi
        + 71 * nc_cpi
        + 72 * nc_pi0
        + 73 * nc_nopi
    )
    mc_arrays["filter"] = filter_cat
    
    # optical filter:
    mc_arrays["optical_filter"] = (tree.array("_opfilter_pe_beam") > 0) & (tree.array("_opfilter_pe_veto") < 20)
    
    # add the systematic weights for events with a slice:
    for col_mc in ['weightsFlux', 'weightsGenie']:
        #mc_arrays[col_mc] = (tree.array(col_mc)*(tree.array('nslice')==1)).astype(np.float32).regular()
        jagged_mc = tree.array(col_mc).astype(np.float16)
        mask_mc = (jagged_mc.ones_like()*tree.array('nslice')).astype(np.bool)
        mc_arrays[col_mc] = jagged_mc[mask_mc]
        
    print("\t\t", np.unique(filter_cat, return_counts=True))

    for key in col_load.filter_cols:
        mc_arrays.pop(key)
    return mc_arrays


def calc_max_angle(tree):
    dir_x = tree.array("trk_dir_x_v")
    dir_y = tree.array("trk_dir_y_v")
    dir_z = tree.array("trk_dir_z_v")
    x1, x2 = dir_x.pairs(nested=True).unzip()
    y1, y2 = dir_y.pairs(nested=True).unzip()
    z1, z2 = dir_z.pairs(nested=True).unzip()
    cos_min = (
        (x1 * x2 + y1 * y2 + z1 * z2)
        / (np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))
    ).min()
    return awkward.topandas(cos_min, flatten=True).clip(upper=1)


# Function to load one type of sample:
def load_this_sample(dir_path, folder, root_files, data_scaling):
    start = time.time()
    sample_info = {}
    enum_keys = [e for e in Sample_dict.keys() if folder in e]
    cols_load = col_load.cols_reco.copy()
    fields = set()
    daughters = []
    truth = []
    tuples = []
    num_entries = 0

    # Scaling for data:
    if folder in ["beam_on", 'beam_sideband']:
        sample_info["pot"] = data_scaling["tor875_wcut"]
        sample_info["E1DCNT_wcut"] = data_scaling["E1DCNT_wcut"]
        sample_info["scaling"] = 1
    elif folder == "beam_off":
        sample_info["scaling"] = data_scaling["E1DCNT_wcut"] / data_scaling["EXT"]
        sample_info["EXT"] = data_scaling["EXT"]
    else:
        sample_info['pot']={}

    for fn in root_files:
        run_number = int(fn[3])
        print(enum_keys, fn)
        sample_name = max([e for e in enum_keys if e in fn], key=len)
        sample_enum = Sample_dict[sample_name]
        print("\t", fn, sample_name, sample_enum, run_number)

        uproot_file = uproot.open(dir_path + folder + "/" + fn)[root_dir]
        this_entries = uproot_file[main_tree].numentries
        num_entries += this_entries

        this_fields = {f.decode() for f in uproot_file[main_tree].keys()}
        fields |= this_fields
        cols_run3_add = col_load.cols_run3 - this_fields
        this_cols_load = cols_load | cols_run3_add
        
        # part only for mc samples:
        if folder not in ["beam_on", "beam_off", 'beam_sideband']:
            this_pot = uproot_file["SubRun"].array("pot").sum()
            sample_info['pot'][(sample_enum,run_number)] = this_pot
            # Create the truth array
            truth.append(
                load_truth_event(uproot_file[main_tree], run_number, sample_enum)
            )
            this_cols_load |= col_load.col_backtracked
        
        missing_columns = this_cols_load - this_fields
        availible_load_cols = this_cols_load - missing_columns
        print('\t\tMissing columns: {}'.format(str(missing_columns).strip('[]')))
        this_daughters = uproot_file[main_tree].pandas.df(
            availible_load_cols, flatten=True
        )
        this_daughters.index.names = ["event", "daughter"]
        duplicates = sum(
            this_daughters.xs(0, level="daughter")
            .groupby(by=["evt", "sub", "run", "reco_nu_vtx_z"])
            .size()
            > 1
        )
        if duplicates > 0:
            print("\t\tDuplicated events in sample: {}".format(duplicates))
        if len(cols_run3_add) < len(col_load.cols_run3):
            for col_not_avail in col_load.cols_run3 - cols_run3_add:
                print("\t\tCol not found and defaulted with 9999:", col_not_avail)
                this_daughters[col_not_avail] = 9999
        
        # add reconstructed energy   
        reco_e = uproot_file[main_tree].arrays(['n_pfps', 'shr_energy_tot_cali', 'trk_energy_tot'], namedecode="utf-8")  
        this_daughters['reco_e'] = np.repeat(reco_e['shr_energy_tot_cali']/0.83+reco_e['trk_energy_tot'], reco_e['n_pfps'])
        
        # this_daughters["trk_min_cos"] = calc_max_angle(uproot_file[main_tree])
        tuples.append((sample_enum, run_number))

        
        pass_rate = sum(uproot_file[main_tree].array("n_pfps") != 0) / this_entries
        print(
            "\t\t{:.0f} events\t NeutrinoID: {:.1%}".format(
                this_entries,
                pass_rate,
            )
        )

        daughters.append(this_daughters)
    sample_info["numentries"] = num_entries
    sample_info["daughters"] = pd.concat(daughters, sort=False, verify_integrity=True, copy = False, keys=tuples)
    sample_info["daughters"].index.names = ["sample", "Run", "event", "daughter"]
    sample_info["fields"] = fields
    if truth:
        sample_info["mc"] = {}
        for col_mc in truth[0].keys():
            sample_info["mc"][col_mc] = awkward.concatenate([t[col_mc] for t in truth])
        
        print("\tSize of concatenated output:", len(sample_info["mc"][col_mc]))
    end = time.time()
    print("\tCompleted, time passed: {:0.1f}s.".format(end - start))
    return sample_info


# Load, Add vars, Pickle!
pickle_bool = input("Do you want to pickle the data? (y/n) ") == "y"
sub_folders = filter(
    lambda x: os.path.isdir(dir_path + x) and x not in exclude_samples,
    os.listdir(dir_path),
)
data_scaling = pd.read_csv(
    dir_path + "scaling.txt", index_col=0, sep="\t", header=None
).T.iloc[0]

for folder in sub_folders:
    print("\n\n-----", folder, "-----")
    root_files = [os.path.basename(x) for x in glob.glob(dir_path + folder + "/*.root")]
    output = load_this_sample(dir_path, folder, root_files, data_scaling)
    if pickle_bool:
        pickle.dump(output, open(dir_path + "{}_slimmed.pckl".format(folder), "wb"))

print("Done!")
