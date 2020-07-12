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
dir_path = "/uboone/data/users/wvdp/searchingfornues/July2020/"

# for every entry in this dict, one output pickle will be created.
# the subdicts describes in which folder to look for input root files with the dict key names.
out_samples = {
    "nue": ["run1", "run3"],
    #"beam_on": ["run1", "run3"],
    #"beam_off": ["run1", "run2", "run3"],
    "nu": ["run1", "run3"],
    "dirt": ["run1", "run3"],
    "filter": ["run1", "run3"],
    #"set1": ["fake/run1", "fake/run3"],
    #"set2": ["fake/run1", "fake/run3"],
    #"set3": ["fake/run1", "fake/run3"],
    #"set4": ["fake/run1", "fake/run3"],
    #"beam_sideband": ["sideband"],
}

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


def load_truth_event(tree, period, sample_enum):
    mc_arrays = tree.arrays(
        col_load.table_cols | col_load.filter_cols, namedecode="utf-8"
    )
    # Fix weighting
    mc_arrays["weightSplineTimesTune"] = np.clip(
        np.nan_to_num(mc_arrays["weightSplineTimesTune"], nan=1, posinf=1, neginf=1),
        0,
        20,
    )

    has_fiducial_vtx = is_fid(
        mc_arrays["true_nu_vtx_x"],
        mc_arrays["true_nu_vtx_y"],
        mc_arrays["true_nu_vtx_z"],
    )
    has_electron = mc_arrays["nelec"] > 0
    signal_mask = has_fiducial_vtx & has_electron
    mc_arrays["true_fid_vol"] = has_fiducial_vtx
    mc_arrays["nueccinc"] = signal_mask
    mc_arrays["Run"] = np.repeat(period, len(signal_mask))
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
    mc_arrays["optical_filter"] = (tree.array("_opfilter_pe_beam") > 0) & (
        tree.array("_opfilter_pe_veto") < 20
    )
    mc_arrays["pdg12_broadcast"] = (tree.array("slpdg") == 12) * tree.array("n_pfps")
    mc_arrays["pdg14_broadcast"] = (tree.array("slpdg") == 14) * tree.array("n_pfps")
    # add the systematic weights for events with a slice:
    for col_mc in ["weightsFlux", "weightsGenie", "weightsReint"]:
        # Save the universes as float16 in a block numpy array for all events with a slice
        jagged_mc = (tree.array(col_mc)/1000).astype(np.float16)[tree.array("nslice")].regular()
        mc_arrays[col_mc] = np.clip(np.nan_to_num(jagged_mc,nan=1,posinf=1,neginf=1,),0,100)

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


def run_to_period(run):
    run2 = 8316
    run3 = 13696
    return 1 + (run > run2) + (run > run3)


# Function to load one type of sample:
def load_this_sample(sample_name, root_files):
    start = time.time()
    enum_keys = Sample_dict.keys()
    cols_load = col_load.cols_reco.copy()

    d = {}
    d["pot"] = {}
    d["triggers"] = {}
    d["fields"] = set()
    d["numentries"] = 0
    d["daughters"] = []
    truth = []
    tuples = []
    is_data = ("beam" in sample_name) or ("set" in sample_name)

    for fn in root_files:
        sample_type = max([e for e in enum_keys if e in fn], key=len)
        sample_enum = Sample_dict[sample_type]

        uproot_file = uproot.open(dir_path + fn)[root_dir]

        max_run = uproot_file["SubRun"].array("run").max()
        min_run = uproot_file["SubRun"].array("run").min()
        max_period = run_to_period(max_run)
        min_period = run_to_period(min_run)
        if max_period == min_period and max_run < 9999:
            period = min_period
        elif "run3" in fn:
            period = 3
        elif "run1" in fn:
            period = 1
        else:
            period = 0
            print("Input file contains mixture of data-taking periods.")
        print("\t", fn, sample_name, sample_type, sample_enum, period)

        this_fields = {f.decode() for f in uproot_file[main_tree].keys()}
        d["fields"] |= this_fields
        this_cols_load = cols_load | col_load.cols_run3

        if is_data:
            if "beam" in sample_name:
                scaling_file_name = dir_path + fn.split("/")[0] + "/scaling.txt"
                data_scaling = pd.read_csv(
                    scaling_file_name, index_col=0, sep="\t", header=None
                ).T.iloc[0]
                print('data_scaling', data_scaling)
                if sample_name in ["beam_on", "beam_sideband"]:
                    d["pot"][(sample_enum, period)] = data_scaling["tor875_wcut"]
                    d["triggers"][(sample_enum, period)] = data_scaling["E1DCNT_wcut"]
                    print(d["pot"][(sample_enum, period)])
                elif "off" in sample_name:
                    d["triggers"][(sample_enum, period)] = data_scaling["EXT"]
                    print(d["triggers"][(sample_enum, period)])
            else:  # fake datasets
                this_pot = uproot_file["SubRun"].array("pot").sum()
                d["pot"][(sample_enum, period)] = this_pot
        else:  # part only for mc samples:
            this_pot = uproot_file["SubRun"].array("pot").sum()
            d["pot"][(sample_enum, period)] = this_pot
            # Create the truth array
            truth.append(load_truth_event(uproot_file[main_tree], period, sample_enum))
            this_cols_load |= col_load.col_backtracked

        missing_columns = this_cols_load - this_fields
        availible_load_cols = this_cols_load - missing_columns
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
        if len(missing_columns):
            for col_not_avail in missing_columns:
                print("\t\tCol not found and defaulted with 9999:", col_not_avail)
                this_daughters[col_not_avail] = 9999

        # add reconstructed energy
        reco_e = uproot_file[main_tree].arrays(
            ["n_pfps", "shr_energy_tot_cali", "trk_energy_tot"], namedecode="utf-8"
        )
        this_daughters["reco_e"] = np.repeat(
            reco_e["shr_energy_tot_cali"] / 0.83 + reco_e["trk_energy_tot"],
            reco_e["n_pfps"],
        )

        numentries = uproot_file[main_tree].numentries
        d["numentries"] += numentries
        pass_rate = sum(uproot_file[main_tree].array("n_pfps") != 0) / numentries
        print("\t\t{:.0f} events\t NeutrinoID: {:.1%}".format(numentries, pass_rate,))

        d["daughters"].append(this_daughters)
        tuples.append((sample_enum, period))

    d["daughters"] = pd.concat(
        d["daughters"], sort=False, verify_integrity=True, copy=False, keys=tuples
    )
    d["daughters"].index.names = ["sample", "Run", "event", "daughter"]
    if not is_data:
        d["mc"] = {}
        for col_mc in truth[0].keys():
            d["mc"][col_mc] = awkward.concatenate([t[col_mc] for t in truth])
        print("\tSize of concatenated output:", len(d["mc"][col_mc]))
    print("pickling sample...")
    pickle.dump(
        d,
        open(dir_path + "/combined/{}_slimmed.pckl".format(sample_name), "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    end = time.time()
    print("\tCompleted, time passed: {:0.1f}s.".format(end - start))


# Gather the files:
all_root_files = glob.glob(dir_path + "**/*.root", recursive=True)
all_root_files = [f.replace(dir_path, "") for f in all_root_files]

print("Pickling protocol:", pickle.HIGHEST_PROTOCOL)

for sample in out_samples:
    print("\n\n-----", sample, "-----")
    sample_files = []
    for subset in out_samples[sample]:
        for f in all_root_files:
            if subset in f:
                possible_samples = [s for s in out_samples.keys() if s in f]
                if possible_samples:
                    if max(possible_samples, key=len) == sample:
                        sample_files.append(f)
    print(sample_files)
    load_this_sample(sample, sample_files)

