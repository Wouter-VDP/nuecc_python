import numpy as np
from helpers import plot_class as plotter

# given a variation type, get the neccessary data arrays
def get_syst_data(data_dict, query, field, label, pot_target):
    nu_sample = "nu_" + label
    nue_sample = "nue_" + label
    cc_sample = "ccpi0_" + label
    nc_sample = "ncpi0_" + label
    if nc_sample not in data_dict:
        return False, None, None, None
    else:
        nu_view = data_dict[nu_sample]["daughters"].query(query)
        nue_view = data_dict[nue_sample]["daughters"].query(query)
        cc_view = data_dict[cc_sample]["daughters"].query(query)
        nc_view = data_dict[nc_sample]["daughters"].query(query)
        views = [nu_view, nue_view, cc_view, nc_view]
        data_array = np.hstack([v.eval(field) for v in views])
        filter_array = np.hstack([v["filter"] for v in views])
        weights_array = np.hstack([v["weightSplineTimesTune"] for v in views])

        # pot bookkeeping
        pot_nu = sum(data_dict[nu_sample]["pot"].values())
        pot_nue = sum(data_dict[nue_sample]["pot"].values()) + pot_nu
        pot_nc = sum(data_dict[nc_sample]["pot"].values()) + pot_nu
        pot_cc = sum(data_dict[cc_sample]["pot"].values()) + pot_nu
        pot_scale_nu = pot_target / pot_nu
        pot_scale_nue = pot_target / pot_nue
        pot_scale_nc = pot_target / pot_nc
        pot_scale_cc = pot_target / pot_cc

        weights_array[filter_array == 4] *= pot_scale_nue
        weights_array[filter_array == 62] *= pot_scale_cc
        weights_array[filter_array == 72] *= pot_scale_nc
        remaining = (filter_array != 4) & (filter_array != 62) & (filter_array != 72)
        weights_array[remaining] *= pot_scale_nu
        filter_array[remaining] = 0

        return True, data_array, filter_array, weights_array


# group bins together for better stats
def get_new_n_bins(N_bins, max_bin_syst):
    if N_bins <= max_bin_syst:
        return N_bins
    elif N_bins % 2 == 1:
        print("Odd number of bins greater than 6 is not supported!", N_bins)
        return N_bins
    else:
        return int(N_bins / 2)


# for a single
def get_syst_bins(x_min, x_max, new_bins, weights, data, tagger):
    values_dict = {}
    # for the enhanced samples:
    values_dict["nue"] = {}
    values_dict["nue"]["bins"], edges = np.histogram(
        data[tagger == 4],
        weights=weights[tagger == 4],
        range=(x_min, x_max),
        bins=new_bins,
    )
    values_dict["nue"]["err"] = plotter.hist_bin_uncertainty(
        data[tagger == 4], weights[tagger == 4], x_min, x_max, edges
    )
    values_dict["cc"] = {}
    values_dict["cc"]["bins"], edges = np.histogram(
        data[tagger == 62],
        weights=weights[tagger == 62],
        range=(x_min, x_max),
        bins=new_bins,
    )
    values_dict["cc"]["err"] = plotter.hist_bin_uncertainty(
        data[tagger == 62], weights[tagger == 62], x_min, x_max, edges
    )
    values_dict["nc"] = {}
    values_dict["nc"]["bins"], edges = np.histogram(
        data[tagger == 72],
        weights=weights[tagger == 72],
        range=(x_min, x_max),
        bins=new_bins,
    )
    values_dict["nc"]["err"] = plotter.hist_bin_uncertainty(
        data[tagger == 72], weights[tagger == 72], x_min, x_max, edges
    )
    values_dict["nu"] = {}
    values_dict["nu"]["bins"], edges = np.histogram(
        data[tagger == 0], weights=weights[tagger == 0], range=(x_min, x_max), bins=1
    )
    values_dict["nu"]["err"] = plotter.hist_bin_uncertainty(
        data[tagger == 0], weights[tagger == 0], x_min, x_max, edges
    )
    return values_dict


def get_syt_var(field, query, x_min, x_max, N_bins, data_dict, pot_target):
    filter_samples = ["nue", "cc", "nc", "nu"]
    max_bin_syst = 6
    dict_syst_split = {}
    dict_syst_merged = {}
    new_n_bins = get_new_n_bins(N_bins, max_bin_syst)

    # first fill data for the CV set:
    result, cv_data, cv_filter, cv_weights = get_syst_data(
        data_dict, query, field, "CV", pot_target
    )
    cv_full_bins, _ = np.histogram(
        cv_data,
        weights=cv_weights,
        range=(x_min, x_max),
        bins=N_bins,
    )/sum(cv_weights)
    
    if result:
        cv_values = get_syst_bins(
            x_min, x_max, new_n_bins, cv_weights, cv_data, cv_filter
        )

    labels = np.array([s.split("_")[1] for s in data_dict.keys() if "nue" in s])
    for lab in labels:
        if lab == "CV":
            continue
        result, var_data, var_filter, var_weights = get_syst_data(
            data_dict, query, field, lab, pot_target
        )
        if not result:
            print(lab)
            continue
        dict_syst_split[lab] = {}
        var_values = get_syst_bins(
            x_min, x_max, new_n_bins, var_weights, var_data, var_filter
        )

        # look at the differences between the variation and the CV:
        for sample in filter_samples:
            var_diff = abs(var_values[sample]["bins"] - cv_values[sample]["bins"])
            var_diff_sig = var_diff / np.sqrt(
                var_values[sample]["err"] ** 2 + cv_values[sample]["err"] ** 2
            )
            var_diff[var_diff_sig < 1] = 0
            # print(lab, sample, var_diff)
            dict_syst_split[lab][sample] = var_diff

    # group the variation effects together:
    dict_syst_merged["total"] = np.zeros(N_bins)
    for sample in filter_samples:
        dict_syst_merged[sample] = 0
        for lab in dict_syst_split:
            dict_syst_merged[sample] += dict_syst_split[lab][sample] ** 2
        dict_syst_merged[sample] = np.sqrt(dict_syst_merged[sample])
        # print(sample, dict_syst_merged[sample])
        # group and broadcast the filter samples together:
        if sample == "nu":
            dict_syst_merged["total"] += (
                (dict_syst_merged[sample] * cv_full_bins) ** 2
            )
        elif new_n_bins != N_bins:
            ori_bin_syst = np.zeros(N_bins)
            for i in range(new_n_bins):
                two_bin_norm = (cv_full_bins[2*i]+ cv_full_bins[2*i+1])
                ori_bin_syst[2*i] = dict_syst_merged[sample][i] * cv_full_bins[2*i] / two_bin_norm
                ori_bin_syst[2*i+1] = dict_syst_merged[sample][i] * cv_full_bins[2*i+1] / two_bin_norm
            dict_syst_merged["total"] += ori_bin_syst** 2
        else:
            dict_syst_merged["total"] += dict_syst_merged[sample] ** 2

    dict_syst_merged["total"] = np.sqrt(dict_syst_merged["total"])
    return dict_syst_merged["total"]
