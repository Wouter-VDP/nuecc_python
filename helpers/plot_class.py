import numpy as np
import pandas as pd
import uproot
import scipy.stats
from helpers import plot_dicts_nue
from helpers import plot_dicts_numu
from helpers import helpfunction
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gc
from joblib import Parallel, delayed
import time

text_dict = {0: ("left", 0.02), 1: ("center", 0.5), 2: ("right", 0.98)}


class Plotter:
    """
    Class to make self.data/MC plots
    Initialise using a data dictionary as produced by RootMerger.py
    """

    gr = 1.618
    grouper = ["sample", "Run", "event"]

    # location: path to pckl dictionary
    # signal: nue or numu
    # genie version: mcc9, mcc9.0 or mcc9.1
    # norm_pot: scale plots to a fixed pot, currently breaka KS test
    # master_query: applied on daughters for all plots, reduced memory
    # beam_on: on or sideband
    # pot_dict: overwrites the pot in the pckled file
    # load_syst: string set with keys in the ['mc'] dict
    # show_lee: show lee_model by default on plots, bool
    # pi0_scaling: apply a predefined pi0 scaling on the MC
    # dirt: bool, do you wan to include dirt info?
    # n_uni_max: per systematic variation, what is the max number of universes we want to use.
    def __init__(
        self,
        location,
        signal="nue",
        genie_version="mcc9.1",
        norm_pot=0,
        master_query=None,
        beam_on="on",
        pot_dict={},
        load_syst=None,
        show_lee=False,
        pi0_scaling=False,
        dirt=True,
        n_uni_max=2000,
    ):

        self.signal = signal
        self.syst_weights = {}
        self.show_lee = show_lee
        self.ratio_purity_dict = {}
        self.n_uni_max = n_uni_max

        data = self.load_data(location, beam_on, master_query, load_syst, dirt)
        self.keys = set(data["nu"]["daughters"].keys())

        weights_field = "weightSplineTimesTune"
        # Use the Genie v2 model by upweighting.
        if genie_version == "mcc8":
            numu_spline = uproot.open("./helpers/numu_v13_v12_Ratio.root")
            nue_spline = uproot.open("./helpers/nue_v13_v12_Ratio.root")
            spline_x = np.append(numu_spline["Graph"].__dict__["_fX"], 100)
            spline_y_numu = np.append(numu_spline["Graph"].__dict__["_fY"], 1)
            spline_y_nue = np.append(nue_spline["Graph"].__dict__["_fY"], 1)

            mask_nu = data["nu"]["daughters"].eval("abs(nu_pdg)==14")
            bins_nu_e = np.digitize(data["nu"]["daughters"]["nu_e"], spline_x)

            data["nu"]["daughters"]["weightSpline"] = (
                ~mask_nu / spline_y_nue[bins_nu_e] + mask_nu / spline_y_numu[bins_nu_e]
            )
            print(
                "Using the energy/pdg dependent spline weights as in MCC8 Genie V2 tune1"
            )
            weights_field = "weightSpline"
        if genie_version == "mcc9.0":
            weights_field = "weightSpline"
            print("Using the spline weights as in MCC9.0 Genie V3")
        if genie_version == "mcc9.1":
            print("Using the spline weights as in MCC9.1 Genie V3 tune 1")
            if pi0_scaling:
                weights_field = "weightSplineTimesTune_pi0scaled"
                print(
                    "Empirical scaling of 1-0.4*E applied on events with neutral pions."
                )

        # Default values loaded and can be overwritten with dict.
        print('data[beam_on]["pot"]', data[beam_on]["pot"])
        print('data[beam_on]["triggers"]', data[beam_on]["triggers"])
        print('data[beam_off]["triggers"]', data["off"]["triggers"])
        data[beam_on]["pot"] = sum(data[beam_on]["pot"].values())
        data[beam_on]["triggers"] = sum(data[beam_on]["triggers"].values())
        data["off"]["triggers"] = sum(data["off"]["triggers"].values())
        if len(pot_dict):
            if "pot" in pot_dict:
                data[beam_on]["pot"] = pot_dict["pot"]
            if "E1DCNT_wcut" in pot_dict:
                data[beam_on]["triggers"] = pot_dict["E1DCNT_wcut"]
            if "EXT" in pot_dict:
                data["off"]["triggers"] = pot_dict["EXT"]

        self.title_str = r"MicroBooNE {:.1e}$\,$POT, Preliminary".format(
            data[beam_on]["pot"]
        ).replace("+", "")

        # We want to fill the label plot weight to match the norm_pot, if 0, use Beam_on_norm
        if norm_pot == 0:
            norm_scale = 1
        else:
            norm_scale = norm_pot / data[beam_on]["pot"]
        print(
            "All plots are scaled to {:.2e} POT, Beam_on data corresponds to {:.2e} POT".format(
                norm_scale * data[beam_on]["pot"], data[beam_on]["pot"]
            )
        )

        data[beam_on]["daughters"]["plot_weight"] = norm_scale
        data["off"]["scaling"] = data[beam_on]["triggers"] / data["off"]["triggers"]
        data["off"]["daughters"]["plot_weight"] = data["off"]["scaling"] * norm_scale

        self.on_daughters = data[beam_on]["daughters"]
        self.off_daughters = data["off"]["daughters"]
        self.mc_daughters = pd.concat(
            [data["nu"]["daughters"], data["dirt"]["daughters"]],
            copy=False,
            sort=False,
        )
        del data["nu"]["daughters"]
        gc.collect()
        print("Loaded all daughter dataframes.")

        # The samples were produced assuming 1e21 as event_scale
        self.mc_daughters["plot_weight"] = (
            self.mc_daughters["event_scale"]
            * norm_scale
            * (data[beam_on]["pot"] / 1e21)
            * self.mc_daughters[weights_field]
        )

        if load_syst:
            print("Started loading systematic weights.")
            plot_weight = (
                self.mc_daughters["plot_weight"]
                .groupby(self.grouper, sort=False)
                .first()
            )
            print()
            for type_w in load_syst:
                self.syst_weights[type_w] = np.multiply(
                    np.vstack([data["nu"]["mc"][type_w], data["dirt"]["mc"][type_w]]),
                    plot_weight[:, np.newaxis],
                )
                print("Loaded all universes for {}.".format(type_w))

        del data
        gc.collect()

        if signal == "nue":
            self.dicts = plot_dicts_nue
            self.cats = [1, 10, 11]
        elif signal == "numu":
            self.dicts = plot_dicts_numu
            self.cats = [30, 31, 32]
        else:
            print("Error, unknown signal string, choose nue or numu!")

        print("Initialisation completed!")

    # Load the pickled dataframe from location
    def load_data(self, location, beam_on, master_query, load_syst, dirt):
        data = pd.read_pickle(location)
        required_keys = {"nu", beam_on, "off", "dirt"}
        [data.pop(key) for key in set(data.keys()) - required_keys]

        if not all([k in data.keys() for k in required_keys]):
            print("Error, missing samples in the data set!")

        if master_query:
            data[beam_on]["daughters"].query(master_query, inplace=True)
            data["off"]["daughters"].query(master_query, inplace=True)
            dirt_eval = data["dirt"]["daughters"].eval(master_query)
            nu_eval = data["nu"]["daughters"].eval(master_query)
            data["dirt"]["daughters"] = data["dirt"]["daughters"][dirt_eval]
            data["nu"]["daughters"] = data["nu"]["daughters"][nu_eval]

            if load_syst:
                dirt_eval_grouped = dirt_eval.groupby(self.grouper, sort=False).max()
                nu_eval_grouped = nu_eval.groupby(self.grouper, sort=False).max()
                print("Applying the master query on the systematic universes")
                for type_w in load_syst:
                    print(type_w)
                    data["nu"]["mc"][type_w] = data["nu"]["mc"][type_w][nu_eval_grouped]
                    data["dirt"]["mc"][type_w] = data["dirt"]["mc"][type_w][
                        dirt_eval_grouped
                    ]

        # data["dirt"]["daughters"]["category"] = 5 # Dirt is out of FV
        # data["dirt"]["daughters"]["cat_int"] = 7
        if not dirt:
            data["dirt"]["daughters"]["event_scale"] = 0

        return data

    # Get the purity of a selection
    def get_purity(self, selector, cats):
        weighted_selector = "(" + selector + ")*plot_weight"
        weights_denom = np.hstack(
            [
                self.mc_daughters.eval(weighted_selector),
                self.off_daughters.eval(weighted_selector),
            ]
        )
        weights_nom = []
        for cat in cats:
            weights_nom.append(
                np.array(
                    self.mc_daughters.query("category==@cat").eval(weighted_selector)
                )
            )
        weights_nom = np.hstack(weights_nom)
        purity, error_purity = helpfunction.effErr(weights_nom, weights_denom)

        return purity, error_purity

    def get_ratio_and_purity(self, query="", return_syst_err=False):
        mc_weight_arr = np.array(self.mc_daughters.query(query)["plot_weight"])
        off_weight_arr = np.array(self.off_daughters.query(query)["plot_weight"])
        on_weight_arr = np.array(self.on_daughters.query(query)["plot_weight"])
        mc_weights = np.sum(mc_weight_arr)
        off_weights = np.sum(off_weight_arr)
        on_weights = np.sum(on_weight_arr)
        ratio1 = (on_weights - off_weights) / mc_weights
        ratio1_err = np.sqrt(mc_weights + off_weights) / mc_weights
        ratio2 = on_weights / (mc_weights + off_weights)
        ratio = [ratio1, ratio2, ratio1_err]

        if len(self.syst_weights) > 0 & return_syst_err:
            err_data = np.sum(np.square(np.hstack([on_weight_arr, off_weight_arr])))
            cov = 0
            n_uni = 0
            mask = self.mc_daughters.eval(query).groupby(self.grouper, sort=False).sum()
            if max(mask) > 1:
                print("Systematics only supported for one row per event")
            else:
                mask = np.array(mask.astype(np.bool))
                for type_sys, weights in self.syst_weights.items():
                    n_uni += weights.shape[1]
                    n_syst_i = np.sum(weights[mask], axis=0)
                    cov += sum((n_syst_i - mc_weights) ** 2)
                cov /= n_uni
            err_mc = cov + np.sum(np.square(mc_weight_arr))
            err_ratio = ratio1 * np.sqrt(
                err_data / (on_weights - off_weights) ** 2 + err_mc / mc_weights ** 2
            )
            ratio[2] = err_ratio
        purity = self.get_purity(query, self.cats)
        return (
            ratio,
            purity,
        )

    def plot_panel_data_mc(
        self,
        ax,
        field,
        x_label,
        N_bins,
        x_min,
        x_max,
        query="",
        title_str="",
        legend=True,
        y_max_scaler=1.025,
        kind="cat",
        show_data=True,
        show_syst=True,
        syst_fractions=None,
        y_label="Events per bin",
        show_lee=None,
    ):

        """
        Plot a data/MC panel with the ratio.

        Arguments:
            ax {tuple} -- matplotlib axes, length should be 2
            field {string} -- string that can be given as an .eval() pandas command
            x_label {string} -- x-axis label
            N_bins -- number or bins
            x_min {float} -- the minimum number along x
            x_max {float} -- the maximum number along x
            query {string} -- string that can be given as a .query() pandas command
            title_str {string} -- right title string of the upper plot
            legend {bool} -- plot the legend on the right of the panel
            y_max_scaler {float} -- scale the y-axis of the plot
            kind {string} -- cat / pdg / int 

        Outputs:
            ratio -- [(on-off)/MC, on/(MC+off), (on-off)/MC Error]
            purity -- purity of the inclusive channel depending on the signal
            ks_test_p -- p-value of the KS-test
            dict -- Output of the plot {labels: string, bins: 1d array}
        """
        if show_lee is None:
            show_lee = self.show_lee
        if title_str is not "":
            title_str = "\n" + title_str

        plot_data = []
        weights = []
        labels = []
        colors = []
        bin_err = []

        if kind == "cat":
            kind_labs = self.dicts.category_labels
            kind_colors = self.dicts.category_colors
            column_check = "category"

        elif kind == "pdg":
            kind_labs = self.dicts.pdg_labels
            kind_colors = self.dicts.pdg_colors
            column_check = "backtracked_pdg"

        elif kind == "int":
            kind_labs = self.dicts.int_labels
            kind_colors = self.dicts.int_colors
            column_check = "cat_int"
        elif kind == "syst":
            kind_labs = self.dicts.category_labels
            kind_colors = self.dicts.category_colors
            column_check = "category"
            syst_grouper = []
        else:
            print("Unknown plotting type, please choose from int/pdg/cat")

        # MC contribution
        temp_view = self.mc_daughters.query(query)
        mc_data = temp_view.eval(field)
        mc_weights = temp_view["plot_weight"]

        for cat in kind_labs.keys():
            temp_view_cat = temp_view.query("abs({})==@cat".format(column_check))
            if len(temp_view_cat.index) > 0 and cat != 6:
                plot_data.append(temp_view_cat.eval(field).values)
                weights.append(temp_view_cat["plot_weight"].values)

                num_events = sum(weights[-1])
                precision = int(max(np.floor(np.log10(num_events)) + 1, 2))
                labels.append(
                    kind_labs[cat] + ": {:#.{prec}g}".format(num_events, prec=precision)
                )
                colors.append(kind_colors[cat])
                print("MC category:", labels[-1], "\t#entries", len(plot_data[-1]))
                if kind == "syst":
                    syst_grouper.append(self.dicts.syst_groups_cat[cat])

        if (self.signal == "nue") & show_lee:
            print("Showing LEE model")
            # LEE contribution
            plot_data.append(temp_view.query("leeweight>0.001").eval(field).values)
            weights.append(
                temp_view.query("leeweight>0.001").eval("leeweight*plot_weight").values
            )
            labels.append(r"$\nu_e$ LEE" + ": {0:#.2g}".format(sum(weights[-1])))
            colors.append(self.dicts.category_colors[111])
            if kind == "syst":
                syst_grouper.append(self.dicts.syst_groups_cat[111])

        # Off Contribution
        temp_view = self.off_daughters.query(query)
        plot_data.append(temp_view.eval(field).values)
        weights.append(temp_view["plot_weight"].values)
        num_events = sum(weights[-1])
        precision = int(max(np.floor(np.log10(num_events)) + 1, 2))
        labels.append("BNB Off" + ": {:#.{prec}g}".format(num_events, prec=precision))
        colors.append("grey")
        # On Contribution
        temp_view = self.on_daughters.query(query)
        plot_data.append(temp_view.eval(field).values)
        weights.append(temp_view["plot_weight"].values)
        labels.append("BNB On" + ": {0:0.0f}".format(sum(weights[-1])))
        colors.append("k")

        # Statistical tests
        if query in self.ratio_purity_dict:
            print("Obtained ratio and purity from dict")
            (ratio, purity) = self.ratio_purity_dict[query]
        else:
            print("Calculating ratio and purity")
            ratio, purity = self.get_ratio_and_purity(query, return_syst_err=True)
            self.ratio_purity_dict[query] = (ratio, purity)
        # KS-test
        flattened_MC = np.concatenate(plot_data[:-1]).ravel()
        flattened_weights = np.concatenate(weights[:-1]).ravel()
        ks_test_d, ks_test_p = kstest_weighted(
            flattened_MC, plot_data[-1], flattened_weights, weights[-1]
        )
        cnp = None

        # Start binning
        edges, edges_mid, bins, max_val = histHelper(
            N_bins, x_min, x_max, plot_data, weights=weights
        )
        beam_on_bins = bins[-1]

        for data_i, weight_i in zip(plot_data[:-2], weights[:-2]):
            bin_err.append(hist_bin_uncertainty(data_i, weight_i, x_min, x_max, edges))
        err_on = hist_bin_uncertainty(plot_data[-1], weights[-1], x_min, x_max, edges)
        err_off = hist_bin_uncertainty(plot_data[-2], weights[-2], x_min, x_max, edges)
        err_off[err_off == 0] = 0.4 * np.mean(
            weights[-2]
        )  # add-hoc increase the error on the off if no events
        err_mc = hist_bin_uncertainty(mc_data, mc_weights, x_min, x_max, edges)
        err_combined2 = err_off ** 2 + err_mc ** 2
        widths = edges_mid - edges[:-1]

        if show_data:
            # On
            ax[0].errorbar(
                edges_mid,
                beam_on_bins,
                xerr=widths,
                yerr=err_on,
                color=colors[-1],
                fmt=".",
                label=labels[-1],
            )
        # Off
        ax[0].bar(
            edges_mid,
            bins[-2],
            lw=2,
            label=labels[-2],
            width=2 * widths,
            color=colors[-2],
        )
        prediction = np.copy(bins[-2])
        if kind == "syst":
            for k, (lab_i, col_i) in self.dicts.sys_col_labels.items():
                bin_i = np.sum(np.array(bins[:-2])[np.array(syst_grouper) == k], axis=0)
                if sum(bin_i) > 0:
                    ax[0].bar(
                        edges_mid,
                        bin_i,
                        lw=2,
                        label=lab_i + ": {}".format(int(round(10 * sum(bin_i))) / 10),
                        width=2 * widths,
                        bottom=prediction,
                        color=col_i,
                    )
                    prediction += bin_i
            if syst_fractions:
                assert len(syst_fractions) == N_bins
                err_combined2 += (syst_fractions * prediction) ** 2
        else:
            for bin_i, lab_i, col_i in zip(bins[:-2], labels[:-2], colors[:-2]):
                ax[0].bar(
                    edges_mid,
                    bin_i,
                    lw=2,
                    label=lab_i,
                    width=2 * widths,
                    bottom=prediction,
                    color=col_i,
                )
                prediction += bin_i

        if show_syst:
            cov = self.get_cov(edges, query, mc_data, mc_weights)
            err_combined2 += np.diag(cov)  # used for the error on prediction
            # cov matrix used for the chi2_CNP
            err_stat_cnp = np.where(
                (beam_on_bins > 0) & (prediction > 0),
                3 / (1 / beam_on_bins + 2 / prediction),
                prediction / 2 + beam_on_bins,
            )
            cov[np.diag_indices_from(cov)] += err_combined2 + err_stat_cnp
            diff = beam_on_bins - prediction
            chisq = diff.dot(np.linalg.inv(cov)).dot(diff.T)
            chisq_p = 1 - scipy.stats.chi2.cdf(chisq, N_bins)
            cnp = (chisq, chisq_p, N_bins)

        err_combined = np.sqrt(err_combined2)
        for m, v, e, w in zip(edges_mid, prediction, err_combined, widths):
            ax[0].add_patch(
                patches.Rectangle(
                    (m - w, v - e),
                    2 * w,
                    2 * e,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )
            sc_err = e / v
            ax[1].add_patch(
                patches.Rectangle(
                    (m - w, 1 - sc_err),
                    2 * w,
                    sc_err * 2,
                    hatch="\\\\\\\\\\",
                    Fill=False,
                    linewidth=0,
                    alpha=0.4,
                )
            )

        ax[0].set_ylabel(y_label)
        ax[0].set_title(self.title_str + title_str, loc="left")
        ax[0].set_ylim(
            0,
            y_max_scaler
            * max(max_val[-1] + np.sqrt(max_val[-1]), max(prediction + err_combined)),
        )
        ax[0].set_xlim(x_min, x_max)

        # Ratio plots
        y_min_r = max(0, min((beam_on_bins - err_on) / prediction) * 0.9)
        y_max_r = min(2, max((beam_on_bins + err_on) / prediction) * 1.1)
        ax[1].set_ylim(y_min_r, y_max_r)
        ax[1].set_xlim(x_min, x_max)
        if show_data:
            ax[1].errorbar(
                edges_mid,
                beam_on_bins / prediction,
                xerr=widths,
                yerr=err_on / prediction,
                alpha=1.0,
                color="k",
                fmt=".",
            )
        ax[1].set_ylabel(r"$\frac{BNB\ On}{BNB\ Off\,+\,MC}$")
        ax[1].set_xlabel(x_label)

        if legend:
            ax[0].legend(bbox_to_anchor=(1.02, 0.5), loc="center left")
        best_text_loc = get_best_text_loc(prediction, N_bins)

        return ratio, purity, ks_test_p, cnp, best_text_loc

    def get_cov(self, bin_edges, query, cv_data, cv_weights):
        mask = self.mc_daughters.eval(query).groupby(self.grouper, sort=False).sum()
        N_bins = len(bin_edges) - 1
        cov = np.zeros([N_bins, N_bins])
        n_cv, _ = np.histogram(cv_data, bins=bin_edges, weights=cv_weights)

        if max(mask) > 1:
            print(
                "Covariance matrices supported for one daughter per event, max found: {}".format(
                    max(mask)
                )
            )
        else:
            mask = mask.astype(np.bool)
            for type_sys, weights in self.syst_weights.items():
                n_uni = min(weights.shape[1], self.n_uni_max)
                # Method 1
                start = time.time()
                cov_this = np.zeros([N_bins, N_bins])
                for i in range(n_uni):
                    n, _ = np.histogram(
                        cv_data, weights=weights[mask].T[i], bins=bin_edges,
                    )

                    cov_this += np.outer(n - n_cv, n - n_cv)
                cov += cov_this / n_uni
                # Medthod 2 - Parallel loop, actually slower :(
                # mid = time.time()
                # cov_this_pool = sum(Parallel(n_jobs=4)(delayed(cov_universe)(weights[mask].T[i], cv_data, n_cv, bin_edges) for i in range(n_uni)))
                # end = time.time()
                # print('does parrallellisation work?', np.allclose(cov_this,cov_this_pool))
                # print('Serial time:',mid-start,'Parallel time', end-mid)
        return cov


def cov_universe(weights, cv_data, n_cv, bin_edges):
    n, _ = np.histogram(cv_data, weights=weights, bins=bin_edges,)
    return np.outer(n - n_cv, n - n_cv)


def efficiency(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    """
    Calculate the efficiency given two populations: one containg 
    the totatility of the events,and one containing only events 
    that pass the selection.
    It uses a frequentist approach to evaluate the uncertainty.
    Other methods are to be implemented.
    
     Arguments:
        num {tuple} -- The totality of the events
        den {tuple} -- The events that pass the selection
        num_w {tuple} -- Optional, the weight for every event
        den_w {tuple} -- Optional, the weight for every selected event
        n_bins {int} -- Optional, the number of bins
        x_min {float} -- Optional, the minimum number along x
        x_max {float} -- Optional, the maximum number along x
        conf_level {float} -- Optional, the confidence level to be used
        
    Outputs:
        eff {tuple} -- The efficiency per bin
        unc_low {tuple} -- The lower uncertainty per bin
        unc_up {tuple} -- The upper uncertainty per bi
        bins {tuple} -- The bin edges
    """

    if num_w is None:
        num_w = [1.0] * len(num)

    if den_w is None:
        den_w = [1.0] * len(den)

    if conf_level is None:
        conf_level = 0.682689492137

    num = np.asarray(num)
    num_w = np.asarray(num_w)
    den = np.asarray(den)
    den_w = np.asarray(den_w)

    bins = np.linspace(x_min, x_max, n_bins)

    num_h, _ = np.histogram(num, bins=bins)
    num_w_h, _ = np.histogram(num, weights=num_w, bins=bins)
    num_w2_h, _ = np.histogram(num, weights=num_w ** 2, bins=bins)

    den_h, _ = np.histogram(den, bins=bins)
    den_w_h, _ = np.histogram(den, weights=den_w, bins=bins)
    den_w2_h, _ = np.histogram(den, weights=den_w ** 2, bins=bins)

    eff = num_w_h / den_w_h

    variance = (num_w2_h * (1.0 - 2 * eff) + den_w2_h * eff * eff) / (den_w_h * den_w_h)
    sigma = np.sqrt(variance)
    prob = 0.5 * (1.0 - conf_level)
    delta = -scipy.stats.norm.ppf(prob) * sigma

    unc_up = []
    unc_low = []

    for eff_i, delta_i in zip(eff, delta):
        if eff_i - delta_i < 0:
            unc_low.append(eff_i)
        else:
            unc_low.append(delta_i)

        if eff_i + delta_i > 1:
            unc_up.append(1.0 - eff_i)
        else:
            unc_up.append(delta_i)

    return eff, unc_low, unc_up, bins


# Helper class duplicating the last bin, useful to use in combination with matplotlib step function.
def efficiency_post(
    num, den, num_w=None, den_w=None, n_bins=10, x_min=0, x_max=10, conf_level=None
):
    eff, unc_low, unc_up, edges = efficiency(
        num, den, num_w, den_w, n_bins, x_min, x_max, conf_level
    )
    eff = np.append(eff, eff[-1])
    unc_low = np.append(unc_low, unc_low[-1])
    unc_up = np.append(unc_up, unc_up[-1])
    return eff, unc_low, unc_up, edges


def hist_bin_uncertainty(data, weights, x_min, x_max, bin_edges):
    """
        Calculate the error on the bins in the histogram including the weights.

        Arguments:
            edges {1d tuple} -- The bin edges
            data {1d tuple} -- array with the data of a variable
            weigths {1d tuple} -- weights, same shape as data
            
        Outputs:
            bin_uncertainties {1d tuple} -- Uncertainty on each bin
        """
    # Bound the data and weights to be within the bin edges
    mask_in_range = (data > x_min) & (data < x_max)
    in_range_data = data[mask_in_range]
    in_range_weights = weights[mask_in_range]

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [
            in_range_weights[np.where(bin_index == idx)[0]]
            for idx in range(1, len(bin_edges))
        ]
    )
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights]
    )
    return bin_uncertainties


def kstest_weighted(data1, data2, wei1, wei2):
    """
    2-sample KS test unbinned probability.
    Takes into account the weight of the events.
    stackoverflow.com/questions/40044375/
    how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples/40059727
    
    Arguments:
        data1 {1d tuple} -- array with the data of a variable
        wei1 {1d tuple} -- weights, same shape as data
        data2 {1d tuple} -- array with the data of a variable
        wei2 {1d tuple} -- weights, same shape as data

    Outputs:
        d -- KS-text max separation
        prob -- KS-test p-value
    """
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side="right")]
    cdf2we = cwei2[np.searchsorted(data2, data, side="right")]
    d = np.max(np.abs(cdf1we - cdf2we))
    # Note: d absolute not signed distance
    n1 = sum(wei1)
    n2 = sum(wei2)
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    prob = scipy.stats.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    return d, prob


def histHelper(n_bins, x_min, x_max, data, weights=0, where="mid", log=False):
    """
    Wrapper around the numpy histogram function.

    Arguments:
        n_bins {int} -- the number of bins
        x_min {float} -- the minimum number along x
        x_max {float} -- the maximum number along x
        data {2d tuple} -- array or list of arrays
        weigths {2d tuple} -- same shape as data or 0 if all weights are equal
        where {string} -- if where='post': duplicate the last bin
        log {bool} -- log==True: return x-axis log

    Outputs:
        edges {1d tuple} -- The bin edges
        edges_mid {1d tuple} -- The middle of the bins
        bins {2d tuple} -- The bin values, same depth as data
        max_val {1d tuple}-- the maximum value, same depth as data
    """
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    else:
        edges = np.linspace(x_min, x_max, n_bins + 1)
    edges_mid = [edges[i] + (edges[i + 1] - edges[i]) / 2 for i in range(n_bins)]
    if weights == 0:
        weights = [[1] * len(d) for d in data]

    bins = [
        np.histogram(data_i, bins=edges, weights=weights_i)[0]
        for data_i, weights_i in zip(data, weights)
    ]
    max_val = [max(x) for x in bins]
    if where == "post":
        bins = [np.append(b, b[-1]) for b in bins]

    return edges, edges_mid, bins, max_val


def add_text(ax, which, locator, y=1):
    y_pos = (1 - 0.08 * sum(x is not None for x in which)) * y
    text_str = []
    if which[0] is not None:
        ratio = which[0]
        text_str.append(r"(On-Off)/MC: {:.2f}$\pm${:.2f}".format(ratio[0], ratio[2]))
    if which[3] is not None:
        (cnp_chi, cnp_p, dof) = which[3]
        text_str.append(
            "$\chi^2$/dof: {:.1f}/{}, p: {:.2f}".format(cnp_chi, dof, cnp_p)
        )
    if which[1] is not None:
        purity = which[1]
        text_str.append(r"$\nu_e$" + " CC purity: {:<3.1%}".format(purity[0]))
    if which[2] is not None:
        ks_p = which[2]
        text_str.append("KS p-value: {:<5.2f}".format(ks_p))

    ax.text(
        (ax.get_xlim()[1] - ax.get_xlim()[0]) * text_dict[locator][1]
        + ax.get_xlim()[0],
        (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_pos + ax.get_ylim()[0],
        "\n".join(text_str),
        horizontalalignment=text_dict[locator][0],
        fontsize="medium",
    )


def get_best_text_loc(prediction, N_bins):
    # if we want to write on the figure, left, middle or right?
    bin_count = int(N_bins / 2)
    left = prediction[0:bin_count].sum()
    middle = prediction[int(N_bins / 4) : int(N_bins / 4) + bin_count].sum()
    right = prediction[-(bin_count + 1) : -1].sum()
    return np.argmin([left, middle, right])
