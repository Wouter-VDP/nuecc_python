# ### Input ###

import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
import time
import os

import helpfunction as helper
import nue_selection_columns as columns


# ### Constants ###

# models
model_dir = "./models/"
# correction in the spacecharge
x_sce_magic = 1.03
# unique identifier for groupby operations
grouper = ["sample", "Run", "event"]
# POT used in the loader on the gpvm.
pot_target = 1e21


# ### Definitions ###

# Manipulate the input frames and perform the selection
def SelectNues(sample, data):
    if sample in helper.data_samples:
        data["daughters"]["optical_filter"] = True
    else:
        ConvertWeights(sample, data["mc"])
        if sample == "dirt":
            FixDirtWeights(data["mc"], sum(data["pot"].values()))

        AddSimulatedFields(sample, data)

    if sample == "sideband":
        FixSidebandRuns(data["daughters"])

    AddRecoFields(data["daughters"])
    AddOtherDaughterAngleDiff(data["daughters"])
    PerformClassification(data["daughters"])

    pass_rate = data["daughters"]["select"].sum() / data["numentries"]
    return "Selected {:d}/{:d} events, {:.3%}".format(
        data["daughters"]["select"].sum(), data["numentries"], pass_rate
    )

# Convert the systematic weights from jagged arrays to numpy float16 matrices.
def ConvertWeights(sample, mc_data):
    for w in helper.syst_weights:
        mc_data[w] = np.clip(
            np.nan_to_num(
                mc_data[w][mc_data[w].counts > 0].regular().astype("float16"),
                nan=1,
                posinf=1,
                neginf=1,
            ),
            0,
            100,
        )


# Fix the Run period for the sideband sample
def FixSidebandRuns(daughters):
    run2_a = 8316
    run2_b = 10000
    run3 = 13696
    daughters.eval(
        "Run = 1 + (run>@run2_a) + 0.5*(run>@run2_b) + 0.5*(run>@run3)", inplace=True
    )


def FixDirtWeights(mc_data, mc_pot):
    mc_data["event_scale"] = np.full(len(mc_data["Run"]), pot_target / mc_pot)


def AddRecoFields(daughters):
    # calibration of shower energy.
    daughters["shr_energy_y_v"] /= 0.83
    daughters["trk_energy_proton_v"] *= 1000
    daughters["shr_energy_y_v"].clip(0, 9999, inplace=True)
    daughters["trk_energy_proton_v"].clip(0, 9999, inplace=True)

    # Add fiducial reco sce vtx
    daughters["reco_fid_vol"] = np.repeat(
        helper.is_fid(
            *daughters[["reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z"]]
            .xs(0, level="daughter")
            .values.T
        ),
        daughters["n_pfps"].xs(0, level="daughter"),
    )

    # Add pfp at vtx and tracks at vtx:
    daughters.eval("n_pfpvtx = trk_distance_v<3 & trk_distance_v>=0", inplace=True)
    daughters["n_pfpvtx"] = (
        daughters["n_pfpvtx"].groupby(grouper, sort=False).transform(sum)
    )
    daughters.eval("n_pfp_farvtx = n_pfps-n_pfpvtx", inplace=True)
    daughters.eval(
        "trk_at_vtx = trk_distance_v<3 & trk_distance_v>=0 & trk_score_v>0.3",
        inplace=True,
    )
    daughters["trk_at_vtx"] = (
        daughters["trk_at_vtx"].groupby(grouper, sort=False).transform(sum)
    )

    # Number clusters and subclusters in the particle:
    daughters["pfp_clusters_v"] = (
        daughters[
            ["pfpplanesubclusters_U", "pfpplanesubclusters_V", "pfpplanesubclusters_Y"]
        ]
        .astype(bool)
        .sum(axis=1)
    )
    daughters.eval(
        "shr_subclusters_v = (pfpplanesubclusters_U+pfpplanesubclusters_V+pfpplanesubclusters_Y)",
        inplace=True,
    )

    e_cand_bool = daughters.eval(helper.e_cand_str)
    e_cand_maxe = (
        daughters[e_cand_bool]["shr_energy_y_v"]
        .groupby(grouper, sort=False)
        .transform(max)
        == daughters[e_cand_bool]["shr_energy_y_v"]
    )
    daughters["e_candidate"] = False
    daughters.loc[e_cand_maxe[e_cand_maxe == True].index, "e_candidate"] = True

    # Add weighted dedx:
    ## clip to reasonable values
    for col in columns.dedx_cols:
        daughters[col] = np.clip(
            np.nan_to_num(daughters[col], nan=0, posinf=20, neginf=0), 0, 20
        )
    ## for the first 4 cm
    str_dedx_weighted_mean = "(shr_tkfit_dedx_u_v*shr_tkfit_dedx_nhits_u_v+shr_tkfit_dedx_v_v*shr_tkfit_dedx_nhits_v_v+shr_tkfit_dedx_y_v*shr_tkfit_dedx_nhits_y_v)/(shr_tkfit_dedx_nhits_u_v+shr_tkfit_dedx_nhits_v_v+shr_tkfit_dedx_nhits_y_v)"
    daughters["shr_tkfit_4cm_dedx_wm_v"] = daughters.eval(str_dedx_weighted_mean)

    # Add the number of hits per length:
    daughters.eval("hits_per_tklen_v = pfnhits/trk_len_v", inplace=True)
    # Add the sum of shower subclusters:
    daughters.eval("shr_tkfit_hitratio_v = shr_tkfit_nhits_v/pfnhits", inplace=True)
    # MCS muon momentum consistency:
    daughters.eval(
        "trk_muon_hypothesis_ratio_v = trk_mcs_muon_mom_v/trk_range_muon_mom_v",
        inplace=True,
    )
    # Proton track energy consistency:
    daughters.eval(
        "trk_proton_hypothesis_ratio_v = trk_calo_energy_y_v/trk_energy_proton_v",
        inplace=True,
    )

    this_pop_cols = [s for s in columns.pop_cols if s in daughters.keys()]
    daughters.drop(this_pop_cols, axis=1, inplace=True)

    # Perform the preselection
    daughters["preselect"] = (
        daughters.eval(helper.query_preselect)
        .groupby(grouper, sort=False)
        .transform(max)
    )


def AddOtherDaughterAngleDiff(daughters):
    e_pre_temp = daughters.query("e_candidate & preselect")
    pre_temp = daughters.query("~e_candidate & preselect")
    dir_e_x = np.repeat(
        e_pre_temp.eval("trk_sce_end_x_v-trk_sce_start_x_v"), e_pre_temp["n_pfps"] - 1
    ).values
    dir_e_y = np.repeat(
        e_pre_temp.eval("trk_sce_end_y_v-trk_sce_start_y_v"), e_pre_temp["n_pfps"] - 1
    ).values
    dir_e_z = np.repeat(
        e_pre_temp.eval("trk_sce_end_z_v-trk_sce_start_z_v"), e_pre_temp["n_pfps"] - 1
    ).values
    dir_d_x = pre_temp.eval("trk_sce_end_x_v-trk_sce_start_x_v").values
    dir_d_y = pre_temp.eval("trk_sce_end_y_v-trk_sce_start_y_v").values
    dir_d_z = pre_temp.eval("trk_sce_end_z_v-trk_sce_start_z_v").values

    e_vec = np.array([dir_e_x, dir_e_y, dir_e_z]).T
    d_vec = np.array([dir_d_x, dir_d_y, dir_d_z]).T
    cos_sim = (dir_e_x * dir_d_x + dir_e_y * dir_d_y + dir_e_z * dir_d_z) / (
        np.linalg.norm(d_vec, axis=1) * np.linalg.norm(e_vec, axis=1)
    )
    daughters["e_candidate_anglediff"] = 0
    daughters.loc[pre_temp.index, "e_candidate_anglediff"] = cos_sim


def PerformClassification(daughters):
    # Clean up training data
    for col in (
        columns.col_train_electron + columns.col_train_other + columns.col_train_event
    ):
        if col in daughters.keys():
            daughters[col] = np.clip(
                np.nan_to_num(daughters[col], nan=-5, posinf=1000, neginf=-100),
                -100,
                1000,
            )
    # Load the pre-trained models
    model_e = load(model_dir + "model_e.pckl")
    model_d = load(model_dir + "model_d.pckl")
    model_event = load(model_dir + "model_event.pckl")
    # Predict the daughter classification
    daughters["score"] = -1
    mask_e_cand = daughters.eval("preselect & e_candidate")
    daughters.loc[mask_e_cand, "score"] = model_e.predict_proba(
        daughters[columns.col_train_electron][mask_e_cand]
    ).T[1]
    mask_d = daughters.eval("preselect & ~e_candidate")
    daughters.loc[mask_d, "score"] = model_d.predict_proba(
        daughters[columns.col_train_other][mask_d]
    ).T[1]
    # Prepare for event training
    mask_e = daughters.eval("preselect & e_candidate & n_pfps>1")
    daughters["score_other_max"] = 1

    daughters.loc[mask_e, "score_other_max"] = (
        daughters.query("~e_candidate & preselect")["score"]
        .groupby(grouper, sort=False)
        .max()
        .values
    )
    daughters["score_other_mean"] = 1
    daughters.loc[mask_e, "score_other_mean"] = (
        daughters.query("~e_candidate & preselect")["score"]
        .groupby(grouper, sort=False)
        .mean()
        .values
    )
    daughters["score_other_min"] = 1
    daughters.loc[mask_e, "score_other_min"] = (
        daughters.query("~e_candidate & preselect")["score"]
        .groupby(grouper, sort=False)
        .min()
        .values
    )
    # Predict the event classification
    daughters["score_event"] = -1
    mask_e_cand = daughters.eval("preselect & e_candidate")
    daughters.loc[mask_e_cand, "score_event"] = model_event.predict_proba(
        daughters[columns.col_train_event][mask_e_cand]
    ).T[1]
    # Final selection
    cut_val = 0.87  # 0.805
    query_select = "e_candidate & preselect & score_event>@cut_val"
    daughters["select"] = daughters.eval(query_select)


def AddSimulatedFields(k, v):
    # Add distance between reco_sce and true vertex
    true_vtx = [
        v["mc"][f][v["mc"]["n_pfps"] > 0]
        for f in ["true_nu_vtx_x", "true_nu_vtx_y", "true_nu_vtx_z"]
    ]
    reco_vtx = (
        v["daughters"][["reco_nu_vtx_sce_x", "reco_nu_vtx_sce_y", "reco_nu_vtx_sce_z"]]
        .xs(0, level="daughter")
        .values.T
    )
    reco_vtx[0] -= x_sce_magic  # Correct x location
    v["daughters"]["true_vtx_distance"] = np.repeat(
        np.linalg.norm(true_vtx - reco_vtx, axis=0),
        v["mc"]["n_pfps"][v["mc"]["n_pfps"] > 0],
    )

    # Add the modified purity/completeness to account for overlay.
    overlay_mask = v["daughters"].eval("backtracked_overlay_purity>backtracked_purity")
    v["daughters"].loc[overlay_mask, "backtracked_pdg"] = 0
    v["daughters"].loc[overlay_mask, "backtracked_purity"] = v["daughters"].loc[
        overlay_mask, "backtracked_overlay_purity"
    ]
    v["daughters"].loc[overlay_mask, "backtracked_completeness"] = 0

    # Pi0 scaling
    pi0_max_e = 0.6
    v["mc"]["weightSplineTimesTune_pi0scaled"] = v["mc"]["weightSplineTimesTune"] * (
        1 - 0.4 * v["mc"]["mc_E"][v["mc"]["mc_pdg"] == 111].max().clip(0, pi0_max_e)
    )

    ## add truth fields:
    for true_f in columns.add_mc_fields:
        if true_f in v["mc"]:
            v["daughters"][true_f] = np.repeat(v["mc"][true_f], v["mc"]["n_pfps"])
        else:
            print("Truth field {} is not in sample {}".format(true_f, k))

    # add true fiducial colume:
    v["daughters"]["true_fid_vol"] = np.repeat(
        helper.is_fid(
            v["mc"]["true_nu_vtx_x"],
            v["mc"]["true_nu_vtx_y"],
            v["mc"]["true_nu_vtx_z"],
        ),
        v["mc"]["n_pfps"],
    )


def CreateAfterTraining(plot_samples, input_dir, one_file=True):
    available_samples = os.listdir(input_dir)
    all_samples = {}

    for sample in plot_samples:
        start_time = time.time()
        sample_file = min([f for f in available_samples if sample in f], key=len)
        data = pd.read_pickle(input_dir + sample_file)

        # data is passed by reference and not copied
        sel_str = SelectNues(sample, data)

        if not one_file:
            pickle_out = open(
                "{}lite/{}_after_training.pckl".format(input_dir, sample), "wb"
            )
            pickle.dump(data, pickle_out)
            pickle_out.close()
        else:
            all_samples[sample] = data
        print(
            "Finished {} ({}), took {:0.1f} seconds.".format(
                sample, sample_file, time.time() - start_time
            )
        )
        print(sel_str + "\n")
    if one_file:
        pickle_out = open("{}lite/after_training.pckl".format(input_dir), "wb")
        pickle.dump(all_samples, pickle_out)
        pickle_out.close()



# ### Load Samples
plot_samples = {"dirt", "sideband", "on", "nu", "off"}
input_dir = "./input/Jun2020/"
CreateAfterTraining(plot_samples, input_dir, one_file=True)
