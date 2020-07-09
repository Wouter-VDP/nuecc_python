# ### Input ###

import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
import time
import os

from helpers import helpfunction as helper
from helpers import numu_selection_columns as columns


# ### Constants ###

# models
# model_dir = "./models/"
# correction in the spacecharge
x_sce_magic = 1.03
# unique identifier for groupby operations
grouper = ["sample", "Run", "event"]
# POT used in the loader on the gpvm.
pot_target = 1e21
# Final selection BDT cut
# cut_val = 0.87  # 0.805


# ### Definitions ###

# Manipulate the input frames and perform the selection
def SelectNumus(sample, data):
    if sample in helper.data_samples:
        data["daughters"]["optical_filter"] = True
    else:
        if sample == "dirt":
            FixDirtWeights(data["mc"], sum(data["pot"].values()))
        ConvertWeights(sample, data["mc"])
        AddSimulatedFields(sample, data)
        AddNumuCategories(data["daughters"])

    if sample == "sideband":
        FixSidebandRuns(data["daughters"])

    AddRecoFields(data["daughters"])

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
    run2 = 8316
    run3 = 13696
    daughters.eval("Run = 1 + (run>@run2) + (run>@run3)", inplace=True)

# Weight the dirt sample in the same way as the other mc samples
def FixDirtWeights(mc_data, mc_pot):
    mc_data["event_scale"] = np.full(len(mc_data["Run"]), pot_target / mc_pot)

# Add additional fields needed for the selection
def AddRecoFields(daughters):
    # calibration of shower energy.
    daughters["trk_energy_proton_v"] *= 1000
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
    
    # Add containment fields
    daughters["all_start_contained"] = helper.is_fid(
        daughters["trk_sce_start_x_v"],
        daughters["trk_sce_start_y_v"],
        daughters["trk_sce_start_z_v"],
    )
    daughters["all_start_contained"] = (
        daughters["all_start_contained"].groupby(grouper).transform(min)
    )
    daughters["all_end_contained"] = helper.is_fid(
        daughters["trk_sce_end_x_v"],
        daughters["trk_sce_end_y_v"],
        daughters["trk_sce_end_z_v"],
    )
    daughters["all_end_contained"] = (
        daughters["all_end_contained"].groupby(grouper).transform(min)
    )

    # Add pfp at vtx and tracks at vtx:
    daughters.eval(
        "is_trk_at_vtx = trk_distance_v<3 & trk_distance_v>=0 & trk_score_v>0.3",
        inplace=True,
    )
    daughters["trk_at_vtx"] = (
        daughters["is_trk_at_vtx"].groupby(grouper, sort=False).transform(sum)
    )
    
    q_ev_considered = 'reco_fid_vol &\
                       topological_score>0.06 &\
                       all_start_contained &\
                       all_end_contained'
    daughters["numu_ev_considered"] = daughters.eval(q_ev_considered)
    
    q_muon_candidate_no_pid = "(is_trk_at_vtx &\
                               trk_score_v>0.8 &\
                               trk_len_v>10 &\
                               pfp_generation_v==2)"
    q_muon_candidate = q_muon_candidate_no_pid + " & (trk_llr_pid_score_v>0.2)"
    
    muon_candidate_no_pid = daughters.eval(q_muon_candidate_no_pid)
    muon_cand_no_pid_maxll = (
        daughters[muon_candidate_no_pid]["trk_llr_pid_score_v"]
        .groupby(grouper)
        .transform(max)
        == daughters[muon_candidate_no_pid]["trk_llr_pid_score_v"]
    )
    daughters["muon_candidate_no_pid"] = False
    daughters.loc[
        muon_cand_no_pid_maxll[muon_cand_no_pid_maxll == True].index, "muon_candidate_no_pid"
    ] = True
    
    muon_candidate = daughters.eval(q_muon_candidate)
    daughters["n_mu_cand"] = muon_candidate.groupby(grouper).transform(sum)
    muon_cand_maxll = (
        daughters[muon_candidate]["trk_llr_pid_score_v"]
        .groupby(grouper)
        .transform(max)
        == daughters[muon_candidate]["trk_llr_pid_score_v"]
    )
    daughters["muon_candidate"] = False
    daughters.loc[
        muon_cand_maxll[muon_cand_maxll == True].index, "muon_candidate"
    ] = True
    
    q_ev_selected = q_ev_considered + " & (n_mu_cand > 0)"
    daughters["numu_ev_selected"] = daughters.eval(q_ev_selected)
    
    daughters["non_muon_cand_maxll"] = False
    non_muon_cand_maxll = (
        daughters.query("~muon_candidate & is_trk_at_vtx")["trk_llr_pid_score_v"]
        .groupby(grouper)
        .transform(max)
        == daughters.query("~muon_candidate & is_trk_at_vtx")["trk_llr_pid_score_v"]
    )
    daughters.loc[
        non_muon_cand_maxll[non_muon_cand_maxll == True].index, "non_muon_cand_maxll"
    ] = True
    
    daughters["non_muon_cand_minll"] = False
    non_muon_cand_minll = (
        daughters.query("~muon_candidate & is_trk_at_vtx")["trk_llr_pid_score_v"]
        .groupby(grouper)
        .transform(min)
        == daughters.query("~muon_candidate & is_trk_at_vtx")["trk_llr_pid_score_v"]
    )
    daughters.loc[
        non_muon_cand_minll[non_muon_cand_minll == True].index, "non_muon_cand_minll"
    ] = True
    

    this_pop_cols = [s for s in columns.pop_cols if s in daughters.keys()]
    daughters.drop(this_pop_cols, axis=1, inplace=True)

# Add truth based fields to the daughter dataframe
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

# Add categories used for numu plotting
def AddNumuCategories(daughters):
    q_30 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==14 & nproton==1 & (npi0+npion)==0"
    q_31 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==14 & nproton==2 & (npi0+npion)==0"
    q_32 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==14 & nproton==0 & npi0==0 & npion==1"
    q_33 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==14 & nproton==1 & npi0==0 & npion==1"
    q_34 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==14 & (~(nproton==1 & (npi0+npion)==0) &\
                                              ~(nproton==2 & (npi0+npion)==0) &\
                                              ~(nproton==0 & npi0==0 & npion==1) &\
                                              ~(nproton==1 & npi0==0 & npion==1))"
    q_35 = "true_fid_vol & ccnc==1"
    q_10 = "true_fid_vol & ccnc==0 & abs(nu_pdg)==12"    
    
    q_5 = "true_fid_vol==0"

    daughters["category"] = (
        daughters.eval(q_30) * 30
        + daughters.eval(q_31) * 31
        + daughters.eval(q_32) * 32
        + daughters.eval(q_33) * 33
        + daughters.eval(q_34) * 34
        + daughters.eval(q_35) * 35
        + daughters.eval(q_10) * 10
        + daughters.eval(q_5) * 5
    )
    cosmic = (daughters["nu_purity_from_pfp"] < 0.5) & (daughters["category"] != 5)
    daughters.loc[cosmic, "category"] = 4
#     daughters["category"][cosmic] = 4

# Generate the pckl file used by the plotter
def CreateAfterTraining(plot_samples, input_dir, one_file=True):
    available_samples = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
    ]
    all_samples = {}
    print(available_samples)

    for sample in plot_samples:
        start_time = time.time()
        sample_file = min([f for f in available_samples if sample in f], key=len)
        data = pd.read_pickle(input_dir + sample_file)

        # data is passed by reference and not copied
        sel_str = SelectNumus(sample, data)

        if not one_file:
            pickle_out = open(
                "{}/{}_after_training.pckl".format(input_dir, sample), "wb"
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
#         print(sel_str + "\n")
    if one_file:
        pickle_out = open("{}/after_reducing.pckl".format(input_dir), "wb")
        pickle.dump(all_samples, pickle_out)
        pickle_out.close()