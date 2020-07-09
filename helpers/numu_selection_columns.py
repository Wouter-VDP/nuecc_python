# Columns to remove:
pop_cols = {
    "shr_tkfit_dedx_u_v",
    "shr_tkfit_dedx_v_v",
    "shr_tkfit_dedx_y_v",
    "shr_tkfit_dedx_nhits_u_v",
    "shr_tkfit_dedx_nhits_v_v",
    "shr_tkfit_dedx_nhits_y_v",
    "shr_tkfit_2cm_dedx_Y",
    "shr_tkfit_2cm_nhits_Y",
    "shr_tkfit_gap10_dedx_y_v",
    "shr_tkfit_2cm_dedx_V",
    "shr_tkfit_2cm_dedx_U",
    "shr_tkfit_2cm_nhits_V",
    "shr_tkfit_2cm_nhits_U",
    "shr_tkfit_dedx_v_v",
    "shr_tkfit_dedx_u_v",
    "shr_tkfit_dedx_nhits_u_v",
    "shr_tkfit_dedx_nhits_v_v",
    "slclustfrac",
    "shrclusfrac2",
    "shrclusdir2",
    "trkshrhitdist2",
    "pfpplanesubclusters_U",
    "pfpplanesubclusters_V",
    "pfpplanesubclusters_Y",
    "trk_range_muon_mom_v",
    "trk_mcs_muon_mom_v",
    "trk_calo_energy_y_v",
    "trk_energy_proton_v",
    "nu_flashmatch_score",
    "_closestNuCosmicDist",
    "crtveto",
    "crthitpe",
    "CosmicIP",
    "secondshower_Y_vtxdist",
    "secondshower_Y_eigenratio",
    "secondshower_Y_dir",
    "secondshower_Y_dot",
    
    "trk_start_x_v",
    "trk_start_y_v",
    "trk_start_z_v",

    "trk_sce_start_x_v",
    "trk_sce_start_y_v",
    "trk_sce_start_z_v",

    "trk_distance_v",

    "trk_theta_v",
    "trk_phi_v",

    "trk_dir_x_v",
    "trk_dir_y_v",
    "trk_dir_z_v",

    "trk_end_x_v",
    "trk_end_y_v",
    "trk_end_z_v",

    "trk_sce_end_x_v",
    "trk_sce_end_y_v",
    "trk_sce_end_z_v",

    "trk_len_v",

    "trk_bragg_p_v",
    "trk_bragg_mu_v",
    "trk_bragg_mip_v",
    "trk_pid_chipr_v",
    "trk_pid_chika_v",
    "trk_pid_chipi_v",
    "trk_pid_chimu_v",
    "trk_pida_v",

    "trk_bragg_p_u_v",
    "trk_bragg_mu_u_v",
    "trk_bragg_mip_u_v",
    "trk_pid_chipr_u_v",
    "trk_pid_chika_u_v",
    "trk_pid_chipi_u_v",
    "trk_pid_chimu_u_v",
    "trk_pida_u_v",

    "trk_bragg_p_v_v",
    "trk_bragg_mu_v_v",
    "trk_bragg_mip_v_v",
    "trk_pid_chipr_v_v",
    "trk_pid_chika_v_v",
    "trk_pid_chipi_v_v",
    "trk_pid_chimu_v_v",
    "trk_pida_v_v",
    
    
    "shr_energy_u_v",
    "shr_energy_v_v",
    "shr_energy_y_v",

    "shr_dedx_u_v",
    "shr_dedx_v_v",
    "shr_dedx_y_v",

    "shr_pfp_id_v",

    "shr_start_x_v",
    "shr_start_y_v",
    "shr_start_z_v",

    "shr_start_U_v",
    "shr_start_V_v",
    "shr_dist_v",

    "shr_px_v",
    "shr_py_v",
    "shr_pz_v",

    "shr_theta_v",
    "shr_phi_v",

    "shr_pitch_u_v",
    "shr_pitch_v_v",
    "shr_pitch_y_v",

    "shr_openangle_v",

    "shr_tkfit_nhits_v",
    "shr_tkfit_start_x_v",
    "shr_tkfit_start_y_v",
    "shr_tkfit_start_z_v",

    "shr_tkfit_start_U_v",
    "shr_tkfit_start_V_v",

    "shr_tkfit_theta_v",
    "shr_tkfit_phi_v",

    "shr_tkfit_pitch_u_v",
    "shr_tkfit_pitch_v_v",
    "shr_tkfit_pitch_y_v",

    "shr_tkfit_dedx_u_v",
    "shr_tkfit_dedx_v_v",
    "shr_tkfit_dedx_y_v",

    "shr_tkfit_gap10_dedx_u_v",
    "shr_tkfit_gap10_dedx_v_v",
    "shr_tkfit_gap10_dedx_y_v",

    "shr_tkfit_dedx_nhits_u_v",
    "shr_tkfit_dedx_nhits_v_v",
    "shr_tkfit_dedx_nhits_y_v",

    "shr_moliere_avg_v",
    "shr_moliere_rms_v",
}
# Columns to add to daughters from mc:
add_mc_fields = {
    "interaction",
    "weightSplineTimesTune",
    "weightSplineTimesTune_pi0scaled",
    "leeweight",
    "nu_e",
    "ccnc",
    "nu_pdg",
    "nueccinc",
    "npion",
    "npi0",
    "nproton",
    "nelec",
    "nmuon",
    "nu_purity_from_pfp",
    "optical_filter",
    "event_scale",
}

# Columns used for BDT training, order matters! 
col_train_electron = [
    "shr_dist_v",
    "shr_tkfit_4cm_dedx_wm_v",
    "shr_tkfit_dedx_y_v",
    "shr_tkfit_2cm_dedx_Y",
    "shr_tkfit_gap10_dedx_y_v",
    "shr_moliere_avg_v",
    "shr_tkfit_hitratio_v",
    "shr_subclusters_v",
    "secondshower_Y_nhit",
]
col_train_other = [
    "trk_score_v",
    "trk_distance_v",
    "trk_llr_pid_score_v",
    "pfp_trk_daughters_v",
    "pfp_shr_daughters_v",
    "e_candidate_anglediff",
    "pfp_generation_v",
    "trk_muon_hypothesis_ratio_v",
    "trk_proton_hypothesis_ratio_v",
]

col_train_event = [
    "n_showers",
    "n_pfp_farvtx",
    "contained_fraction",
    "score",
    "score_other_max",
    "score_other_min",
    "score_other_mean",
    ## nueccinc
    ## train_weight -> use the train weight of the electron candidate
]