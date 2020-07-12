import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import validation_curve

### Constants
gr = 1.618
mass_p = 0.93827
min_p_energy = mass_p + 0.04
min_e_energy = 0.020
data_samples = {"on", "off", "sideband",'set1','set2','set3','set4'}
syst_weights = ["weightsFlux", "weightsGenie", 'weightsReint']

### Electron and preselection queries
e_cand_str = "pfp_clusters_v==3 & \
              trk_score_v<0.3 & \
              shr_tkfit_2cm_nhits_Y >0 & \
              pfp_generation_v==2 & \
              trk_llr_pid_score_v>0.4 & \
              trk_len_v < 350"

query_preselect = "optical_filter & \
                   e_candidate & \
                   slpdg==12 &\
                   reco_fid_vol & \
                   shr_energy_y_v>100 & \
                   CosmicIPAll3D>30 & \
                   CosmicDirAll3D>-0.98 & \
                   CosmicDirAll3D<0.98 & \
                   topological_score > 0.15 & \
                   contained_fraction>0.4"

### POT factors
# pot_dict = {
#     "sideband": {},
#     # "sideband12": {"pot": 3.988e20, "E1DCNT_wcut": 92086705},
#     # "sideband3": {"pot": 1.842e20, "E1DCNT_wcut": 44050047},
#     "sideband12": {"pot": 4.279e20, "E1DCNT_wcut": 99029235},
#     "sideband3": {"pot": 2.561e20, "E1DCNT_wcut": 61214217},
#     "ext12": 186993192,
#     "ext3": 86991453,
#     "fake": {"pot": 5.01e20, "E1DCNT_wcut": 0}
# }
# pot_dict["sideband"]["pot"] = (
#     pot_dict["sideband12"]["pot"] + pot_dict["sideband3"]["pot"]
# )
# pot_dict["sideband"]["E1DCNT_wcut"] = (
#     pot_dict["sideband12"]["E1DCNT_wcut"] + pot_dict["sideband3"]["E1DCNT_wcut"]
# )
# pot_dict["ext"] = pot_dict["ext12"] + pot_dict["ext3"]

### Labels for angle plots
phi_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
phi_labs = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
theta_ticks = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
theta_labs = [r"$0$", r"$\pi/4$", r"$\pi$/2", r"$3\pi$/4", r"$\pi$"]

### Fiducial volume
lower = np.array([-1.55, -115.53, 0.1])
upper = np.array([254.8, 117.47, 1036.9])
fid_vol = np.array([[10, 10, 20], [10, 10, 50]])
contain_vol = np.array([[5, 6, 10], [5, 6, 10]])
fid_box = np.array([lower + fid_vol[0], upper - fid_vol[1]]).T
contain_box = np.array([lower + contain_vol[0], upper - contain_vol[1]]).T
tpc_box = np.array([lower, upper]).T


def is_in_box(x, y, z, box):
    bool_x = (box[0][0] < x) & (x < box[0][1])
    bool_y = (box[1][0] < y) & (y < box[1][1])
    bool_z = (box[2][0] < z) & (z < box[2][1])
    return bool_x & bool_y & bool_z


def is_fid(x, y, z):
    return is_in_box(x, y, z, fid_box)


def is_contain(x, y, z):
    return is_in_box(x, y, z, contain_box)


def is_tpc(x, y, z):
    return is_in_box(x, y, z, tpc_box)


### Get the pitch
def get_pitch(dir_y, dir_z, plane):
    if plane == 0:
        cos = dir_y * (-np.sqrt(3) / 2) + dir_z * (1 / 2)
    if plane == 1:
        cos = dir_y * (np.sqrt(3) / 2) + dir_z * (1 / 2)
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


# Function to plot the training evaluation
def analyse_training(
    model_file, X_test, X_train, y_test, y_train, train_ana, labels, depth
):
    fig, ax = plt.subplots(ncols=4, figsize=(8 * 1.618, 3.5), constrained_layout=True)

    y_pred = model_file.predict_proba(X_test).T[0]
    y_pred_train = model_file.predict_proba(X_train).T[0]
    fpr, tpr, _ = roc_curve(y_test[labels["train_label"]], y_pred)
    fpr_train, tpr_train, _ = roc_curve(y_train[labels["train_label"]], y_pred_train)
    roc_auc = auc(tpr, fpr)
    roc_auc_train = auc(tpr_train, fpr_train)

    ax[0].hist(
        y_pred[y_test[labels["train_label"]] == 0],
        alpha=0.5,
        bins=50,
        range=(0, 1),
        label=labels['signal'],
        density=False,
    )
    ax[0].hist(
        y_pred[y_test[labels["train_label"]] == 1],
        alpha=0.5,
        bins=50,
        range=(0, 1),
        label=labels['background'],
        density=False,
    )
    ax[0].legend(loc="upper left")
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel(labels['xlabel'])
    ax[0].set_ylabel("Entries per bin")
    ax[0].set_title(labels['title'])

    ax[1].plot(tpr, fpr, label="Test data (area = %0.3f)" % roc_auc)
    ax[1].plot(tpr_train, fpr_train, label="Train data (area = %0.3f)" % roc_auc_train)
    ax[1].plot([0, 1], [0, 1], linestyle="--")
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC curve")
    ax[1].legend(loc="lower right")

    # retrieve performance metrics
    results = model_file.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)
    # plot log loss
    ax[2].plot(x_axis, results["validation_1"]["logloss"], label="Test data")
    ax[2].plot(x_axis, results["validation_0"]["logloss"], label="Train data")
    ax[2].legend()
    ax[2].set_ylabel("Binary logistic loss")
    ax[2].set_xlabel("Training epoch")
    ax[2].set_title("XGBoost logistic loss")

    if train_ana:
        print('Started training for different depths')
        param_range = range(1, 8)
        train_scores, test_scores = validation_curve(
            XGBClassifier(),
            X_train,
            y_train[labels["train_label"]],
            param_name="max_depth",
            param_range=param_range,
            scoring="accuracy",
            n_jobs=3,
            cv=2,
        )
        print('Finished training for different depths')
    
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax[3].set_title("Validation Curve")
        ax[3].set_xlabel(r"Tree depth")
        ax[3].set_ylabel("Accuracy")
        ax[3].plot(param_range, test_scores_mean, label="Testing accuracy")
        ax[3].fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
        )
        ax[3].plot(param_range, train_scores_mean, label="Training accuracy")
        ax[3].fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
        )
        ax[3].axvline(x=depth, label="Training depth", color="C2", alpha=0.5)
        ax[3].legend()
        ax[3].set_xticks(param_range)

    fig.savefig(labels['file_name'])
