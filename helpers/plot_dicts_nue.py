category_labels = {
    1: r"$\nu_e$ CC other",
    10: r"$\nu_e$ CC 0$\pi$0p",
    11: r"$\nu_e$ CC 0$\pi$Np",
    111: r"MiniBooNE LEE",
    2: r"$\nu_{\mu}$ CC other",
    21: r"$\nu_{\mu}$ CC $\pi^{0}$",
    3: r"$\nu$ NC",
    31: r"$\nu$ NC $\pi^{0}$",
    4: r"Cosmic",
    5: r"Out of FV",
    6: r"other",
    7: r"Out of Cryo",  # DIRT sample
    0: r"No slice",
}
category_colors = {
    4: "xkcd:salmon",
    5: "xkcd:brick",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    3: "xkcd:cobalt",
    31: "xkcd:sky blue",
    1: "xkcd:green",
    10: "xkcd:mint green",
    11: "xkcd:lime green",
    111: "xkcd:goldenrod",
    6: "xkcd:black",
    7: "xkcd:tomato",
    0: "xkcd:black",
}



pdg_labels = {
    2212: r"$p$",
    13: r"$\mu$",
    11: r"$e$",
    111: r"$\pi^0$",
    -13: r"$\mu$",
    -11: r"$e$",
    211: r"$\pi^{\pm}$",
    -211: r"$\pi$",
    2112: r"$n$",
    22: r"$\gamma$",
    321: r"$K$",
    -321: r"$K$",
    0: "Cosmic",
}
pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "#e31a1c",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "xkcd:salmon",
}



int_labels = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    3: "Coherent",
    4: "Coherent Elastic",
    5: "Electron scatt.",
    6: "IMDAnnihilation",
    7: r"Inverse $\beta$ decay",
    8: "Glashow resonance",
    9: "AMNuGamma",
    10: "MEC",
    11: "Diffractive",
    12: "EM",
    13: "Weak Mix",
}
int_colors = {
    0: "bisque",
    1: "darkorange",
    2: "goldenrod",
    3: "lightcoral",
    4: "forestgreen",
    5: "turquoise",
    6: "teal",
    7: "deepskyblue",
    8: "steelblue",
    9: "royalblue",
    10: "crimson",
    11: "mediumorchid",
    12: "magenta",
    13: "pink",
    111: "black",
}

sys_col_labels = {
    0: (r"$\nu_e$ CC", "xkcd:green"),
    2: (r"$\nu_e$ LEE", "xkcd:goldenrod"),
    1: (r"$\nu$ bkgds","xkcd:sky blue"),
}

syst_groups_cat = {
    1: 0,
    10: 0,
    11: 0,
    2: 1,
    21: 1,
    3: 1,
    31: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    0: 1,
    111:2,
}