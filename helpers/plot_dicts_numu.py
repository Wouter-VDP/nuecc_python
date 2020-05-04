category_labels = {
    30: r"$\nu_\mu$ CC 0p",
    31: r"$\nu_\mu$ CC 0$\pi$Np",
    32: r"$\nu_\mu$ CC M$\pi$Np",
    33: r"$\nu_\mu$ NC",
    34: r"$\nu_{\mu}$ CC other",
    10: r"$\nu_e$",
    4: r"Cosmic",
    5: r"Out of FV",
    6: r"other",
    7: r"Out of Cryo",  # DIRT sample
    0: r"No slice"
}

category_colors = {
    4: "xkcd:salmon",
    5: "xkcd:brick",
    2: "xkcd:cyan",
    21: "xkcd:cerulean",
    3: "xkcd:cobalt",
    33: "xkcd:sky blue",
    31: "xkcd:green",
    30: "xkcd:mint green",
    32: "xkcd:lime green",
    10: "xkcd:goldenrod",
    6: "xkcd:black",
    7: "xkcd:tomato",
    0: "xkcd:black"
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
    0: "Cosmic"
}

pdg_colors = {
    2212: "#a6cee3",
    22: "#1f78b4",
    13: "#b2df8a",
    211: "#33a02c",
    111: "#137e6d",
    0: "xkcd:salmon",
    11: "#ff7f00",
    321: "#fdbf6f",
    2112: "#cab2d6",
}

int_labels_true = {
    0: "QE",
    1: "Resonant",
    2: "DIS",
    10: "MEC",
}

int_labels = {
    30: "QE",
    31: "Resonant",
    32: "DIS",
    310: "MEC",
    33: r"$\nu_\mu$ NC",
    10: r"$\nu_e$",
    4: r"Cosmic",
    5: r"Out of FV",
    6: r"other",
    7: r"Out of Cryo",  # DIRT sample
    0: r"No slice"
}

int_colors = {
    4: "xkcd:salmon",
    5: "xkcd:brick",
    3: "xkcd:cobalt",
    33: "xkcd:sky blue",
    30: "xkcd:green",
    31: "xkcd:mint green",
    32: "xkcd:lime green",
    310: "darkgreen",
    10: "xkcd:goldenrod",
    6: "xkcd:black",
    7: "xkcd:tomato",
    0: "xkcd:black"
}

sys_col_labels = {
    0: (r"$\nu_\mu$ CC", "xkcd:green"),
    1: (r"$\nu$ bkgds","xkcd:sky blue"),
}

syst_groups_cat = {
    30: 0,
    31: 0,
    32: 0,
    33: 1,
    34: 1,
    10: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    0: 1,
}