# Import
import uproot
import numpy as np
import pandas as pd
import glob
from pathlib import Path

dir_path = "/uboone/data/users/wvdp/searchingfornues/March2020/combined/"
root_dir = 'nuselection'
tree_name = "SubRun"

folders = ['beam_on', 'beam_off']
run_subrun_all = []
for folder in folders:
    fns = glob.glob(dir_path+folder+'/*.root')
    for fn in fns:
        tree=uproot.open(fn)[root_dir][tree_name]
        out_title = Path(fn).stem
        out_name = str(Path(fn).parent) + "/run_subrun_" + out_title + ".txt"
        print(out_title)
        run_subrun = np.array(tree.arrays(["run", "subRun"], outputtype=pd.DataFrame))
        np.savetxt(out_name, run_subrun, fmt="%d")
        run_subrun_all.append(run_subrun)
    np.savetxt(dir_path+folder+"/run_subrun_" + folder + ".txt", np.vstack(run_subrun_all), fmt="%d")
