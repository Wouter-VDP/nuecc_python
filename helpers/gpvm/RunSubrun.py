# Import
import uproot
import numpy as np
import pandas as pd
import glob
from pathlib import Path

dir_path = "/uboone/data/users/wvdp/searchingfornues/July2020/"
root_dir = 'nuselection'
tree_name = "SubRun"
files = glob.glob(dir_path+'*/beam*.root')

for fn in files:
    tree=uproot.open(fn)[root_dir][tree_name]
    out_title = Path(fn).stem
    out_name = str(Path(fn).parent) + "/run_subrun_" + out_title + ".txt"
    print(out_title)
    run_subrun = tree.arrays(["run", "subRun"], outputtype=pd.DataFrame)
    print('Number of rows before duplicates:', len(run_subrun))
    run_subrun.drop_duplicates(keep='first', inplace=True)
    print('Number of rows after removing duplicates:', len(run_subrun))
    np.savetxt(out_name, np.array(run_subrun), fmt="%d")
