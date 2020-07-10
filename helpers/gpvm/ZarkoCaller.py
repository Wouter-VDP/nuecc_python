# Python 2.7 needed for zarko script!
# Import
from subprocess import check_output
import glob

dir_path = "/uboone/data/users/wvdp/searchingfornues/July2020/"
files = glob.glob(dir_path+'*/run_subrun*.txt')
pot_dict = {}

for fn in files:
    run = fn.split('/')[-2]
    on_or_off = fn.split('_')[-1].split('.')[0]
    print(run, on_or_off)
    
    if run not in pot_dict:
        pot_dict[run] = {}

    if on_or_off == 'on':
        on = check_output("/uboone/app/users/zarko/getDataInfo.py -v2 --run-subrun-list {}".format(fn), shell=True)
        lines= on.split('\n')
        pot_dict[run][lines[1].split()[7]]=lines[2].split()[7]
        pot_dict[run][lines[1].split()[5]]=lines[2].split()[5]
    elif on_or_off == 'off':
        off = check_output("/uboone/app/users/zarko/getDataInfo.py -v2 --run-subrun-list {}".format(fn), shell=True)
        lines= off.split('\n')
        pot_dict[run][lines[1].split()[0]]=lines[2].split()[0]
    else:
        print('Unknown sample:', run, on_or_off)
        
print(pot_dict)
for run in pot_dict.keys():
    f = open("{}{}/scaling.txt".format(dir_path,run),"w")
    for k, v in pot_dict[run].items():
        f.write(str(k) + '\t'+ str(v) + '\n')
    f.close()
