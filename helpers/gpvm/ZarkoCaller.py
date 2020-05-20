# Python 2.7 needed for zarko script!
# Import
from subprocess import check_output

dir_path = "/uboone/data/users/wvdp/searchingfornues/March2020/combined/"
pot_dict = {}

folders = ['beam_on/run_subrun_run123_filtered_beam_on_uniq.txt']#, 'beam_off/run_subrun_beam_off.txt']

print "Beam On Sample:"
on = check_output("/uboone/app/users/zarko/getDataInfo.py -v2 --run-subrun-list {}{}".format(dir_path, folders[0]), shell=True)
print on
lines= on.split('\n')
pot_dict[lines[1].split()[7]]=lines[2].split()[7]
pot_dict[lines[1].split()[5]]=lines[2].split()[5]

print "Beam Off Sample:"
off = check_output("/uboone/app/users/zarko/getDataInfo.py -v2 --run-subrun-list {}{}".format(dir_path, folders[1]), shell=True)
print off
lines= off.split('\n')
pot_dict[lines[1].split()[0]]=lines[2].split()[0]

if all(k in pot_dict for k in ("EXT","E1DCNT_wcut", "tor875_wcut")):
    f = open("{}scaling.txt".format(dir_path),"w")
    for k, v in pot_dict.items():
        f.write(str(k) + '\t'+ str(v) + '\n')
    f.close()
    print str(pot_dict)
else: 
    print "Failed!" 
