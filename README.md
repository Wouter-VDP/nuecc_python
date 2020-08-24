# Inclusive electron neutrino searches in MicroBooNE with the BNB 

This set of python scripts and jupyter notebooks leads to the selection and plots of inclusive electron neutrino search.
For any additional questions, feel free to email me at <wvdp@mit.edu>

The input for these scripts are the *NTuples* produced by the [searchingfornues repository](https://github.com/ubneutrinos/searchingfornues "https://github.com/ubneutrinos/searchingfornues") 

## Preprocessing

### Finding or producing the input *NTuples*

The input *NTuples* for this analysis are coming from fermilab computing grid jobs of the *searchingfornues* LArSoft analyser. At the time of writing, these are produced by the PeLEE team and documented in [this spreadsheet](https://docs.google.com/spreadsheets/d/1vdcm3FoYIF1XiS6qx4qTCbaTH79vu-Sb5j8dnqctaTM/edit#gid=1726284664). The location of these *RAW* input ROOT *NTuples* is currently:
> /uboone/data/users/davidc/searchingfornues/

For additional information on how to produce these *NTuples* starting from the *Samweb* *reco2* ARTROOT files, see the [searchingfornues wiki](https://github.com/ubneutrinos/searchingfornues/wiki).

For an organised workflow, we will create symbolic links of these input *NTuples* in our *uboone/data/users* folder. 
These symbolic links can be create with wildcard bash comments such as:
```bash
for FILE in /uboone/data/users/davidc/searchingfornues/v08_00_00_25/cc0pinp/1205/*G*.root; 
do ln -s "$FILE"; 
done
```
The following scripts assume a structure in your data folder similar to */uboone/data/users/wvdp/searchingfornues/July2020*, 
including the folllowing folders:
> combined  fake  run1  run2  run3  sideband  syst.

Here, the default samples, including data and filters are seperated into the run period (run1-3 folders). This should look similar to: 
```
beam_on.root
beam_off.root  
nu.root 
nue.root
dirt.root           
filter_cc_nopi.root  
filter_nc_cpi.root   
filter_nc_pi0.root                 
filter_cc_cpi.root  
filter_cc_pi0.root   
filter_nc_nopi.root  
nue.root
```
The naming convention is specified in [helpers/gpvm/enum_sample.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/enum_sample.py)

### Processing the *NTuples* to python style objects.

The files handling the preprosessing are part of this repository in the [helpers/gpvm](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/) folder. Except of the file [ZarkoCaller.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/ZarkoCaller.py), they rely on an environment running python 3 with numpy, pandas and [uproot](https://github.com/scikit-hep/uproot) availible. On the interactive nodes, this can be easily achieved by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html)  

1. 

