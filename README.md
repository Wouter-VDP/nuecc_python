# Inclusive electron neutrino search in MicroBooNE with BNB 

This set of python scripts and [jupyter notebooks](https://jupyter.org/) leads to the selection and plots of inclusive electron neutrino search.
For any additional questions, feel free to email me at <wvdp@mit.edu>.

The input for these scripts are the *NTuples* produced by the [searchingfornues repository](https://github.com/ubneutrinos/searchingfornues "https://github.com/ubneutrinos/searchingfornues") 

## Preprocessing

### Finding or producing the input *NTuples*

The input *NTuples* for this analysis are coming from the Fermilab computing grid jobs of the *searchingfornues* LArSoft analyser. At the time of writing, these are produced by the PeLEE team and documented in [this spreadsheet](https://docs.google.com/spreadsheets/d/1vdcm3FoYIF1XiS6qx4qTCbaTH79vu-Sb5j8dnqctaTM/edit#gid=1726284664). The location of these *RAW* input ROOT *NTuples* is currently `/uboone/data/users/davidc/searchingfornues/`.

For additional information on how to produce these *NTuples* starting from the *Samweb* *reco2* ARTROOT files, see the [searchingfornues wiki](https://github.com/ubneutrinos/searchingfornues/wiki).

For an organised workflow, we will create symbolic links of these input *NTuples* in our `uboone/data/users` folder. 
These symbolic links can be create with wildcard bash comments such as:
```bash
for FILE in /uboone/data/users/davidc/searchingfornues/v08_00_00_25/cc0pinp/1205/*G*.root; 
do ln -s "$FILE"; 
done
```
The following scripts assume a structure in your data folder similar to `/uboone/data/users/wvdp/searchingfornues/July2020`, 
including the following folders:
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

The *syst* folder contains samples that are used to evaluate the detector variations. They are again split by run and depending on the availibility of the samples, the contents could be similar to:

| BNB Electron neutrino     | BNB All neutrinos        | Muon Charged-Current pi0    | Muon Neutral-Current pi0    |
|---------------------------|--------------------------|-----------------------------|-----------------------------|
| nue_CV.root               | nu_CV.root               | ccpi0_CV.root               | ncpi0_CV.root               |
| nue_LYAttenuation.root    | nu_LYAttenuation.root    |                             |                             |
| nue_LYDown.root           | nu_LYDown.root           | ccpi0_LYDown.root           | ncpi0_LYDown.root           |
| nue_LYRayleigh.root       | nu_LYRayleigh.root       | ccpi0_LYRayleigh.root       | ncpi0_LYRayleigh.root       |
| nue_Recomb2.root          | nu_Recomb2.root          |                             |                             |
| nue_SCE.root              | nu_SCE.root              | ccpi0_SCE.root              | ncpi0_SCE.root              |
| nue_WireModAngleXZ.root   | nu_WireModAngleXZ.root   | ccpi0_WireModAngleXZ.root   |                             |
| nue_WireModAngleYZ.root   | nu_WireModAngleYZ.root   | ccpi0_WireModAngleYZ.root   | ncpi0_WireModAngleYZ.root   |
| nue_WireModScaledEdX.root | nu_WireModScaledEdX.root | ccpi0_WireModScaledEdX.root | ncpi0_WireModScaledEdX.root |
| nue_WireModScaleX.root    | nu_WireModScaleX.root    | ccpi0_WireModScaleX.root    | ncpi0_WireModScaleX.root    |
| nue_WireModScaleYZ.root   | nu_WireModScaleYZ.root   | ccpi0_WireModScaleYZ.root   | ncpi0_WireModScaleYZ.root   |

These samples are only necessary to include detector variations in the error bars on the data-to-simulation comparison plots ([see later](#datamc)).

The sideband folder and the fake folder are containing ROOT *NTuples* that can be swapped in the plotting framework with the `beam_on.root` sample. For example, in the folder sideband, there can be a single ROOT file named `beam_sideband.root` containing the sideband information of different runs. It is also possible to split the ROOT files by runs in subfolders as before, which is done for the fake data studies (see `/uboone/data/users/wvdp/searchingfornues/July2020/fake/`).

### Processing the *NTuples* to python style objects.

The files handling the preprocessing are part of this repository in the [helpers/gpvm](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/) folder. Except for the file [ZarkoCaller.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/ZarkoCaller.py), they rely on an environment running python 3 with numpy, pandas and [uproot](https://github.com/scikit-hep/uproot) available. On the interactive nodes, this can be easily achieved by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html)  

1. Protons-on-target (POT) counting: 
   * [RunSubrun.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/RunSubrun.py)
   * [ZarkoCaller.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/ZarkoCaller.py)
2. Restructuring and slimming the data
   * [RootLoader.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/Rootloader.py)
   * [Merger.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/Merger.py)
   
## Applying the selection and adding additional variables

## Plotting the outcome

### Data to simulation comparisons
<a name="datamc"></a>

