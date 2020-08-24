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

The files handling the preprocessing are part of this repository in the [helpers/gpvm](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/) folder. Except for the file [ZarkoCaller.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/ZarkoCaller.py), they rely on an environment running python 3 with numpy, pandas and [uproot](https://github.com/scikit-hep/uproot) available. On the interactive nodes, this can be easily achieved by installing [miniconda](https://docs.conda.io/en/latest/miniconda.html). If processing the *NTuples* from scratch, it is advised to run the four following scripts in this order:

1. Protons-on-target (POT) counting: 
   * [RunSubrun.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/RunSubrun.py)
     * Configurable parameters: `dir_path`.
     * Function: build a `txt` file with the run subrun information for data (beam_on, beam_off, beam_sideband). 
   * [ZarkoCaller.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/ZarkoCaller.py)
     * Warning: This is the only script which relies on python 2.x and samweb tools being setup, these dependencies are enforced by `/uboone/app/users/zarko/getDataInfo.py`.
     * Configurable parameters: `dir_path`.
     * Function: creates a `scaling.txt` file with the POT/triggers information for data samples. 
    
2. Restructuring and slimming the data
   * [RootLoader.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/Rootloader.py)
     * Function: Restructure the information from the ROOT tree into a dictionary with the following keys:
       * pot: POT count of samples, per run period.
       * triggers: triggers for data samples, used for correct normalisation, per run period.
       * fields: the list of fields that were availible in the original RAW intput *NTuple*.
       * numentries: Number of events in sample, summer over run periods.
       * daughters: dataframe filled for every daughter in events passing the NeutrinoID.
       * mc: dictionary filled for every event in the sample, contains ground truth information and a field to broadcast between the `mc` and `daughters` indexes, by default, this is `n_pfps` (the number of daughters in an event).
     * Configurable parameters: `dir_path`, `syst_loading`, `out_samples`.
       The fields that will be loaded into the `mc` and `daughter` keys are defined in [col_load.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/col_load.py)
   * [Merger.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/Merger.py)
     * Function: After loading the samples in the previous step, we still have a plethora of simulated samples which contain overlap: the nu sample, nue sample, filtered samples. This python script groups these together in two outgoing samples with the same structure: a BNB nu sample which includes the increased statistics and correctly weighted the events; and a train sample which is used for retraining. 
     * Configurable parameters: `input_dir`, `training`, `remove_universes`, `reduce_query`
     * Warning: the output files of this step are needed for the subsequent selection and will dictate the memory requirement of the chain. The file size can be reduced by `remove_universes`, which removes the weights of the universes corresponding to xsec, flux and reinteraction systematics. Another way to reduce the size is by trimming the `daughters` dataframe, which is by far the largest object after the universes. The `reduce_query` throws away certain daughters. It is important that there is still a way to broadcast data between the `mc` and `daughters` data structure.
     
After succesfully running these four scripts, you should at least have the following set of files:
```
beam_on_slimmed.pckl 
beam_off_slimmed.pckl  
nu_new_slimmed.pckl    
dirt_slimmed.pckl    
```
If you intend to retrain the boosted decision trees in the selection ([see later](#bdttraining)), you should also have a file called `training_new_slimmed.pckl`. And, optionally, if you process fake dataset or sideband samples:
```
beam_sideband_slimmed.pckl  
set1_slimmed.pckl   
set2_slimmed.pckl  
set3_slimmed.pckl  
set4_slimmed.pckl
set5_slimmed.pckl
```
If you also processed the detector variation samples, you will have a large set of additional `pckl` files as listed in the table above. These files are the input of the selection and plotting framework. Personally, at this stage I copy those files to my local environment, but using miniconda, the selection and plotting can also be performed on the interactive nodes. 

## Applying the selection and adding additional variables

### Without retraining the BDTs

### With retraining the BDTs
<a name="bdttraining"></a>


## Plotting the outcome

### Data to simulation comparisons
<a name="datamc"></a>

