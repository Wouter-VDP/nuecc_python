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

The first stage of the selection is the NeutrinoID, one can verify if an event/daughter passed this step by testing for `n_pfps>0`. Events that do not pass the NeutrinoID will not have reconstructed pfps (Particle flow particles).

The electron neutrino selection is performed on the `daughters` entry of the dictionary. The goal is to add three boolean columns:
* e_candidate: True for all daughters that fullfill the `e_cand_str` query defined in [helpers/helpfunction.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/helpfunction.py).
* preselect: True for all daughters in an event that passes the preselection cuts, which are defined by `query_preselect` in [helpers/helpfunction.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/helpfunction.py).
* select: True for e_candidate daughters that pass the full BDT-based selection. This selection is parametrised by a single cut value on the event BDT response, defined as `cut_val` in [nue_selection_helper.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/nue_selection_helper.py).

The preselect stage also requires a fiducial volume, this is defined in [helpers/helpfunction.py#L66-L72](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/helpfunction.py#L66-L72).

After applying the selection on the samples of interest, we end up with maximum three `pckl` files, which should be located in `intput/*/lite/`:
* `after_training.pckl`
* `sys_after_training.pckl`
* `detvar_dict.pckl`

### Without retraining the BDTs

The selection flow is excecuted by [nue_selection_helper.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/nue_selection_helper.py). 

This can be excuted with a command similar to:
```python
import nue_selection_helper as nue_helper
input_dir = "./input/July2020/"
output_dir = "./output/"
plot_samples = {'nu','set1','set2','set3','set4','set5',"dirt", "on", "off",'sideband'}
nue_helper.CreateAfterTraining(plot_samples, 
                               input_dir, 
                               one_file=input_dir+'lite/after_training.pckl')
```
Which will create `after_training.pckl` including the `plot_samples` list as keys. Note that detector variations are not included in `plot_samples`. Although one could do this, to reduce file sizes, they are stored in `sys_after_training.pckl`. The latter is created by [NuePlots_DetSys.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_DetSys.ipynb).

### With retraining the BDTs
<a name="bdttraining"></a>

The inclusive electron neutrino selection contains three boosted decision trees. These are modelled by the [XGBoost](https://xgboost.readthedocs.io/en/latest/) package and the trained models are stored in [models/](https://github.com/Wouter-VDP/nuecc_python/tree/master/models).

The training is performed in the bulk of [NueSelection.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NueSelection.ipynb) and requires an input file called `training_new.pckl` which can be created by [helpers/gpvm/Merger.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/gpvm/Merger.py). One should be extremely carefull not to have any duplicated events between `training_new.pckl` and `nu_new.pckl`. In the past, the training set consisted of unused filters, Run 2 overlay samples, low-energy electron neutrino samples and the redundant events (as replaced by the filters) in the Run 1 and Run 3 BNB nu overlay events. No data was used in the training process. Note that these combinations are not set in stone and one is free to construct a training data-set as pleased as long as it is disjunct from the plotting data-sets.

The configuration of the selection is identical to [nue_selection_helper.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/nue_selection_helper.py) but is excecuted step-by-step to enhance the tunability and intermediate outputs of the selection.

Additionaly there are a set of configurable parameters connected to the training:
* `retrain` (bool): retrain the three XGBoost models. 
* `train_ana` (bool): perform a scan over a set of tree depths to determine the omptimal depth. This is needed to produce the plots in [output/training](https://github.com/Wouter-VDP/nuecc_python/tree/master/output/training) created by `nue_helper.helper.analyse_training`
* `test_size` (0-1 range): fraction of events in `training_new.pckl` that is used for training, versus used for the evaluation metrics. Note that even if this is set to 0, the events will be completely disjunct from `nu_new.pckl`, as should be the case. While evaluating, 0.25 works well; for final trianing, 0 can be used.
* `lee_focus` (default 1.0): variable that increases training weight for low energetic electron neutrino events. 

The columns that are used for training the three boosted decision trees as defined in [helpers/nue_selection_columns.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/nue_selection_columns.py). In the same file, the columns that are kept and removed to create the `after_training.pckl` file can be changed, depending on the fields one want to plot.

## Plotting the outcome

Congratulations, you made it to actually plotting the results!
The plotting is handled by a class defined in [helpers/plot_class.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/plot_class.py).
The class is able to make plots for both muon and electron selections, the categories for the plots are defined in [helpers/plot_dicts_nue.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/plot_dicts_nue.py) and [helpers/plot_dicts_numu.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/plot_dicts_numu.py).

### Truth-based plots

Plots that only require simulation information are done in [NuePlots_truth.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_truth.ipynb). These do not require the plotting class but do use some basic help functions to aide the efficiency calculations. The notebook should be self-explanatory. 

### Data to simulation comparisons
<a name="datamc"></a>

All data to simualtion comparison plots are inside [NuePlots_datamc.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_datamc.ipynb) and rely on the plotting class. The class is initialised as follows:
    
```(python)
plotter = plot_class.Plotter(
      location,                     # Path to the input file after selection (after_training.pckl)
      signal="nue",                 # Signal can be either 'nue' or 'numu'
      genie_version="mcc9.1",       # Defines the basic event weight from genie: mcc8, mcc9.0 or mcc9.1
      norm_pot=0,                   # In case you want the POT scaled to a fixed value instead of the data.
      master_query=None,            # Query that is applied on all events that will be loaded, reduces memory.
      beam_on="on",                 # Sample that is used as the neutrino data; on, sidebands, fake data-sets ...
      pot_dict={},                  # Overwrite the pot scaling of the data with a custom dict, can be useful to look at specific Runs.
      load_syst=None,               # List of strings that contain the multiverse systematics.
      load_detvar=None,             # Path to the dictionary taking care of the detector variations.
      show_lee=False,               # Default choice of showing the MiniBooNE LEE model in the plots.
      pi0_scaling=False,            # Apply a predefined pi0 scaling on the events.
      dirt=True,                    # Default choice to show the dirt sample in the plots.
      n_uni_max=2000,               # Maximum amount of universes used be load_syst vartiation, reduces memory.
      write_slimmed_output=False,   # Write some fields of selected events to plain text file.
  )
```
Example initialisations of these fields can be found in [NuePlots_datamc.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_datamc.ipynb).

To make actual plots, the function `plotter.plot_panel_data_mc` is used:

```
def plot_panel_data_mc(
        ax,                      # {tuple} -- matplotlib axes, length should be 2
        field,                   # {string} -- argument of pandas.eval()
        x_label,                 # {string} -- x-axis label
        N_bins,                  # {int} -- number or bins
        x_min,                   # {float} -- the minimum number along x
        x_max,                   # {float} -- the maximum number along x
        query="",                # {string} -- pandas.query() argument applied on all events
        title_str="",            # {string} -- right title string of the upper plot
        legend=True,             # {bool} -- plot the legend on the right of the panel
        y_max_scaler=1.025,      # {float} -- increase the upper y-axis range
        kind="cat",              # {string} -- 'cat' (categories) / 'pdg' / 'int' (interaction type)  
        show_data=True,          # {bool} -- plot the beam data
        show_syst=True,          # {bool} -- include systematic errors
        syst_fractions=None,     # {list} -- deprecated! list of fractional errors per bin
        y_label="Events per bin",
        show_lee=None,           # {bool} -- overwrite the show_lee bool in the class
)
```
The function returns:
* `ratio`: data to simulation ratio and error
* `purity`: signal purity of the selection for events passing the `query`
* `ks_test_p`: probability of the KS test, two samples, weigthed as defined on [github.com/scipy/scipy/issues/12315](https://github.com/scipy/scipy/issues/12315)
* `cnp`: Combined Neyman–Pearson Chi-square and p-cvalue ([arxiv.org/pdf/1903.07185](https://arxiv.org/pdf/1903.07185.pdf)).
* `best_text_loc`: Best position to write text on the returned plot. 0: left, 1: middle, 2: right

Plenty of examples are avilible in the [NuePlots_datamc.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_datamc.ipynb) notebook to demonstrate the use in practice.

### Covariance matrices 

The covariance matrix is generated for every variable when `plot_panel_data_mc` is called. In [](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_Cov.ipynb) it is demonstrated how the covariance matrix can be extracted on its own and plotted as a 2D histogram using the function `get_cov` in the plotting class. 

### Detector Variations 

Detector variations are handled by [NuePlots_DetSys.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_DetSys.ipynb) which mostly relies on [helpers/detvar.py](https://github.com/Wouter-VDP/nuecc_python/blob/master/helpers/detvar.py).

The basic procedure is:
* Take the `nue`, `ncpi0` and `ccpi0` central value (CV) detvar sample and weight it up to a fixed POT (by default, `1e21` is taken)
* Given a selection query, a plotting variable (field) and the x-axis range, create a new number of bins to optimise statistics: if less than 8 bins, keep bins, otherwise, reduce the number of bins with factor 2. For the `nu` sample, follow the same procedure but take a single bin due to low statistics after selection.
* For the CV variation, with this binning, calculate the histogram.
* For each variation, calculate the histogram and compare with the CV. If the difference is within one sigma combined statistical error of both samples, ignore the difference for that bin/variation. If it is larger, keep it.
* Sum the differences in quadrature over the different variations for each bin.
* Go back to the original binning:
  * Do nothing if the binning is the same as the original
  * Otherwise, divide the difference over the 2 bins proportional to the CV sample.
* Add the four samples (`nu`, `nue`, `ncpi0`, `ccpi0`) in quadrature
* The result, the error per bin, gets stored in a dictionary.
* Access this dictionary with the plotter class and scale to the POT of the plot.

This is fully automated by running the data to simulation plots first. In this step, requests will be added to the dictionary that keeps track of the detector variations. Now, run the cell in [NuePlots_DetSys.ipynb](https://github.com/Wouter-VDP/nuecc_python/blob/master/NuePlots_DetSys.ipynb) that calls `detvar.get_syt_var` for every combination. This will calculate the detecotr variations for all the plots we want to make and update the dictionary file accordingly. Finally, rerun the data to simulation comparison plots. THhe dictionary should not contain the detector variations and they will be included in the plots.


