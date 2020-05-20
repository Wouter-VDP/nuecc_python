import pandas as pd
import numpy as np
import pickle
import awkward

input_dir = "/uboone/data/users/wvdp/searchingfornues/March2020/combined/"
target_pot = 1e21
grouper = ["sample", "Run", "event"]

train = pickle.load(open(input_dir+"train_slimmed.pckl", 'rb'))
nu = pickle.load(open(input_dir+"nu_slimmed.pckl", 'rb'))
nue = pickle.load(open(input_dir+"nue_slimmed.pckl", 'rb'))
filtered =  pickle.load(open(input_dir+"filter_slimmed.pckl", 'rb'))
input_samples = {'nu':nu, 'nue': nue, 'filter': filtered, 'train':train}


### What samples are included in these files?

for sample_name, sample_dict in input_samples.items():
    # filter_cat?
    print('\n',sample_name, '\n',np.unique(sample_dict['mc']['filter'], return_counts=True),'\n')
    # pot and event number?
    for sample_tuple, pot in sample_dict['pot'].items():
        num_events = sum((sample_dict['mc']['sample']==sample_tuple[0]) & (sample_dict['mc']['Run']==sample_tuple[1]))
        print('\t', sample_tuple,'\t POT:', pot, '\t events', num_events)


### Construct new scales

# nue sample -> easy, keep everything!
nue_mc_scale = np.full(len(nue["mc"]['Run']), target_pot/sum(nue['pot'].values()))
nue_daughter_mask = np.repeat(nue_mc_scale!=0, nue['mc']['n_pfps'])    

# nu sample, keep if filter is 0
nu_mc_scale = target_pot/sum(nu['pot'].values())*(nu['mc']['filter']==0)
nu_daughter_mask = np.repeat(nu_mc_scale!=0, nu['mc']['n_pfps'])

# filtered sample
total_entries = len(filtered["mc"]['Run'])
filtered_mc_scale = np.zeros(total_entries)
total_pot = {sample:sum([filtered['pot'][k] for k in filtered['pot'] if k[0]==sample]) for sample in np.unique(filtered['mc']['sample'])}
for i,pot in total_pot.items():
    print('filter: sample:',i,'pot',pot)
    filtered_mc_scale+= target_pot/pot*(filtered['mc']['filter']==i)
filtered_daughter_mask = np.repeat(filtered_mc_scale!=0, filtered['mc']['n_pfps'])    


### Construct new samples

nu_new = {}

nu_new["mc"] = {}
truth = [nu['mc'], nue['mc'], filtered['mc']]
truth_mask = [nu_mc_scale!=0, nue_mc_scale!=0, filtered_mc_scale!=0]
for col_mc in truth[0].keys():
    nu_new["mc"][col_mc] = awkward.concatenate([t[col_mc][b] for t,b in zip(truth, truth_mask)])
# to lower the data size, remove the weights of events without a slice:    
mask_no_slice = nu_new["mc"]['n_pfps']==0
for col_mc in ['weightsFlux', 'weightsGenie', 'weightsReint']:
    nu_new["mc"][col_mc][mask_no_slice] = None    
nu_new['mc']['event_scale'] = np.hstack([nu_mc_scale[nu_mc_scale!=0],nue_mc_scale[nue_mc_scale!=0],filtered_mc_scale[filtered_mc_scale!=0]])
nu_new["numentries"] = len(nu_new['mc']['event_scale'])
daughters = [nu['daughters'][nu_daughter_mask], nue['daughters'][nue_daughter_mask], filtered['daughters'][filtered_daughter_mask]]
nu_new["daughters"] = pd.concat(daughters, sort=False, verify_integrity=True, copy = False)
nu_new["daughters"].index.names = ["sample", "Run", "event", "daughter"]

pickle.dump(nu_new, open(input_dir + "nu_new_slimmed.pckl", "wb"))

assert(sum(nu_new['mc']['n_pfps'])==len(nu_new['daughters']))
assert(nu_new['numentries']==len(nu_new['mc']['n_pfps']))

print('Number of BNB nu event in {:.0e} POT: {:.0f} before'.format(target_pot, nu['numentries']*target_pot/sum(nu['pot'].values()) ))
print('Number of BNB nu event in {:.0e} POT: {:.0f} after'.format(target_pot,sum(nu_new['mc']['event_scale'])))

del nu_new


training_new = {}

training_new["mc"] = {}
truth = [nu['mc'], filtered['mc'], train['mc']]
truth_mask = [nu_mc_scale==0, filtered_mc_scale==0, np.full(len(train['mc']['Run']),True)]
for col_mc in truth[0].keys():
    # our training sample does not need systematic errors!
    if col_mc not in ['weightsFlux', 'weightsGenie', 'weightsReint']
        training_new["mc"][col_mc] = awkward.concatenate([t[col_mc][b] for t,b in zip(truth, truth_mask)])

training_new["numentries"] = len(training_new['mc']['Run'])
daughters = [nu['daughters'][~nu_daughter_mask], filtered['daughters'][~filtered_daughter_mask], train['daughters']]
training_new["daughters"] = pd.concat(daughters, sort=False, verify_integrity=True, copy = False)
training_new["daughters"].index.names = ["sample", "Run", "event", "daughter"]

pickle.dump(training_new, open(input_dir + "train_new_slimmed.pckl", "wb"))

assert(sum(training_new['mc']['n_pfps'])==len(training_new['daughters']))
del training_new
