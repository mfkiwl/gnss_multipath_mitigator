# Autoreload 
import autoreload
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
#import tensorflow as tf
#from keras.utils import to_categorical
#from tqdm import tqdm, tqdm_notebook                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
from PIL import Image
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

#from skimage import filters
from sklearn.model_selection import train_test_split

import json
import glob

from keras import optimizers
import keras_metrics
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#%% Import modules
from data_generator_resample import CorrDatasetResample
from utils import visualize_plt, visualize_3d_discr
from model import MultiTargetRegressor


#%%
# Main for data generation 
discr_size_fd = 40
scale_code = 40

tau = [0, 2]
dopp = [-2500, 2500]

delta_tau = [0.1, 0.8]
delta_dopp = [250, 1000]
delta_phase = 0
alpha_att = [0.5, 0.9]
Tint = 20e-3

#%% Check CorrDataset
Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=True,
                        delta_tau_interv=delta_tau, 
                        delta_dopp_interv=delta_dopp,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att,
                        tau=tau, dopp=dopp,
                        cn0_log=50, w=10**6)

# generate 1 peak to check
#matrix, x, y = Dataset.generate_peak()

# generate 10 samples
samples, module, delta_doppi, delta_taui, alpha_atti = Dataset.build(nb_samples=1)
#samples, module = Dataset.build(nb_samples=1)

#visualize_plt(samples[0]['table'][...,0])
#visualize_plt(samples[0]['table'][...,1])
#visualize_plt(module[...,0])

#%% Visualize peaks with plotly
#for channel in ['I', 'Q', 'module']:
#    for delta_phase in [0, np.pi/4, np.pi/2, np.pi]:

channel = 'module'
delta_phase = 0
    
Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
                    scale_code=scale_code,
                    Tint=Tint,
                    multipath_option=False,
                    delta_tau_interv=delta_tau, 
                    delta_dopp_interv=delta_dopp,
                    delta_phase=delta_phase,
                    alpha_att_interv=alpha_att,
                    tau=tau, dopp=dopp,
                    cn0_log=50, w=10**6)

samples, module = Dataset.build(nb_samples=1)

filename = 'visu_plotly/plotly_visu_phase-{}_channel-{}.html'.format(delta_phase, channel)

img_dict = {'I': samples[0]['table'][...,0], 
            'Q': samples[0]['table'][...,1],
            'module': module}

visualize_3d_discr(func=img_dict[channel],
                   discr_size_fd=discr_size_fd,
                   scale_code=scale_code,
                   tau_interv=tau, 
                   dopp_interv=dopp,
                   Tint=Tint,
                   delta_dopp=0,
                   delta_tau=0,
                   alpha_att=0,
                   delta_phase=delta_phase,
                   filename=filename)

#%% Check reference feature extractor
Dataset_mp = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=True,
                        delta_tau_interv=delta_tau, 
                        delta_dopp_interv=delta_dopp,
                        delta_phase=delta_phase,
                        alpha_att_interv=alpha_att,
                        tau=tau, dopp=dopp,
                        cn0_log=50, w=10**6)
Dataset_nomp = CorrDatasetV2(discr_size_fd=discr_size_fd,
                        scale_code=scale_code,
                        Tint=Tint,
                        multipath_option=False,
                        tau=tau, dopp=dopp,
                        cn0_log=50, w=10**6)

#%%
# generate 1 peak to check
#matrix, x, y = Dataset.generate_peak()

# generate 10 samples
samples, ref_data_samples_mp, module_mp, delta_doppi, delta_taui, alpha_atti = Dataset_mp.build(nb_samples=1, ref_features=True)   
visualize_plt(module_mp[...,0])
print(ref_data_samples_mp)
samples, ref_data_samples_nomp, module_nomp = Dataset_nomp.build(nb_samples=1, ref_features=True)   
visualize_plt(module_nomp[...,0])
print(ref_data_samples_nomp)

# create DF from ref_data_samples dict
df_mp = pd.DataFrame(list(ref_data_samples_mp))
df_nomp = pd.DataFrame(list(ref_data_samples_nomp))
df = pd.concat([df_mp, df_nomp])

#%%
X, y = preprocess_df(df)
df_av = pd.DataFrame(X, columns=['f2', 'f3'])
df_av['label'] = y
# df.groupby('label').std()
#%% Train/ Val SVM on ref features
features = ['f2', 'f3']
target = ['label']

# shuffle dataset
df_av = df_av.sample(frac=1).reset_index(drop=True)

df_train_val, df_test = train_test_split(df_av, test_size=0.2, random_state=42,
                                         shuffle=True, stratify=df_av['label'])


# Train, test split
Xtrain_val, Xtest, ytrain_val, ytest = df_train_val[features], df_test[features], df_train_val[target], df_test[target] 
Xtrain_val, Xtest, ytrain_val, ytest = Xtrain_val.values, Xtest.values, ytrain_val.values, ytest.values


# define k-fold cross val / SVC
kfold = KFold(n_splits=3, shuffle=True)
model = SVC(kernel='rbf', gamma='auto', verbose=True)

# train SVC
scores = []
for train_index, val_index in kfold.split(Xtrain_val):
    # train_test split
    Xtrain, Xval = Xtrain_val[train_index], Xtrain_val[val_index]
    ytrain, yval = ytrain_val[train_index], ytrain_val[val_index]
    
    #scale features
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain_norm = scaler.transform(Xtrain)
    Xval_norm =scaler.transform(Xval)
    
    model.fit(Xtrain_norm, ytrain)
    scores.append(model.score(Xval_norm, yval))

# check preds / classification report
Xtest_norm = scaler.transform(Xtest)
preds = model.predict(Xtest_norm)
confusion_matrix = pd.crosstab(ytest.T[0], preds, rownames=['Real'], colnames=['Predicted'])
print(confusion_matrix)
print(classification_report(ytest.T[0], preds))


precisions, recalls, _, __ = precision_recall_fscore_support(ytest.T[0], preds, average=None)
precision, recall = precisions[1], recalls[1]
accuracy = accuracy_score(ytest.T[0], preds)

#%% Check multi output model

# Read config files
	
#configs = []
#	allFiles = glob.glob("config_ti20/config_*.json") #config_dopp_ph0.json 
#	for file_ in allFiles:
#		with open(file_) as json_config_file:
#			configs.append(json.load(json_config_file))
#	
#	print(configs[0])


	#% Create dataset for given config

#	for config in configs:
#		tau = [0, 2]
#		dopp = [-2500, 2500]
#
#		delta_tau = [config['delta_tau_min'], config['delta_tau_max']]
#		delta_dopp = [config['delta_dopp_min'], config['delta_dopp_max']]
#		alpha_att = [config['alpha_att_min'], config['alpha_att_max']]
#		delta_phase = config['delta_phase'] * np.pi / 180
#		cn0_logs = config['cn0_log']
#
#		print('CHECK JSON READ: ', delta_tau[0], delta_tau[1], delta_dopp[0], delta_dopp[1], alpha_att[0], alpha_att[1])

Tint = 1e-3
w = 10**6 # correlator bandwidth

file_ = 'config/config_dopp_ph0.json'
configs = []
with open(file_) as json_config_file:
	configs.append(json.load(json_config_file))

tau = [0, 2]
dopp = [-2500, 2500]

config = configs[0]
delta_tau = [config['delta_tau_min'], config['delta_tau_max']]
delta_dopp = [config['delta_dopp_min'], config['delta_dopp_max']]
alpha_att = [config['alpha_att_min'], config['alpha_att_max']]
delta_phase = config['delta_phase'] * np.pi / 180
cn0_log = config['cn0_log'][2]
dataset = np.array([])
for multipath_option in [True, False]:
    if multipath_option: 
        Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau, 
									    delta_dopp_interv=delta_dopp,
									    delta_phase=delta_phase,
									    alpha_att_interv=alpha_att,
									    tau=tau, dopp=dopp,
									    cn0_log=cn0_log, w=w)
    else:
        Dataset = CorrDatasetV2(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau, 
									    delta_dopp_interv=delta_dopp,
									    delta_phase=0,
									    alpha_att_interv=alpha_att,
									    tau=tau, dopp=dopp,
									    cn0_log=cn0_log, w=w)
    dataset_temp = Dataset.build(nb_samples=1000)
    dataset = np.concatenate((dataset, dataset_temp[0]), axis=0)                
				

# Split the data into train and val
np.random.shuffle(dataset)
data_train, data_val = train_test_split(dataset, test_size=0.2)

X_train = np.array([x['table'] for x in data_train])
X_val = np.array([x['table'] for x in data_val])

#y_train = np.array([x['label'] for x in data_train])
#y_val = np.array([x['label'] for x in data_val])
y_dopp_train = np.array([x['delta_dopp'] for x in data_train])[...,None]
y_dopp_val = np.array([x['delta_dopp'] for x in data_val])[...,None]

y_tau_train = np.array([x['delta_tau'] for x in data_train])[...,None]
y_tau_val = np.array([x['delta_tau'] for x in data_val])[...,None]

y_alpha_train = np.array([x['alpha'] for x in data_train])[...,None]
y_alpha_val = np.array([x['alpha'] for x in data_val])[...,None]

y_phase_train = np.array([x['delta_phase'] for x in data_train])[...,None]
y_phase_val = np.array([x['delta_phase'] for x in data_val])[...,None]

y_train = np.concatenate((y_dopp_train, y_tau_train, y_alpha_train, y_phase_train), axis=1)
y_val = np.concatenate((y_dopp_val, y_tau_val, y_alpha_val, y_phase_val), axis=1)

               
model = MultiTargetRegressor(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
				

batch_size = 16
train_iters = 10
learning_rate = 1e-4

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
			    
model.model.compile(
        loss='mse',
		optimizer=optimizers.Adam(lr=learning_rate),
		metrics=[rmse])

history = model.model.fit(
					  x=X_train,
					  y=y_train,
					  validation_data=(X_val, y_val),
					  epochs=train_iters,
					  batch_size=batch_size
				    )

# write the logs to .json file
history_dict = {k:[np.float64(i) for i in v] for k,v in history.history.items()}


#%% Tests parametrization

Tint = 1e-3
w = 10**6 # correlator bandwidth

file_ = 'config_combin/config_combin_ph_multi_target_regr.json'
configs = []
with open(file_) as json_config_file:
	configs.append(json.load(json_config_file))
config = configs[0]

tau = [0, 2]
dopp = [-2500, 2500]

delta_tau_interv = [config['delta_tau_min'], config['delta_tau_max']]
delta_dopp_interv = [config['delta_dopp_min'], config['delta_dopp_max']]
alpha_att_interv = [config['alpha_att_min'], config['alpha_att_max']]
delta_phase_set = list(np.array(config['delta_phase']) * np.pi / 180)
cn0_log_set = config['cn0_log']

#print('CHECK JSON READ: ', delta_tau[0], delta_tau[1], delta_dopp[0], delta_dopp[1], alpha_att[0], alpha_att[1])


#for test_iter in range(20):
dataset = np.array([])
for multipath_option in [True, False]:
    Dataset = CorrDatasetResample(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau_interv, 
									    delta_dopp_interv=delta_dopp_interv,
									    alpha_att_interv=alpha_att_interv,
                                             delta_phase_set=delta_phase_set,		   
									    cn0_log_set=cn0_log_set,
                                tau=tau, dopp=dopp)
    dataset_temp = Dataset.build(nb_samples=1000)
    # Concatenate and shuffle arrays
    dataset = np.concatenate((dataset, dataset_temp), axis=0)

# Split the data into train and val
np.random.shuffle(dataset)
data_train, data_val = train_test_split(dataset, test_size=0.2)

X_train = np.array([x['table'] for x in data_train])
X_val = np.array([x['table'] for x in data_val])

y_dopp_train = np.array([x['delta_dopp'] for x in data_train])[...,None]
y_dopp_val = np.array([x['delta_dopp'] for x in data_val])[...,None]

y_tau_train = np.array([x['delta_tau'] for x in data_train])[...,None]
y_tau_val = np.array([x['delta_tau'] for x in data_val])[...,None]

y_alpha_train = np.array([x['alpha'] for x in data_train])[...,None]
y_alpha_val = np.array([x['alpha'] for x in data_val])[...,None]

y_phase_train = np.array([x['delta_phase'] for x in data_train])[...,None]
y_phase_val = np.array([x['delta_phase'] for x in data_val])[...,None]

y_train = np.concatenate((y_dopp_train, y_tau_train, y_alpha_train, y_phase_train), axis=1)
y_val = np.concatenate((y_dopp_val, y_tau_val, y_alpha_val, y_phase_val), axis=1)
               
model = MultiTargetRegressor(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

batch_size = 16
train_iters = 20
learning_rate = 1e-4

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res / (ss_tot + K.epsilon()))
   
model.model.compile(
        loss='mse',
		optimizer=optimizers.Adam(lr=learning_rate),
		metrics=[rmse, r2])

# Define callbacks
reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

history = model.model.fit(
					  x=X_train,
					  y=y_train,
					  validation_data=(X_val, y_val),
					  epochs=train_iters,
					  batch_size=batch_size,
                         callbacks=[reduce_lr, early_stopping],
                         verbose=1
				    )

# write the logs to .json file
history_dict = {k:[np.float64(i) for i in v] for k,v in history.history.items()}
                
#%% Test model

from sklearn.metrics import r2_score, mean_squared_error

dataset_test = np.array([])
for multipath_option in [True, False]:
    Dataset = CorrDatasetResample(discr_size_fd=discr_size_fd,
									    scale_code=scale_code,
									    Tint=Tint,
									    multipath_option=multipath_option,
									    delta_tau_interv=delta_tau_interv, 
									    delta_dopp_interv=delta_dopp_interv,
									    alpha_att_interv=alpha_att_interv,
                                             delta_phase_set=delta_phase_set,		   
									    cn0_log_set=cn0_log_set,
                                tau=tau, dopp=dopp)
    dataset_test_temp = Dataset.build(nb_samples=1000)
    # Concatenate and shuffle arrays
    dataset_test = np.concatenate((dataset_test, dataset_test_temp), axis=0)

np.random.shuffle(dataset_test)
X_test = np.array([x['table'] for x in dataset_test])

y_dopp_test = np.array([x['delta_dopp'] for x in dataset_test])[...,None]
y_tau_test = np.array([x['delta_tau'] for x in dataset_test])[...,None]
y_alpha_test = np.array([x['alpha'] for x in dataset_test])[...,None]
y_phase_test = np.array([x['delta_phase'] for x in dataset_test])[...,None]
y_test = np.concatenate((y_dopp_test, y_tau_test, y_alpha_test, y_phase_test), axis=1)

y_pred = model.model.predict(x=X_test, batch_size=batch_size, verbose=1)

print('dopp: ', r2_score(y_dopp_test, y_pred[:,0]), np.sqrt(mean_squared_error(y_pred[:,0], y_dopp_test)))
print('tau: ', r2_score(y_tau_test, y_pred[:,1]), np.sqrt(mean_squared_error(y_pred[:,1], y_tau_test)))
print('alpha: ', r2_score(y_alpha_test, y_pred[:,2]), np.sqrt(mean_squared_error(y_pred[:,2], y_alpha_test)))
print('phase: ', r2_score(y_phase_test, y_pred[:,3]), np.sqrt(mean_squared_error(y_pred[:,3], y_phase_test)))


#%% Save/ Load model
import pickle

def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
        
def load_model(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)
    
save_model(model.model, 'saved_models/model_mod1')








