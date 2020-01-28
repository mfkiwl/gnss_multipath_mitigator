import autoreload
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from keras import optimizers
import keras_metrics
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

#%%

from data_generator_resample import CorrDatasetResample
from model import DopplerRegressor
from utils import save_model, load_model


#%% Define metrics
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res / (ss_tot + K.epsilon()))
    
#%%

discr_size_fd = 40
scale_code = 40
Tint = 1e-3
w = 10**6 # correlator bandwidth

file_ = 'config_combin/config_combin_ph_multi_target_regr.json'
configs = []
with open(file_) as json_config_file:
	configs.append(json.load(json_config_file))
config = configs[0]

tau = [0, 2]
dopp = [-2000, 2000]

delta_tau_interv = [config['delta_tau_min'], config['delta_tau_max']]
delta_dopp_interv = [config['delta_dopp_min'], config['delta_dopp_max']]
alpha_att_interv = [config['alpha_att_min'], config['alpha_att_max']]
delta_phase_set = list(np.array(config['delta_phase']) * np.pi / 180)
cn0_log_set = config['cn0_log']

#print('CHECK JSON READ: ', delta_tau[0], delta_tau[1], delta_dopp[0], delta_dopp[1], alpha_att[0], alpha_att[1])
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
    dataset_temp = Dataset.build(nb_samples=3000)
    # Concatenate and shuffle arrays
    dataset = np.concatenate((dataset, dataset_temp), axis=0)
    
# Split the data into train and val
np.random.shuffle(dataset)
data_train, data_val = train_test_split(dataset, test_size=0.2)

X_train = np.array([x['table'] for x in data_train])
X_val = np.array([x['table'] for x in data_val])

y_dopp_train = np.array([x['delta_dopp'] for x in data_train])[...,None]
y_dopp_val = np.array([x['delta_dopp'] for x in data_val])[...,None]

#y_dopp_train = y_dopp_train
#y_dopp_val = y_dopp_val

#%% Train model from scratch
model = DopplerRegressor(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

batch_size = 16
train_iters = 20
learning_rate = 1e-4

model.model.compile(
        loss='mse',
		optimizer=optimizers.Adam(lr=learning_rate),
		metrics=[rmse, r2])

# Define callbacks
reduce_lr = ReduceLROnPlateau()
early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

history = model.model.fit(
					  x=X_train,
					  y=y_dopp_train,
					  validation_data=(X_val, y_dopp_val),
					  epochs=train_iters,
					  batch_size=batch_size,
                         callbacks=[reduce_lr, early_stopping],
                         verbose=1
				    )

# write the logs to .json file
history_dict = {k:[np.float64(i) for i in v] for k,v in history.history.items()}

#%% Load pretrainded model
#model = DopplerRegressor(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
#
#batch_size = 16
#train_iters = 20
#learning_rate = 1e-4
#
#model.model.compile(
#        loss='mse',
#		optimizer=optimizers.Adam(lr=learning_rate),
#		metrics=[rmse, r2])
#
#model.model = load_model(r'saved_models/model_dopp_v1.pkl')

#%% Test model

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
#y_test = y_dopp_test

y_pred = model.model.predict(x=X_test, batch_size=batch_size, verbose=1)
print('dopp: ', r2_score(y_dopp_test, y_pred[:,0]), np.sqrt(mean_squared_error(y_pred[:,0], y_dopp_test)))

#%% save model
save_model(model.model, 'saved_models/model_dopp_v1_2000.pkl')


#%% make classif-regression pipeline
labels = {(-1000, -500): 0, (-500, 0): 1, (0, 500): 2, (500, 1000): 3}

# encode y_dopp labels in dataset

clf = DopplerClassifier(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))





