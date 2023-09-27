#%%
from datetime import datetime
start=datetime.now()



## Differences to Aniel's Code
## V2: V1 + Rearrange train, validation and test data to 4 1 1 respectively(Aniel's Code:
##     train validation test, 2 1 1)

import glob
import random
from StormPrediction_Class_Hadi_WS import Scaler_Generator
from StormPrediction_Class_Hadi_WS import Data_Generator
from StormPrediction_Class_Hadi_WS import Get_files
#from StormPrediction_Class import parameters

from sklearn.preprocessing import StandardScaler
import joblib

from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint


from keras import backend as K
import pandas as pd

#%%
#Point to folders with processed files, for example '**PATH/TO/FILE**/#04H_2018-05-23_04_EPS_Storms_Hourly'
#files = glob.glob('/Users/aboulfazlsimorgh/Desktop/HadiWorkspace/Data/*')
#files = glob.glob('/mnt/dav/*Output/ML*/*')
#files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
files = glob.glob('/mnt/data/IsoBar_Fused_Files_V2/*')
#files = glob.glob('Z:\Codes_Output\ML_Data\*')
#files = glob.glob('E:\Data\IsoBar_Fused_Files_V2\*')


print('Found '+str(len(files))+' Files')


#Select Days to use for training/validating, format 'YYYY-MM-DD'


train_days = ['2018-06-03',
             '2018-06-04',
             '2018-06-05',
             '2018-06-06',
             '2018-06-07',
             '2018-06-10',
             '2018-06-11',
             '2018-06-12',
             '2018-06-13',
             '2018-06-14',
             '2018-06-17',
             '2018-06-18',
             '2018-06-19',
             '2018-06-20',
             '2018-06-21',
             '2018-06-24',
             '2018-06-25',
             '2018-06-26',
             '2018-06-27',
             '2018-06-28']


validate_days = ['2018-06-08',
                 '2018-06-15',
                 '2018-06-22',
                 '2018-06-29']
#%%
##test_days = ['2018-06-09',
#                 '2018-06-16',
#                 '2018-06-23',
#                 '2018-06-30']
#breakpoint()
trainfiles = Get_files(files,train_days)

random.shuffle(trainfiles)

#trainfiles = trainfiles[:15]

validatefiles = Get_files(files,validate_days)
random.shuffle(validatefiles)

#validatefiles = validatefiles[:15]
#breakpoint()
#Parameters used in Training Model are defined in Class File
Initial_parameters = ['2d','2t','cape','cape-1','cape-2','cape-3','cin','cp','crr','hcct','kx', 'lsp','lsrr',
              'slhf','sp','sshf','tcc','tcw','tcwv','totalx','z','Range','Hour']
#Fusion_Operatings=['Min','Mean','Median','Max','DynRng','StD','Entropy','Energy','Contrast','Homogeneity']
Fusion_Operatings=['Min']
Not_Fusing_Params=['TimeSlot','Lat','Lon','Member','Binary','Count','z','Range','Hour']
parameters=[]
for idx_parameters,column_name in enumerate(Initial_parameters):
    if not column_name in str(Not_Fusing_Params):
        for idx_Fused_Feature,Fused_Feature in enumerate(Fusion_Operatings):
            new_parameter=column_name+Fused_Feature
            parameters.extend([new_parameter])
    else:
        new_parameter=column_name
        parameters.extend([new_parameter])

print(str(len(parameters))+' Parameters are used:' + str(parameters))

print('Creating Scaler Function....This may take a while...')
ScalerGen = Scaler_Generator(trainfiles,parameters)
#
scaler = StandardScaler()
#
#
#  # Pass through all training files to fit scaler function
#
counter=0
print(counter)
print('trainfiles = '+str(len(trainfiles)))
for data in ScalerGen:
    counter+=1
    print('Normalizing '+str(counter)+' from '+str(len(trainfiles)))
    scaler.partial_fit(data)
#
#
joblib.dump(scaler, 'Storm_Model_Scaler')

#print('Loading Scaler')
#scaler = joblib.load('Storm_Model_Scaler')

TrainGen = Data_Generator(trainfiles,1,1,parameters, scaler)
ValidateGen = Data_Generator(validatefiles,1,1,parameters,scaler)

print('Building Model...')
#Build Model
model_hourly = models.Sequential()
model_hourly.add(layers.Dense(16, input_dim=len(parameters), activation='relu'))
model_hourly.add(layers.Dropout(0.2))
model_hourly.add(layers.Dense(16, activation='relu'))
model_hourly.add(layers.Dropout(0.2))
model_hourly.add(layers.Dense(1, activation='sigmoid'))

# # compile the keras model
model_hourly.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy','binary_accuracy'])

Folder_Name='NN_16_16'
#print('Loading Model...')
#model_hourly = models. load_model(Folder_Name +'/modelfinal-07-0.940.hdf5')
max_iter=10
for i in range(max_iter):

    print('Iteration: '+str(i+1)+ ' from '+str(max_iter))

    class_weight = {0: .1,
                1: 1.}

    checkpoint = ModelCheckpoint(Folder_Name +'/modelfinal-{epoch:02d}-{val_binary_accuracy:.5f}.hdf5', monitor='val_binary_crossentropy', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint]

    model_hourly.fit_generator(generator=TrainGen,
                   steps_per_epoch =(TrainGen.__len__()),
                   epochs = 10,
                    validation_data=ValidateGen,
                   verbose = 1,
                    class_weight=class_weight,
                    use_multiprocessing=False,
                    callbacks = callbacks_list,
                    max_queue_size = 1,
                    workers=1)



Run_Time=datetime.now()-start
print('Run Time= '+ str(Run_Time))