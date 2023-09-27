## Differences to Aniel's Code
## V1: initial weitghs free classification

import glob
import random
from StormPrediction_Classv2 import Scaler_Generator
from StormPrediction_Classv2 import Data_Generator
from StormPrediction_Classv2 import Get_files
#from StormPrediction_Class import parameters

from sklearn.preprocessing import StandardScaler
import joblib

from keras import models, layers, optimizers
from keras.callbacks import ModelCheckpoint


from keras import backend as K

#Point to folders with processed files, for example '**PATH/TO/FILE**/#04H_2018-05-23_04_EPS_Storms_Hourly'
#files = glob.glob('/Users/aboulfazlsimorgh/Desktop/HadiWorkspace/Data/*')
#files = glob.glob('/mnt/dav/*Output/ML*/*')
files = glob.glob('D:\postdoc_hadi\ISOBAR\Data\*')

print('Found '+str(len(files))+' Files')


#Select Days to use for training/validating, format 'YYYY-MM-DD'


train_days = ['2018-06-07',
             '2018-06-08']#,
#             '2018-06-17',
#             '2018-06-18']


validate_days = ['2018-06-09']#,
#                '2018-06-19']

##test_days = ['2018-06-10',
##                '2018-06-20']

trainfiles = Get_files(files,train_days)
random.shuffle(trainfiles)

print(len(trainfiles))

trainfiles = trainfiles[:15]

validatefiles = Get_files(files,validate_days)
random.shuffle(validatefiles)

validatefiles = validatefiles[:15]

#Parameters used in Training Model are defined in Class File
parameters = ['2d','2t','cape','cape-1','cape-2','cape-3','cin','cp','crr','hcct','kx', 'lsp','lsrr',
              'slhf','sp','sshf','tcc','tcw','tcwv','totalx','z','Range','Hour']

print('Parameters used:' + str(parameters))

print('Creating Scaler Function....This may take a while...')
ScalerGen = Scaler_Generator(trainfiles,parameters)
#
#scaler = StandardScaler()
#
#
#  # Pass through all training files to fit scaler function
#
#for data in ScalerGen:

#    scaler.partial_fit(data)
#
#
#joblib.dump(scaler, 'Storm_Model_Scaler')

print('Loading Scaler')
scaler = joblib.load('Storm_Model_Scaler')

TrainGen = Data_Generator(trainfiles,2,25,parameters, scaler)

ValidateGen = Data_Generator(validatefiles,3,50,parameters,scaler)
print(vars(TrainGen))
print(dir(TrainGen))
print(TrainGen.parameters)
a=TrainGen.__getitem__
print('a content')
print(a)