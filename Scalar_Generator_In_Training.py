#%%
from datetime import datetime
start=datetime.now()




## Differences to Aniel's Code
## V2: V1 + Rearrange train, validation and test data to 4 1 1 respectively(Aniel's Code:
##     train validation test, 2 1 1)

import glob
import random
from StormPrediction_Class_Hadi import Scaler_Generator
from StormPrediction_Class_Hadi import Get_files
#from StormPrediction_Class import parameters

from sklearn.preprocessing import StandardScaler
import joblib


#%%
#Point to folders with processed files, for example '**PATH/TO/FILE**/#04H_2018-05-23_04_EPS_Storms_Hourly'
#files = glob.glob('/Users/aboulfazlsimorgh/Desktop/HadiWorkspace/Data/*')
#files = glob.glob('/mnt/dav/*Output/ML*/*')
#files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
#files = glob.glob('Z:\Codes_Output\ML_Data\*')
files = glob.glob('E:\Data\IsoBar_Fused_Files_V2\*')


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
Fusion_Operatings=['Homogeneity']
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
for data in ScalerGen:
    counter+=1
    print('Normalizing '+str(counter)+' from '+str(len(trainfiles)))
    scaler.partial_fit(data)
#
#
joblib.dump(scaler, 'Storm_Model_Scaler'+Fusion_Operatings[0])
