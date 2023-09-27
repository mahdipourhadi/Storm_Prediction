import glob
import pandas as pd
from StormPrediction_Classv2 import Get_files
#from StormPrediction_Class import parameters
from StormPrediction_Classv2 import Test_Generator
from keras import models
from sklearn import externals
import joblib




#Point to folders with processed files, for example '**PATH/TO/FILE**/04H_2018-05-23_04_EPS_Storms_Hourly'
files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
print('Found '+str(len(files))+' Files')

#Select Days to use for testing, format 'YYYY-MM-DD'


test_days = ['2018-06-09',
             '2018-06-16',
             '2018-06-23',
             '2018-06-30']#,
#'2018-06-20']






#Parameter are defined in Class File
#parameters = ['2d','2t','cape','capem1','capem2','capem3','cin','crr','hcct','kx','lsp','lsrr','sp','slhf','sshf','tic','tcw','tcwv','totalx','z','Range','Hour']

parameters = ['2d','2t','cape','cape-1','cape-2','cape-3','cin','cp','crr','hcct','kx', 'lsp','lsrr',
'slhf','sp','sshf','tcc','tcw','tcwv','totalx','z','Range','Hour']

#Load model output from training

model = models.load_model('NN_16_16/modelfinal-04-0.9273.hdf5')

#Load scaler function from training
scaler = joblib.load('Storm_Model_Scaler')


for d in test_days:
    testfiles = Get_files(files,[d])
    print(str(len(testfiles)) + ' Files for testing')

    TestGen = Test_Generator(testfiles,parameters,model, scaler)

    t = pd.concat([data for data in TestGen])
    Accuracy = t.groupby(['Lat','Lon','TimeSlot','Range'])['y_actual','y_pred','Baseline','Count'].mean().reset_index()

    Accuracy.to_pickle('NN_16_16/Results/Results_'+d)
    print('Results_'+d+' Complete')
#Test.reset_index(inplace=True,drop=True)
#Accuracy.to_pickle('Results')

