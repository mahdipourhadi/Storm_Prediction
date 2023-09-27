import glob
import pandas as pd
from StormPrediction_Class_Hadi import Get_files
from StormPrediction_Class_Hadi import Test_Generator
from keras import models
from sklearn import externals
import joblib
import os



#Point to folders with processed files, for example '**PATH/TO/FILE**/04H_2018-05-23_04_EPS_Storms_Hourly'
#files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
files = glob.glob(r'E:\Data\IsoBar_Fused_Files_V2\*')
print('Found '+str(len(files))+' Files')

#Select Days to use for testing, format 'YYYY-MM-DD'



test_days = ['2018-06-04',
             '2018-06-08',
             '2018-06-12',
             '2018-06-16',
             '2018-06-20',
             '2018-06-24',
             '2018-06-28']#,
test_days = ['2018-06-09',
             '2018-06-16',
             '2018-06-23',
             '2018-06-30']





#Parameter are defined in Class File
#parameters = ['2d','2t','cape','capem1','capem2','capem3','cin','crr','hcct','kx','lsp','lsrr','sp','slhf','sshf','tic','tcw','tcwv','totalx','z','Range','Hour']

Initial_parameters = ['2d','2t','cape','cape-1','cape-2','cape-3','cin','cp','crr','hcct','kx', 'lsp','lsrr',
              'slhf','sp','sshf','tcc','tcw','tcwv','totalx','z','Range','Hour']
#Fusion_Operatings=['Min','Mean','Median','Max','DynRng','StD','Entropy','Energy','Contrast','Homogeneity']
#Fusion_Operatings=['Mean','Median','Max','StD','Energy']
Fusion_Operatings=['Mean','Median','Max','StD','Energy']
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


#Load model output from training
NN_models=glob.glob('NN_16_16\\Fusion_'+Fusion_Operatings[0]+'\\modelfinal*.hdf5')
NN_models=glob.glob('NN_16_16\\Fusion_'+'Tops'+'\\modelfinal*.hdf5')
#NN_models.sort(key=os.path.getmtime)
NN_models=sorted(NN_models)

selected_model=NN_models[-1]
print( selected_model+ ' model is selected for ckassifying')
model = models.load_model(selected_model)
#model.save('test_model.hdf5')
#Load scaler function from training
scaler = joblib.load('NN_16_16\\Fusion_'+Fusion_Operatings[0]+'\\Storm_Model_Scaler'+Fusion_Operatings[0])
scaler = joblib.load('NN_16_16\\Fusion_'+'Tops'+'\\Storm_Model_Scaler'+'Tops')

Init_Test_Files=['12H_2018-06-04_36_EPS_DF',
 '12H_2018-06-28_00_EPS_DF',
 '07H_2018-06-24_19_EPS_DF',
 '18H_2018-06-12_06_EPS_DF',
 '01H_2018-06-16_01_EPS_DF',
 '02H_2018-06-04_26_EPS_DF',
 '14H_2018-06-16_02_EPS_DF',
 '20H_2018-06-28_32_EPS_DF',
 '07H_2018-06-16_07_EPS_DF',
 '18H_2018-06-16_30_EPS_DF',
 '03H_2018-06-08_03_EPS_DF',
 '22H_2018-06-04_22_EPS_DF',
 '01H_2018-06-08_25_EPS_DF',
 '10H_2018-06-28_10_EPS_DF',
 '04H_2018-06-24_04_EPS_DF',
 '02H_2018-06-28_26_EPS_DF',
 '16H_2018-06-24_16_EPS_DF',
 '22H_2018-06-24_10_EPS_DF',
 '12H_2018-06-20_36_EPS_DF',
 '19H_2018-06-28_19_EPS_DF',
 '21H_2018-06-04_33_EPS_DF',
 '04H_2018-06-20_28_EPS_DF',
 '14H_2018-06-20_26_EPS_DF',
 '23H_2018-06-28_35_EPS_DF',
 '15H_2018-06-08_15_EPS_DF',
 '09H_2018-06-04_09_EPS_DF',
 '10H_2018-06-08_22_EPS_DF',
 '17H_2018-06-20_29_EPS_DF',
 '12H_2018-06-24_00_EPS_DF',
 '03H_2018-06-08_15_EPS_DF',
 '21H_2018-06-04_09_EPS_DF',
 '21H_2018-06-20_21_EPS_DF',
 '09H_2018-06-28_33_EPS_DF',
 '08H_2018-06-08_32_EPS_DF',
 '14H_2018-06-08_02_EPS_DF',
 '08H_2018-06-04_20_EPS_DF',
 '13H_2018-06-12_25_EPS_DF',
 '19H_2018-06-28_07_EPS_DF',
 '14H_2018-06-16_26_EPS_DF',
 '15H_2018-06-28_03_EPS_DF',
 '16H_2018-06-16_04_EPS_DF',
 '06H_2018-06-16_06_EPS_DF',
 '08H_2018-06-24_08_EPS_DF',
 '21H_2018-06-08_33_EPS_DF',
 '15H_2018-06-16_27_EPS_DF',
 '06H_2018-06-12_06_EPS_DF',
 '08H_2018-06-08_20_EPS_DF',
 '01H_2018-06-20_01_EPS_DF',
 '23H_2018-06-04_11_EPS_DF',
 '20H_2018-06-24_20_EPS_DF']
testfiles=[]
for file in Init_Test_Files:
    testfiles.append('E:\\Data\\IsoBar_Fused_Files_V2\\'+file+'.h5')

#testfiles = Get_files(files,[d])
print(str(len(testfiles)) + ' Files for testing')

TestGen = Test_Generator(testfiles,parameters,model, scaler)

t = pd.concat([data for data in TestGen])
Accuracy = t.groupby(['Lat','Lon','TimeSlot','Range'])['y_actual','y_pred','Baseline','Count'].mean().reset_index()

#Accuracy.to_pickle('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_50files')
Accuracy.to_pickle('NN_16_16\\'+'Fusion_'+'Tops'+'\\Results_50files')
print('Results 50files Complete')
#Test.reset_index(inplace=True,drop=True)
#Accuracy.to_pickle('Results')

