import glob
import pandas as pd
from StormPrediction_Class_Hadi_V2 import Get_files
from StormPrediction_Class_Hadi_V2 import Test_Generator
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





#%% Parameter s defining
#parameters = ['2d','2t','cape','capem1','capem2','capem3','cin','crr','hcct','kx','lsp','lsrr','sp','slhf','sshf','tic','tcw','tcwv','totalx','z','Range','Hour']

Initial_parameters = ['2d','2t','cape','cape-1','cape-2','cape-3','cin','cp','crr','hcct','kx', 'lsp','lsrr',
              'slhf','sp','sshf','tcc','tcw','tcwv','totalx','z','Range','Hour']
#Fusion_Operatings=['Min','Mean','Median','Max','DynRng','StD','Entropy','Energy','Contrast','Homogeneity']
Fusion_Operatings=['Min','Mean','Median','Max','DynRng','StD','Energy','Contrast','Homogeneity']
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
#NN_models=glob.glob('NN_16_16\\Fusion_'+Fusion_Operatings[0]+'\\modelfinal*.hdf5')
#NN_models=glob.glob('NN_16_16\\Fusion_'+'Tops'+'\\modelfinal*.hdf5')
NN_models=glob.glob('NN_16_16\\Fusion_'+'All'+'\\modelfinal*.hdf5')
#NN_models.sort(key=os.path.getmtime)
NN_models=sorted(NN_models)

selected_model=NN_models[-1]
print( selected_model+ ' model is selected for ckassifying')
model = models.load_model(selected_model)
#model.save('test_model.hdf5')
#Load scaler function from training
#scaler = joblib.load('NN_16_16\\Fusion_'+Fusion_Operatings[0]+'\\Storm_Model_Scaler'+Fusion_Operatings[0])
#scaler = joblib.load('NN_16_16\\Fusion_'+'Tops'+'\\Storm_Model_Scaler'+'Tops')
scaler = joblib.load('NN_16_16\\Fusion_'+'All'+'\\Storm_Model_Scaler'+'All')
Num_of_Reduced_Features=15
ipca = joblib.load('NN_16_16\\Fusion_'+'All'+'\\Storm_Model_Scaler'+'All'+'_PCA'+str(Num_of_Reduced_Features))


for d in test_days:
    testfiles = Get_files(files,[d])
    print(str(len(testfiles)) + ' Files for testing')

    TestGen = Test_Generator(testfiles,parameters,model, scaler, ipca)

    t = pd.concat([data for data in TestGen])
    Accuracy = t.groupby(['Lat','Lon','TimeSlot','Range'])['y_actual','y_pred','Baseline','Count'].mean().reset_index()

#    Accuracy.to_pickle('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_'+d)
#    Accuracy.to_pickle('NN_16_16\\'+'Fusion_'+'Tops'+'\\Results_'+d)
    Accuracy.to_pickle('NN_16_16\\'+'Fusion_'+'All'+'\\Results_'+d)
    print('Results_'+d+' Complete')
#Test.reset_index(inplace=True,drop=True)
#Accuracy.to_pickle('Results')

