## In The Name of God

#%%
import glob

#files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Data\*')


print('Found '+str(len(files))+' Files')
#%%
import pandas as pd
import numpy as np
from scipy.stats import entropy
import os

#Features_For_Fusion=['Mean','Min','Max','StD','Entropy']
Features_For_Fusion=['Mean','Min','Max','DynRng','StD','Entropy']
Not_Fusing_Params=['TimeSlot','Lat','Lon','Member','Binary','Count','z','Range']
for each_file in files:
    if not 'Fused' in str(each_file):
        print(each_file)
        DF = pd.read_pickle(each_file)
        parameters=DF.columns
        new_parameters=[]
        new_DF=pd.DataFrame()
        for idx_parameters,column_name in enumerate(parameters):
            if not column_name in str(Not_Fusing_Params):
                for idx_Fused_Feature,Fused_Feature in enumerate(Features_For_Fusion):
                    new_parameter=column_name+Fused_Feature
                    new_parameters.extend([new_parameter])
                    Raw_Array_Data=DF[column_name].to_numpy()
                    tmp=np.reshape(Raw_Array_Data, (50,int(DF.shape[0]/50)))
                    if Fused_Feature=='Mean':
                        new_DF[new_parameter]=np.mean(tmp,axis=0)
                    elif Fused_Feature=='Min':
                        new_DF[new_parameter]=np.min(tmp,axis=0)
                    elif Fused_Feature=='Max':
                        new_DF[new_parameter]=np.max(tmp,axis=0)
                    elif Fused_Feature=='DynRng':
                        DynRng=(np.max(tmp,axis=0)-np.min(tmp,axis=0))
                        if DynRng.max()<=0:
                            print('Not useful feature for fusion: Dynamic range of '+column_name+' feature in '+each_file[-24:]+' file is Zero !!!')
#                        assert DynRng.max()>0
                        new_DF[new_parameter]=DynRng
                    elif Fused_Feature=='StD':
                        new_DF[new_parameter]=np.std(tmp,axis=0)
                    elif Fused_Feature=='Entropy':
                        new_DF[new_parameter]=np.nan_to_num(entropy(tmp,axis=0,base=2))
            else:
                new_parameter=column_name
                new_parameters.extend([new_parameter])
                Raw_TS_Data=DF[column_name]
                new_DF[new_parameter]=Raw_TS_Data[:int(DF.shape[0]/50)]
        New_Path=each_file[:-24]+'Fused_Files_V2\\'
#        New_File_Name=each_file[:-24]+'Fused_Files\\'+each_file[-24:]+'_Fused_V1'
        if not os.path.exists(New_Path):
            os.makedirs(New_Path)
        New_File_Name=New_Path+each_file[-24:]
        new_DF.to_hdf(New_File_Name+'.h5', key='new_DF',complevel=1,complib='bzip2')
#        pd.to_pickle(new_DF, New_File_Name)
            
#                
#

