## In The Name of God

#%%
import glob
import pandas as pd
import numpy as np
from scipy.stats import entropy
import os
np.seterr(divide='ignore', invalid='ignore')

#from pandas.compat.pickle_compat import _class_locations_map

#_class_locations_map.update({
#    ('pandas.core.internals.managers', 'BlockManager'): ('pandas.core.internals', 'BlockManager')
#})

#%%
files = glob.glob('/mnt/dav/Codes_Output/ML_Data/*')
#files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Data\*')
#files = glob.glob(r'Z:\Codes_Output\ML_Data\*')


print('Found '+str(len(files))+' Files')


def Contrast_Homogeneity_Computing(In_Matrix):
    Out_Vector=np.zeros([2,In_Matrix.shape[1]])
    for i in range(In_Matrix.shape[1]):
        In_Vector=In_Matrix[:,i]
        In_Vector=abs(In_Vector)
        if sum(In_Vector)<1e-6:
            Out_Vector[:,i]=0
        else:
            L=(len(In_Vector)+1)/2
            x=np.array(list(range(int(-(L-1)),int(L))))
            if len(In_Vector)%2==0:
                x=np.concatenate((x-0.5,[L-1]),axis=0)
            Out_Vector[0,i]=sum(np.multiply(x**2,In_Vector/sum(In_Vector)))/((L-1)**2)
            Out_Vector[1,i]=sum(np.divide(In_Vector/sum(In_Vector),(1+abs(x))))
    return Out_Vector
def Non_Zero_Mean_Vector(In_Vector):
    while np.mean(In_Vector)==0:
        In_Vector=In_Vector+np.random.normal(0,0.000001,len(In_Vector))
    return In_Vector
#Features_For_Fusion=['Mean','Min','Max','StD','Entropy']
Features_For_Fusion=['Min','Mean','Median','Max','DynRng','StD','Entropy','Energy','Contrast','Homogeneity']
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
                    if Fused_Feature=='Min':
                        new_DF[new_parameter]=np.min(tmp,axis=0)
                    elif Fused_Feature=='Mean':
                        new_DF[new_parameter]=np.mean(tmp,axis=0)
                    elif Fused_Feature=='Median':
                        new_DF[new_parameter]=np.median(tmp,axis=0)
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
                    elif Fused_Feature=='Energy':
                        new_DF[new_parameter]=np.sum(tmp**2,axis=0)
                    elif Fused_Feature=='Contrast':
                        Contrast_Homogeneity_Vecs=Contrast_Homogeneity_Computing(tmp)
                        new_DF[new_parameter]=Contrast_Homogeneity_Vecs[0,:]
                    elif Fused_Feature=='Homogeneity':
                        new_DF[new_parameter]=Contrast_Homogeneity_Vecs[1,:]
            else:
                new_parameter=column_name
                new_parameters.extend([new_parameter])
                Raw_TS_Data=DF[column_name]
                new_DF[new_parameter]=Raw_TS_Data[:int(DF.shape[0]/50)]
#        New_Path=each_file[:-24]+'Fused_Files_V2/'
        New_Path='/mnt/data/'+'IsoBar_Fused_Files_V2/'
#        New_File_Name=each_file[:-24]+'Fused_Files\\'+each_file[-24:]+'_Fused_V1'
        if not os.path.exists(New_Path):
            os.makedirs(New_Path)
        New_File_Name=New_Path+each_file[-24:]
        new_DF.to_hdf(New_File_Name+'.h5', key='new_DF',complevel=1,complib='bzip2')
#        new_DF=pd.read_hdf(r'E:\Data\IsoBar_Fused_Files_V2\00H_2018-06-08_00_EPS_DF.h5')
        print(New_File_Name+'.h5 has been written successfully.')
#        pd.to_pickle(new_DF, New_File_Name)
            
#                
#

