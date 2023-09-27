#%% Importing

import os
import pandas as pd
from datetime import datetime,timedelta
import glob
import pygrib
from joblib import Parallel, delayed
import numpy as np
import keras
import joblib

#%% Days that we need to process GRB Forecasts


#days = ['20190618','20190619','20190620','20190621','20190622',
#       '20190723','20190724','20190725','20190726','20190727',
#       '20190823','20190824','20190825','20190826','20190827']

#%% Functions Defining

def create_col(grbs, member, param):
    x = pd.DataFrame(grbs(number=member, shortName=param)[0].values.flatten())
    x.columns = [param]
    return x


def getEPSMembers(file, member, Params, GEO):
    grbs = pygrib.open(file)
    Data = pd.DataFrame(grbs[1].latitudes)
    Data.columns = ['Lat']
    Data['Lon'] = pd.DataFrame(grbs[1].longitudes)
    Data['Member'] = member
    Data = pd.concat([Data, pd.concat([create_col(grbs, member, i) for i in Params], axis=1)], axis=1)
    Data = pd.merge(Data, GEO, on=['Lat', 'Lon'], how='left')
    return (Data)


def reduce_mem_usage(props, cols):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in cols:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def getEPS_DF(file, Params, GEO, n, outfolder):
    grbs = pygrib.open(file)
    AllData = pd.concat(Parallel(n_jobs=n)(delayed(getEPSMembers)(file, i, Params, GEO) for i in range(1, 51)))
    AllData['TimeSlot'] = datetime.strptime((str(grbs[1].date) + str(grbs[1].dataTime)).ljust(12, '0'),
                                            '%Y%m%d%H%M') + timedelta(hours=grbs[1].step)
    AllData['Range'] = grbs[1].step
    AllData.reset_index(inplace=True, drop=True)
    AllData.fillna(0, inplace=True)
    cols = AllData.columns.drop('TimeSlot')
    # Reduce Memory Size
    AllData, NAlist = reduce_mem_usage(AllData, cols)

    filename = str(AllData['TimeSlot'][0].hour).zfill(2) + 'H_' + str(AllData['TimeSlot'][0])[:10] + '_' + str(
        AllData['Range'][0]).zfill(2) + '_EPS_DF'

    AllData.to_pickle(outfolder + '/' + filename)
    print('Completed ' + filename)


def Get_Previous_Step_Param(file, step, param):
    if os.stat(file).st_size<120000000:
        Data = pd.read_pickle(file)
        print(file+' Read')
        if len(Data.columns) > 23:
            print(file + ' Completed')
    
        else:
            path_lenth = len(file) - len(file.split('/')[-1])
            path = file[:path_lenth]
            filename = file.split('/')[-1]
            file_time = filename[:14]
            file_step = int(filename[15:17])
            ending = filename[17:]
    
            steps = np.arange(1, step + 1)
            for i in steps:
                i = int(i)
                new_time = datetime.strptime(file_time, '%HH_%Y-%m-%d') - timedelta(hours=i)
                new_step = file_step - i
                new_filename = path + new_time.strftime('%HH_%Y-%m-%d') + '_' + str(new_step).zfill(2) + ending
                if new_filename in processed_files:
                    Data[param + '-' + str(i)] = pd.read_pickle(new_filename)[param]
                else:
                    Data[param + '-' + str(i)] = 0
    
            Data.to_pickle(file)
            print(file+' Completed')
    else:
        print('It seems '+file[-24:]+' had 23 params previousely, skipped!')
    return file + ' Complete'

#%% Point to files that I want to read


gribfiles = glob.glob('/mnt/dav/EPS/ECM*/2019-06-22*/*.grib')


gribfiles

#%%


path_out = '/mnt/dav/Codes_Output/Lightning_Data'

#path_out.replace('\\\\','\\')
#%% Get available parameters


grbs = pygrib.open(gribfiles[0])
Params = []
for grb in grbs:
    Params.append(grb.shortName)
Params = list(set(Params))

#%% Read Geopontentail Data, put file somewhere where your code can access


GEO = pd.read_pickle('/home/asap/Hadi/ISOBAR/Code/ReadGribsFiles/GEO')

n = 6 #number of processes for parallel computing

#%% This step gets 20 out of 23 parameters


[getEPS_DF(file,Params,GEO,n,path_out) for file in gribfiles]

#%% View the output files

processed_files = glob.glob(path_out+'/*')

processed_files.sort()

#%% This step gets the last 3 parameters (CAPE-1,CAPE-2,CAPE-3)


[Get_Previous_Step_Param(file, 3, 'cape') for file in processed_files]
#[print(file.replace('\\','\\\\')) for file in processed_files]
