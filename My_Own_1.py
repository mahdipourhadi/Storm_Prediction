#import os
import pandas
#import pandas_datareader
#import keras
import pandas as pd
import numpy as np

#load train data
data0 = {'Train_Files_Name':['23H_2018-05-25_23_Data','23H_2018-05-21_23_Data','23H_2018-06-01_23_Data','00H_2018-06-08_00_Data','15H_2018-07-27_15_Data']}#Insert manually all train data name
df0=pd.DataFrame(data0, columns=['Train_Files_Name'])
print('Hi, this is end of my own')