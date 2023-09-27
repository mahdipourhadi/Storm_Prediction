import glob
import pandas as pd
import datetime
from datetime import timedelta
import numpy as np

Just_Ligthening_Files=glob.glob('/home/asap/Hadi/ISOBAR/Code/ReadGribsFiles/2019-06*_Lightning')
Just_Ligthening_Files.sort()
#Just_Ligthening_Files=Just_Ligthening_Files[4:]
Input_Raw_Dir='/mnt/dav/Codes_Output/Lightning_Data/*'
for f in  Just_Ligthening_Files:
    print('Adding Lightening information of '+f[-20:-10]+ ' !!!')
    DF_Lightening=pd.read_pickle(f)
    for h in range(24):
        Lightening_Date=datetime.datetime.strptime(f[-20:-10],"%Y-%m-%d")+ timedelta(hours=h)
        Input_Files_List=glob.glob(Input_Raw_Dir+ str(h).zfill(2)+'H_'+f[-20:-10]+'*')
        Input_Files_List.sort()
        for raw_f in Input_Files_List:
            DF_Data=pd.read_pickle(raw_f)
            DF_Data['LighteningBinary']=np.zeros(len(DF_Data))
            DF_Data['LighteningSeverity']=np.zeros(len(DF_Data))
            if sum(DF_Lightening['TimeSlot']==Lightening_Date)>0:
                Lightened_Points=DF_Lightening[DF_Lightening['TimeSlot']==Lightening_Date]
                Lightened_Points.reset_index(inplace=True,drop=True)
                for i in range(len(Lightened_Points)):
                    DF_Data['LighteningBinary'][(DF_Data['Lon']==Lightened_Points['Lon'][i])&
                            (DF_Data['Lat']==Lightened_Points['Lat'][i])]=1
                    DF_Data['LighteningSeverity'][(DF_Data['Lon']==Lightened_Points['Lon'][i])&
                            (DF_Data['Lat']==Lightened_Points['Lat'][i])]=Lightened_Points['LightningCount'][i]
            DF_Data.to_pickle(raw_f)
                
