
import os
import pandas as pd
import numpy as np
import pygrib
import glob
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.path as mpltPath
#%%

#Change the path
#Storms = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\ReadGribsFiles\Storms\Storms2019*')
#Storms.sort()
#%%

#Change the path
#grbs = pygrib.open(glob.glob(r'Z:\EPS\ECM*\2019-06-*\*.grib')[0])
#Lons = grbs[1].longitudes
#Lats = grbs[1].latitudes
#Coords = pd.DataFrame(Lats,Lons).reset_index()
#Coords.columns = ['Lon','Lat']
#Coords['Lon'] = Coords['Lon'].astype(float).round(5)
#Coords['Lat'] = Coords['Lat'].astype(float).round(5)
#Coords.drop_duplicates(inplace=True)
#Coords.head()
#%%

def GetPoints(Coords, StormsDF):
    
    Points = []
    StormsDF.reset_index(inplace=True,drop=True)
    StormsDF['latmin']=StormsDF.apply(lambda x: min(list(x.LatContour1)+ list(x.LatContour)),axis=1)
    StormsDF['latmax']=StormsDF.apply(lambda x: max(list(x.LatContour1)+ list(x.LatContour)),axis=1)
    StormsDF['lonmin']=StormsDF.apply(lambda x: min(list(x.LonContour1)+ list(x.LonContour)),axis=1)
    StormsDF['lonmax']=StormsDF.apply(lambda x: max(list(x.LonContour1)+ list(x.LonContour)),axis=1)
    StormsDF = StormsDF[(StormsDF['latmin'] < Coords['Lat'].max() )&
                       (StormsDF['latmax'] > Coords['Lat'].min())&
                       (StormsDF['lonmin'] < Coords['Lon'].max())&
                       (StormsDF['lonmax'] > Coords['Lon'].min())]
    
    StormsDF.reset_index(inplace=True,drop=True)
    
    for j in range(0,len(StormsDF)):
        points = Coords
        points = points[(points['Lat'] <= StormsDF['latmax'][j])&
                        (points['Lat'] >= StormsDF['latmin'][j])&
                        (points['Lon'] <= StormsDF['lonmax'][j])&
                        (points['Lon'] >= StormsDF['lonmin'][j])]
        points.reset_index(inplace=True,drop=True)
        points_zip = list(zip(points['Lon'],points['Lat']))
        
        x1, y1 = np.array(pd.DataFrame(StormsDF['Poly'][j][:])[0]), np.array(pd.DataFrame(StormsDF['Poly'][j][:])[1])
        xy = list(zip(x1,y1))
        path = mpltPath.Path(xy)
        if len(points_zip)>0:
            
    #        breakpoint()
    #        print('j='+str(j))
    #        print('points:')
    #        breakpoint()
    #        print(points[path.contains_points(points_zip)].reset_index(drop=True))
    #        breakpoint()
            TF = points[path.contains_points(points_zip)].reset_index(drop=True)
            if StormsDF['LatContour1'][j].size > 0:
                x1, y1 = np.array(pd.DataFrame(StormsDF['Poly1'][j][:])[0]), np.array(pd.DataFrame(StormsDF['Poly1'][j][:])[1])
                xy = list(zip(x1,y1))
                path = mpltPath.Path(xy)
                TF1 = points[path.contains_points(points_zip)].reset_index(drop=True)
                TF = pd.concat([TF,TF1])
                TF.drop_duplicates(inplace=True)
            TF['CTP'] = StormsDF['CTPressure'][j]
            TF['Severity'] = StormsDF['Severity'][j] + 1
            TF['TimeSlot'] = StormsDF['TimeSlot']
            #TF.groupby(['TimeSlot','Lon','Lat']).agg({'CTP':'min','Severity':'max'}).reset_index(inplace=True)
            Points.append(TF)
    Points = pd.concat(Points)
    Points = Points.groupby(['TimeSlot','Lon','Lat']).agg({'CTP':'min','Severity':'max'}).reset_index()
    #Points.drop_duplicates(inplace=True)
    
    return Points


def Target(file,Coords):
    Storms = pd.read_pickle(file)
    print('Reading '+file[-8:])
    Storms['Time'] = Storms['TimeSlot'].apply(lambda x: x - timedelta(minutes = x.minute))
    Storms.reset_index(drop=True,inplace=True)
    Target = GetPoints(Coords,Storms)
    Target.to_pickle('Storms_Sattlite_'+file[-8:])   #Change Filename
    return print('Completed '+file[-8:])

def Merge_Storm_Inf(Storm_Day_File,Data_Bef_Storm_Dir):
    print('Reading '+Storm_Day_File[-8:])
    DF_Storms = pd.read_pickle(Storm_Day_File)
    for h in range(24):
        Data_filenames=glob.glob(Data_Bef_Storm_Dir + '\\' + str(h).zfill(2)+'H_'+Storm_Day_File[-8:-4]+'-'+Storm_Day_File[-4:-2]+'-'+Storm_Day_File[-2:]+'*')
        Data_filenames.sort()

        if sum(DF_Storms['TimeSlot'].dt.hour==h)>0:
        #    Data_filenames=glob.glob(Data_Bef_Storm_Dir + '\\' + Storm_in_hour['TimeSlot'][0].strftime("%H")+'H_'+Storm_Day_File[-8:-4]+'-'+Storm_Day_File[-4:-2]+'-'+Storm_Day_File[-2:])
        #    Data_filenames.sort()
            Storm_in_hour=DF_Storms.loc[DF_Storms['TimeSlot'].dt.hour==h]
            Storm_in_hour.reset_index(drop=True, inplace=True)
            for Data_File in Data_filenames:
            
                if os.stat(Data_File).st_size<155000000:
                    DF_Data=pd.read_pickle(Data_File)
                    DF_Data['StormBinary']=np.zeros(len(DF_Data))
                    DF_Data['StormSeverity']=np.zeros(len(DF_Data))
                    DF_Data['StormCTP']=np.zeros(len(DF_Data))
                    for i in range(Storm_in_hour.shape[0]):
                        DF_Data['StormBinary'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]=1
                        if Storm_in_hour['Severity'][i]>np.mean(DF_Data['StormSeverity'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]):
                            DF_Data['StormSeverity'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]=Storm_in_hour['Severity'][i]
                        if np.mean(DF_Data['StormCTP'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values])<0.01:
                            DF_Data['StormCTP'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]=Storm_in_hour['CTP'][i]
                        elif Storm_in_hour['CTP'][i]<np.mean(DF_Data['StormCTP'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]):
                            DF_Data['StormCTP'][DF_Data[(DF_Data['Lat']==Storm_in_hour['Lat'][i]) & (DF_Data['Lon']==Storm_in_hour['Lon'][i])].index.values]=Storm_in_hour['CTP'][i]
    
                    DF_Data.to_pickle(Data_File)
                    print('Storm inf added to '+Data_File[-24:])
                else:
                    print('Storm inf is added to '+Data_File[-24:] +' previousely!')
        else:
            Storm_in_hour=pd.DataFrame()
            for Data_File in Data_filenames:
                if os.stat(Data_File).st_size<155000000:
                    DF_Data=pd.read_pickle(Data_File)
                    DF_Data['StormBinary']=np.zeros(len(DF_Data))
                    DF_Data['StormSeverity']=np.zeros(len(DF_Data))
                    DF_Data['StormCTP']=np.zeros(len(DF_Data))
                    DF_Data.to_pickle(Data_File)
                    print('Storm inf added to '+Data_File[-24:])
                else:
                    print('Storm inf is added to '+Data_File[-24:] +' previousely!')

            
    return print('Completed '+Storm_Day_File[-8:])

#Run Functions, will save file to working directory, the file will contain the coordinate and severity.
#[Target(i,Coords) for i in Storms]
#%% read data to merge
#Before merging with Forecast data, you need to create binary variable and group by hour.
#[Merge_Storm_Inf(i,Coords) for i in Storms]

#Storms_Sattlite_Files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\ReadGribsFiles\Storms\Storms_Sattlite_2019*')
Storms_Sattlite_Files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\ReadGribsFiles\Storms\Storms_Sattlite_2019*')
Storms_Sattlite_Files.sort()

Data_Bef_Storm_Dir=r'Z:\Codes_Output\Lightning_Data'
#Data_Bef_Storm_Dir='D:\postdoc_hadi\ISOBAR\Data\Lightning_Data'



[Merge_Storm_Inf(i,Data_Bef_Storm_Dir) for i in Storms_Sattlite_Files]