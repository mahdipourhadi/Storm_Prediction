
import glob
import pandas as pd
import numpy as np
from scipy.spatial import distance

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]

MF_p = pd.read_pickle(glob.glob(r'D:\postdoc_hadi\ISOBAR\Data\Lightning_Data*\*')[0])[['Member','Lat','Lon']]
MF_p = MF_p[MF_p['Member']==1]
MF_p.pop('Member')
#%% Create nodes
nodes = list(zip(MF_p['Lon'].astype(float).round(4),MF_p['Lat'].astype(float).round(4)))

#%% Lightning

# Z:\ISOBAR-Project\EarthNetworks

files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\ReadGribsFiles\Lightening\*')


Lightning = []
for f in files:
    a = pd.read_csv(f)
    print('Reading '+f+' File.')
#    breakpoint()
    if len(a.columns) == 1:
        DF = pd.concat([pd.DataFrame(a.loc[i].item().replace('"','').replace('','').split(',')).transpose() for i in range(0,len(a))])

        DF.columns = a.columns[0].replace('"','').replace('','').split(',')

    else:
        DF = a
    
#    breakpoint()
    DF['time_utc'] = pd.to_datetime(DF['time_utc'],format = '%Y-%m-%dT%H:%M:%S')
    
    Lightning.append(DF)

DF = pd.concat(Lightning)
#DF = DF.groupby([pd.Grouper(key='time_utc',freq='60min'),'latitude','longitude'])['type'].count().reset_index()

#%% Considering lightening in valid range only
DF['latitude']= DF['latitude'].astype(float)
DF['longitude']= DF['longitude'].astype(float)
DF['type']= DF['type'].astype(int)


MF_lightning = DF[(DF['latitude']<MF_p['Lat'].max())&
  (DF['latitude']>MF_p['Lat'].min())&
   (DF['longitude']<MF_p['Lon'].max())&
  (DF['longitude']>MF_p['Lon'].min())
  ]
MF_lightning.reset_index(inplace=True,drop=True)

MF_lightning['point'] = list(zip(MF_lightning.longitude,MF_lightning.latitude))


#%%
from datetime import datetime

MF_lightning['date'] = [x.date() for x in MF_lightning['time_utc']]
dates = MF_lightning['date'].unique()

#%%
def LightGrid(d,AE_lightning,nodes):
    Data = AE_lightning[AE_lightning['date']==d]
    Data['new_point']  = [closest_node(x,nodes) for x in Data['point']]
    Data['Lon'] = Data['new_point'].apply(lambda x:x[0])
    Data['Lat'] = Data['new_point'].apply(lambda x:x[1])

    Data = pd.DataFrame(Data.groupby([pd.Grouper(key='time_utc',freq='60min'),'Lon','Lat'])['type'].count())
    Data.reset_index(inplace=True)
    Data.columns = ['TimeSlot','Lon','Lat','LightningCount']
#    Data.to_pickle(str(d)+'_Lightning_AEMET')
    Data.to_pickle(str(d)+'_Lightning')
    
    
    
[LightGrid(d,MF_lightning,nodes) for d in dates]

#%%
            

                        
#for d in dates:
#    Data = MF_lightning[MF_lightning['date']==d]
#    Data['new_point']  = [closest_node(x,nodes) for x in Data['point']]
#    Data['Lon'] = Data['new_point'].apply(lambda x:x[0])
#    Data['Lat'] = Data['new_point'].apply(lambda x:x[1])
#
#    Data = pd.DataFrame(Data.groupby([pd.Grouper(key='time_utc',freq='60min'),'Lon','Lat'])['type'].count())
#    Data.reset_index(inplace=True)
#    Data.columns = ['TimeSlot','Lon','Lat','LightningCount']
#    Data.to_pickle(str(d)+'_Lightning')
#    print(str(d)+'_Lightning done!!!')