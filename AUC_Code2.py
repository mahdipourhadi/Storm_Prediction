#%%

##Commented lines:414,415,420,421

import pandas as pd
import glob
import matplotlib.pyplot as plt

import math
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


#Point to Result Files
#result_files = glob.glob('D:\postdoc_hadi\ISOBAR\Code\\NN_16_16\Results\Results_2018*')
Method_Name='Fusion'
#result_files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\Fusion\NN_16_16\Results\Results_2018*')
result_files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\Fusion\NN_16_16\Fusion_Mean\Results_2018*')

print('Found '+str(len(result_files))+' Files')

#Merge data into one dataframe
Data = []
for f in result_files:
    Data.append(pd.read_pickle(f))
Data = pd.concat(Data)
#"Data" contains all results


#%%
#Create figure
fig = plt.figure(figsize=(5,4))
fpr, tpr, threshold = metrics.roc_curve(Data['y_actual'],Data['Baseline'])


TH = 0.030#Baseline
roc_auc = metrics.auc(fpr, tpr)

#print(roc_auc)

plt.plot(fpr, tpr, 'k--', label = 'Baseline AUC: %0.3f' %roc_auc + ', fpr:  %0.3f' %fpr[np.argmax(threshold<TH)] + ', tpr:  %0.3f' %tpr[np.argmax(threshold<TH)],alpha=.6)
plt.scatter(fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)],marker = 'x',s=100,color='k')
#plt.show()

print('Baseline')
print(roc_auc)
print(fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)])

cm=confusion_matrix(Data['y_actual']>=TH, Data['y_pred']>=TH)
cm=confusion_matrix(fpr>=TH, tpr>=TH)
print(cm)
#plt.scatter(Threshold['fpr'].iloc[0],Threshold['tpr'].iloc[0],marker = 'x',s=100,color='k',label='Baseline Threshold = %0.3f' % Threshold['threshold'].iloc[0])

#Data['Baseline_CM'] = Data['Baseline'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)

if Method_Name=='Aniel':
    TH = 0.27
elif Method_Name=='PV1':
    TH = 0.27
elif Method_Name=='PV2':
    TH = 0.27
elif Method_Name=='PV3':
    TH = 0.24
elif Method_Name=='Fusion':
    TH = 0.1915

#Getting the area under the curve (AUC) value

fpr, tpr, threshold = metrics.roc_curve(Data['y_actual'],Data['y_pred'])


roc_auc = metrics.auc(fpr, tpr)






plt.plot(fpr, tpr, 'g--', label = Method_Name+' Model AUC: %0.3f' % roc_auc + ', fpr:  %0.3f' %fpr[np.argmax(threshold<TH)] + ', tpr:  %0.3f' %tpr[np.argmax(threshold<TH)],alpha=.6)
plt.scatter(fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)],marker = 'x',s=100,color='g')
fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Method_Name+' NN')
print(roc_auc)
print(fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)])
print(np.sqrt(fpr[np.argmax(threshold<TH)]**2+(1-tpr[np.argmax(threshold<TH)])**2))

############

plt.title('Receiver Operating Characteristic')
#TH=0.5
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(Method_Name+'vsBaseline_Acc_TPR_FPR.png', bbox_inches='tight')
plt.show()
#print(fpr.shape)
#print([min(fpr>=TH),max(fpr>=TH)])
#print(tpr.shape)
#print([min(fpr),max(tpr)])
cm=confusion_matrix(Data['y_actual']>=TH, Data['y_pred']>=TH)
cm=confusion_matrix(fpr>=TH, tpr>=TH)
print(cm)
#print(threshold)
#print(Data)
##
#%%

len(threshold)

#%%

fpr, tpr, threshold = metrics.roc_curve(Data['y_actual'],Data['Baseline'])


roc_auc = metrics.auc(fpr, tpr)


#%%

np.argmax(threshold<.6)

#%%

fig = plt.figure(figsize=(5,4))
fpr, tpr, threshold = metrics.roc_curve(Data[Data['Range']<=12]['y_actual'],Data[Data['Range']<=12]['y_pred'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'c--', label = 'Range less than 12 hours %0.3f'  % roc_auc,alpha=.6)

fpr, tpr, threshold = metrics.roc_curve(Data[(Data['Range']>12)&(Data['Range']<=24)]['y_actual'],Data[(Data['Range']>12)&(Data['Range']<=24)]['y_pred'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b--', label = 'Range between 12 and 24 hours %0.3f'  % roc_auc,alpha=.6)

fpr, tpr, threshold = metrics.roc_curve(Data[(Data['Range']>24)]['y_actual'],Data[Data['Range']>24]['y_pred'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'g--', label = 'Range between 24 and 36 hours %0.3f'  % roc_auc,alpha=.6)


plt.title(Method_Name+' Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(Method_Name+' Range Sensitivity.png', bbox_inches='tight')
plt.show()






#%%

fig = plt.figure(figsize=(5,4))
fpr, tpr, threshold = metrics.roc_curve(Data[Data['Range']<=12]['y_actual'],Data[Data['Range']<=12]['Baseline'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'c--', label = 'Range less than 12 hours %0.3f'  % roc_auc,alpha=.6)

fpr, tpr, threshold = metrics.roc_curve(Data[(Data['Range']>12)&(Data['Range']<=24)]['y_actual'],Data[(Data['Range']>12)&(Data['Range']<=24)]['Baseline'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b--', label = 'Range between 12 and 24 hours %0.3f'  % roc_auc,alpha=.6)

fpr, tpr, threshold = metrics.roc_curve(Data[(Data['Range']>24)]['y_actual'],Data[Data['Range']>24]['Baseline'])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'g--', label = 'Range between 24 and 36 hours %0.3f'  % roc_auc,alpha=.6)


plt.title('Baseline Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig('Baseline Range Sensitivity.png', bbox_inches='tight')
plt.show()




#%%

hist, bins = np.histogram(Data[Data['y_actual']==1]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.5,label='Convective Class', color='red')
hist, bins = np.histogram(Data[Data['y_actual']==0]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Method_Name+ ' Results')
plt.xlabel(Method_Name+' Model Prediction Score')
plt.ylabel('Density')
plt.grid()
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Normalized '+Method_Name+ ' Results.png', bbox_inches='tight')
plt.show()






#%%

hist, bins = np.histogram(Data[Data['y_actual']==1]['Baseline'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.5,label='Convective Class', color='red')
hist, bins = np.histogram(Data[Data['y_actual']==0]['Baseline'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized Baseline Results')
plt.xlabel('Baseline Model Prediction Score')
plt.ylabel('Density')
plt.grid()
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig('Normalized Baseline Results.png', bbox_inches='tight')
plt.show()



#%%

Data.reset_index(inplace=True,drop=True)

#%%

Data['NN_Error'] = Data['y_actual']-Data['y_pred']
Data['NN_Error'] = Data['NN_Error']**2

#%%

Data['B_Error'] = Data['y_actual']-Data['Baseline']
Data['B_Error'] = Data['B_Error']**2

#%%

Data.head()

#%%

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#%%

Data.sort_values(['TimeSlot','Range','Lat','Lon'],ascending=True,inplace=True)
Index = Data.groupby(['TimeSlot','Range']).count().reset_index()
Index = Index[['TimeSlot','Range']]
Count_Array = np.array(Data['Count']).reshape(len(Index),141,181)
yAct_Array = np.array(Data['y_actual']).reshape(len(Index),141,181)
Baseline_Array = np.array(Data['Baseline']).reshape(len(Index),141,181)
yPred_Array = np.array(Data['y_pred']).reshape(len(Index),141,181)
Err_Array = np.array(Data['NN_Error']).reshape(len(Index),141,181)
BErr_Array = np.array(Data['B_Error']).reshape(len(Index),141,181)
LatArray = np.array(Data['Lat']).reshape(len(Index),141,181)
LonArray = np.array(Data['Lon']).reshape(len(Index),141,181)

#%%

import datetime

#%%

Index['ReleaseTime'] = Index['TimeSlot'] - Index['Range'].apply(lambda x: datetime.timedelta(hours=x))

#%%

##Index.loc[196]

#%%

GIF = Index.sort_values(['ReleaseTime','Range'],ascending = [True,True]).reset_index()

#%%

print(Data['Lon'].min(), Data['Lon'].max(), Data['Lat'].min(), Data['Lat'].max())

#%%

#extent = [ 6, 20, 36, 54] # [left, right, bottom, top]
extent = [-10, 7, 32, 49]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Lat = LatArray[0,:,:]
Lon = LonArray[0,:,:]

#m = Basemap(projection='merc',llcrnrlat=Lat.min(),urcrnrlat=Lat.max(),\
 #                           llcrnrlon=Lon.min(),urcrnrlon=Lon.max(),lat_ts=20,resolution='l')
x, y = m(Lon, Lat)

#%%

fig = plt.figure(figsize=(7,12))

m.contourf(x, y, 1-yAct_Array[128,:,:], cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yAct_Array[128,:,:], cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
parallels = np.arange(36,54,2)
meridians = np.arange(5,20,5)
m.drawparallels(parallels)
m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
#plt.savefig('Normalized '+Method_Name+ ' Results.png', bbox_inches='tight')
#plt.show()





#%%

fig = plt.figure(figsize=(7,12))
cmap = plt.cm.jet
#m.contourf(x, y, Baseline_Array[129,:,:], cmap=plt.cm.jet, alpha=0)
levels = np.arange(0, 1, 0.025)
m.contourf(x, y, yPred_Array[128,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
plt.xlabel(Method_Name+' Model Prediction Score')
m.drawparallels(parallels)
m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.savefig(Method_Name+' Model Prediction Score.png', bbox_inches='tight')
plt.show()


#%%

fig = plt.figure(figsize=(7,12))

#m.contourf(x, y, 1-yAct_Array[128,:,:], cmap=plt.cm.Greys, alpha=.7)

m.contourf(x, y, Baseline_Array[128,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
plt.clim(0,.9)
plt.colorbar(extend='both',orientation='horizontal',ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03)
#plt.colorbar(ticks = [0,.15,.30,.45,.60,.75,.90,1],pad=.03,orientation='horizontal',extend='both',alpha=.6)
plt.xlabel('Baseline Model Prediction Score')
m.drawparallels(parallels)
m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.savefig('Baseline Model Prediction Score.png', bbox_inches='tight')
plt.show()
