#%% Loading data and parameter initialization

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
#Methods_Name=["PV2","Aniel"]
Methods_Name=['Aniel','Proposed']
save_path='D:\postdoc_hadi\ISOBAR\Code\\'

for i,Met_Name in enumerate(Methods_Name):
    result_files = glob.glob('D:\postdoc_hadi\ISOBAR\Code\Old\Results_'+ Met_Name +'\Results_2018*')
    print('Found '+str(len(result_files))+' Files')
    
    #Merge data into one dataframe
    if i==0:
        Data1 = []
        for f in result_files:
            Data1.append(pd.read_pickle(f))
        Data1 = pd.concat(Data1)
        #"Data" contains all results
        fpr0, tpr0, threshold0 = metrics.roc_curve(Data1['y_actual'],Data1['Baseline'])
        fpr1, tpr1, threshold1 = metrics.roc_curve(Data1['y_actual'],Data1['y_pred'])
        roc_auc0 = metrics.auc(fpr0, tpr0)
        roc_auc1 = metrics.auc(fpr1, tpr1)
    elif i==1:
        Data2 = []
        for f in result_files:
            Data2.append(pd.read_pickle(f))
        Data2 = pd.concat(Data2)
        #"Data" contains all results
        fpr2, tpr2, threshold2 = metrics.roc_curve(Data2['y_actual'],Data2['y_pred'])
        roc_auc2 = metrics.auc(fpr2, tpr2)




#%%Receiver Operating Characteristic Figure Plotting
fig = plt.figure(figsize=(5,4))


TH0 = 0.038#Baseline:0


#print(roc_auc)

plt.plot(fpr0, tpr0, 'k--', label = 'Baseline AUC: %0.3f' %roc_auc0 + ', fpr:  %0.3f' %fpr0[np.argmax(threshold0<TH0)] + ', tpr:  %0.3f' %tpr0[np.argmax(threshold0<TH0)],alpha=.6)
plt.scatter(fpr0[np.argmax(threshold0<TH0)],tpr0[np.argmax(threshold0<TH0)],marker = 'x',s=100,color='k')
#plt.show()

print('Baseline')
print(roc_auc0)
print(fpr0[np.argmax(threshold0<TH0)],tpr0[np.argmax(threshold0<TH0)])

#plt.scatter(Threshold['fpr'].iloc[0],Threshold['tpr'].iloc[0],marker = 'x',s=100,color='k',label='Baseline Threshold = %0.3f' % Threshold['threshold'].iloc[0])

#Data['Baseline_CM'] = Data['Baseline'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)


TH1 = 0.23#Aniel:1
#10000*Distance from Ideal point [0,1] for TH from 0.2 to 0.3 with Step=0.01
#16507 16294 16167 16141 16158 16234 16390 16624 16921 16264 17682


TH2 = 0.25#PV2:2
#10000*Distance from Ideal point [0,1] for TH from 0.2 to 0.3 with Step=0.01
#16059 15739 15479 15290 15160 15121 15162 15285 15454 15719 16073




plt.plot(fpr1, tpr1, 'g--', label = Methods_Name[0]+' Model AUC: %0.3f' % roc_auc1 + ', fpr:  %0.3f' %fpr1[np.argmax(threshold1<TH1)] + ', tpr:  %0.3f' %tpr1[np.argmax(threshold1<TH1)],alpha=.6)
plt.scatter(fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)],marker = 'x',s=100,color='g')
fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Methods_Name[0]+' Model')
print(roc_auc1)
print(fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)])
print(((fpr1[np.argmax(threshold1<TH1)])**2+(1-tpr1[np.argmax(threshold1<TH1)])**2)**0.5)



plt.plot(fpr2, tpr2, 'r--', label = Methods_Name[1]+' Model AUC: %0.3f' % roc_auc2 + ', fpr:  %0.3f' %fpr2[np.argmax(threshold2<TH2)] + ', tpr:  %0.3f' %tpr2[np.argmax(threshold2<TH2)],alpha=.6)
plt.scatter(fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)],marker = 'x',s=100,color='r')
fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Methods_Name[1]+' Model')
print(roc_auc2)
print(fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)])
print(((fpr2[np.argmax(threshold2<TH2)])**2+(1-tpr2[np.argmax(threshold2<TH2)])**2)**0.5)

############

plt.title('Receiver Operating Characteristic')
#TH=0.5
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+'Models_Acc_TPR_FPR.png', bbox_inches='tight')
plt.show()
#print(fpr.shape)
#print([min(fpr>=TH),max(fpr>=TH)])
#print(tpr.shape)
#print([min(fpr),max(tpr)])
#cm=confusion_matrix(Data['y_actual']>=TH, Data['y_pred']>=TH)
#cm=confusion_matrix(fpr>=TH, tpr>=TH)
#print(cm)
#print(threshold)
#print(Data)
##


#%%Range Sensitivity Figure Plotting








fig = plt.figure(figsize=(5,4))
fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[Data1['Range']<=12]['y_actual'],Data1[Data1['Range']<=12]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_actual'],Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>24)]['y_actual'],Data1[Data1['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.3f'  % roc_auc012,alpha=.6)


plt.title('Baseline'+' Model Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+'Baseline'+' Model Range Sensitivity.png', bbox_inches='tight')
plt.show()





fig = plt.figure(figsize=(5,4))
fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[Data1['Range']<=12]['y_actual'],Data1[Data1['Range']<=12]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_actual'],Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>24)]['y_actual'],Data1[Data1['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.3f'  % roc_auc012,alpha=.6)


plt.title(Methods_Name[0]+' Model Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+Methods_Name[0]+' Model Range Sensitivity.png', bbox_inches='tight')
plt.show()









fig = plt.figure(figsize=(5,4))
fpr012, tpr012, threshold012 = metrics.roc_curve(Data2[Data2['Range']<=12]['y_actual'],Data2[Data2['Range']<=12]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data2[(Data2['Range']>12)&(Data2['Range']<=24)]['y_actual'],Data2[(Data2['Range']>12)&(Data2['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.3f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data2[(Data2['Range']>24)]['y_actual'],Data2[Data2['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.3f'  % roc_auc012,alpha=.6)


plt.title(Methods_Name[1]+' Model Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+Methods_Name[1]+' Model Range Sensitivity.png', bbox_inches='tight')
plt.show()











#%%Normalized Models Results Figure Plotting







hist, bins = np.histogram(Data1[Data1['y_actual']==1]['Baseline'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.5,label='Convective Class', color='red')
hist, bins = np.histogram(Data1[Data1['y_actual']==0]['Baseline'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+'Baseline'+ ' Model Results')
plt.xlabel('Baseline'+' Model Prediction Score')
plt.ylabel('Density')
plt.grid()
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(save_path+'Normalized '+'Baseline'+ ' Model Results.png', bbox_inches='tight')
plt.show()







hist, bins = np.histogram(Data1[Data1['y_actual']==1]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.5,label='Convective Class', color='red')
hist, bins = np.histogram(Data1[Data1['y_actual']==0]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[0]+ ' Model Results')
plt.xlabel(Methods_Name[0]+' Model Prediction Score')
plt.ylabel('Density')
plt.grid()
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(save_path+'Normalized '+Methods_Name[0]+ ' Model Results.png', bbox_inches='tight')
plt.show()





hist, bins = np.histogram(Data2[Data2['y_actual']==1]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.5,label='Convective Class', color='red')
hist, bins = np.histogram(Data2[Data2['y_actual']==0]['y_pred'],bins=20)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[1]+ ' Model Results')
plt.xlabel(Methods_Name[1]+' Model Prediction Score')
plt.ylabel('Density')
plt.grid()
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig(save_path+'Normalized '+Methods_Name[1]+ ' Model Results.png', bbox_inches='tight')
plt.show()







#%%Models Prediction Score Figure Plotting


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

Data1.reset_index(inplace=True,drop=True)

Data1['NN_Error'] = Data1['y_actual']-Data1['y_pred']
Data1['NN_Error'] = Data1['NN_Error']**2

Data1['B_Error'] = Data1['y_actual']-Data1['Baseline']
Data1['B_Error'] = Data1['B_Error']**2

Data1.head()




Data1.sort_values(['TimeSlot','Range','Lat','Lon'],ascending=True,inplace=True)
Index = Data1.groupby(['TimeSlot','Range']).count().reset_index()
Index = Index[['TimeSlot','Range']]
Count_Array = np.array(Data1['Count']).reshape(len(Index),141,181)
yAct_Array = np.array(Data1['y_actual']).reshape(len(Index),141,181)
Baseline_Array = np.array(Data1['Baseline']).reshape(len(Index),141,181)
yPred_Array = np.array(Data1['y_pred']).reshape(len(Index),141,181)
Err_Array = np.array(Data1['NN_Error']).reshape(len(Index),141,181)
BErr_Array = np.array(Data1['B_Error']).reshape(len(Index),141,181)
LatArray = np.array(Data1['Lat']).reshape(len(Index),141,181)
LonArray = np.array(Data1['Lon']).reshape(len(Index),141,181)


import datetime


Index['ReleaseTime'] = Index['TimeSlot'] - Index['Range'].apply(lambda x: datetime.timedelta(hours=x))


##Index.loc[196]


GIF = Index.sort_values(['ReleaseTime','Range'],ascending = [True,True]).reset_index()


print(Data1['Lon'].min(), Data1['Lon'].max(), Data1['Lat'].min(), Data1['Lat'].max())


#extent = [ 6, 20, 36, 54] # [left, right, bottom, top]
extent = [-10, 7, 32, 49]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Lat = LatArray[0,:,:]
Lon = LonArray[0,:,:]

#m = Basemap(projection='merc',llcrnrlat=Lat.min(),urcrnrlat=Lat.max(),\
 #                           llcrnrlon=Lon.min(),urcrnrlon=Lon.max(),lat_ts=20,resolution='l')
x, y = m(Lon, Lat)


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
#plt.savefig(save_path+'Normalized '+Methods_Name[0]+ ' Results.png', bbox_inches='tight')
#plt.show()



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
plt.xlabel(Methods_Name[0]+' Model Prediction Score')
m.drawparallels(parallels)
m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[0]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[0]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()





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
plt.savefig(save_path+'Baseline Model Prediction Score.png', bbox_inches='tight')
plt.show()









Data2.reset_index(inplace=True,drop=True)

Data2['NN_Error'] = Data2['y_actual']-Data2['y_pred']
Data2['NN_Error'] = Data2['NN_Error']**2

Data2['B_Error'] = Data2['y_actual']-Data2['Baseline']
Data2['B_Error'] = Data2['B_Error']**2

Data2.head()




Data2.sort_values(['TimeSlot','Range','Lat','Lon'],ascending=True,inplace=True)
Index = Data2.groupby(['TimeSlot','Range']).count().reset_index()
Index = Index[['TimeSlot','Range']]
Count_Array = np.array(Data2['Count']).reshape(len(Index),141,181)
yAct_Array = np.array(Data2['y_actual']).reshape(len(Index),141,181)
Baseline_Array = np.array(Data2['Baseline']).reshape(len(Index),141,181)
yPred_Array = np.array(Data2['y_pred']).reshape(len(Index),141,181)
Err_Array = np.array(Data2['NN_Error']).reshape(len(Index),141,181)
BErr_Array = np.array(Data2['B_Error']).reshape(len(Index),141,181)
LatArray = np.array(Data2['Lat']).reshape(len(Index),141,181)
LonArray = np.array(Data2['Lon']).reshape(len(Index),141,181)




Index['ReleaseTime'] = Index['TimeSlot'] - Index['Range'].apply(lambda x: datetime.timedelta(hours=x))


##Index.loc[196]


GIF = Index.sort_values(['ReleaseTime','Range'],ascending = [True,True]).reset_index()


print(Data2['Lon'].min(), Data2['Lon'].max(), Data2['Lat'].min(), Data2['Lat'].max())


#extent = [ 6, 20, 36, 54] # [left, right, bottom, top]
extent = [-10, 7, 32, 49]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Lat = LatArray[0,:,:]
Lon = LonArray[0,:,:]

#m = Basemap(projection='merc',llcrnrlat=Lat.min(),urcrnrlat=Lat.max(),\
 #                           llcrnrlon=Lon.min(),urcrnrlon=Lon.max(),lat_ts=20,resolution='l')
x, y = m(Lon, Lat)


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
#plt.savefig(save_path+'Normalized '+Methods_Name[1]+ ' Results.png', bbox_inches='tight')
#plt.show()





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
plt.xlabel(Methods_Name[1]+' Model Prediction Score')
m.drawparallels(parallels)
m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[1]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[1]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()

