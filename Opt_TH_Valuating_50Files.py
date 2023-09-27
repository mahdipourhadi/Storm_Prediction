#%%

##Commented lines:414,415,420,421

import pandas as pd
import glob
import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics


#Point to Result Files
#result_files = glob.glob('D:\postdoc_hadi\ISOBAR\Code\\NN_16_16\Results\Results_2018*')
#Fusion_Operatings=['Min','Mean','Median','Max','DynRng','StD','Entropy','Energy','Contrast','Homogeneity']
Fusion_Operatings=['Tops']
Method_Name='Fusion'
#result_files = glob.glob('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_2018*16')
result_files = glob.glob('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_50files')
#result_files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\Fusion\Results2Comp\Results_Fusion\Results_2018*')
save_path='D:\postdoc_hadi\ISOBAR\Code\Fusion\Results2Comp\\'

print('Found '+str(len(result_files))+' Files')

#Merge data into one dataframe
#pd.read_hdf(ID)
Data = []
for f in result_files:
    Data.append(pd.read_pickle(f))
Data = pd.concat(Data)
#"Data" contains all results


#%%
fig = plt.figure(figsize=(5,4))

fpr, tpr, threshold = metrics.roc_curve(Data['y_actual'],Data['y_pred'])#Data['y_pred'])

roc_auc = metrics.auc(fpr, tpr)


Max_TH_Num=10000
Distance_10=np.ones(Max_TH_Num)
for TH_int in range(1,Max_TH_Num+1,1):
    TH=TH_int/Max_TH_Num
    Distance_10[TH_int-1]=np.sqrt(fpr[np.argmax(threshold<TH)]**2+(1-tpr[np.argmax(threshold<TH)])**2)
    if TH_int%1000==1:
        print(str(TH_int)+' from '+str(Max_TH_Num)+', Best distance from [1,0] so far: '+str(min(Distance_10)) +' (TH='+str((np.argmin(Distance_10)+1)/Max_TH_Num)+')')
#%%
TH=(np.argmin(Distance_10)+1)/Max_TH_Num
plt.plot(fpr, tpr, 'g--', label = Method_Name+' Model AUC: %0.4f' % roc_auc + ', fpr:  %0.4f' %fpr[np.argmax(threshold<TH)] + ', tpr:  %0.4f' %tpr[np.argmax(threshold<TH)],alpha=.6)
plt.scatter(fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)],marker = 'x',s=100,color='g')
fpr[np.argmax(threshold<TH)],tpr[np.argmax(threshold<TH)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Method_Name+Fusion_Operatings[0]+' NN')
print('TH = '+str(TH),', Best distance from [1,0] is: '+str(np.sqrt(fpr[np.argmax(threshold<TH)]**2+(1-tpr[np.argmax(threshold<TH)])**2)))
print('AUC = '+str(roc_auc))
print('TPR = '+str(tpr[np.argmax(threshold<TH)]),', FPR = '+str(fpr[np.argmax(threshold<TH)]))

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
plt.savefig(save_path+Method_Name+' '+Fusion_Operatings[0]+' Acc_TPR_FPR.png', bbox_inches='tight')
plt.show()
#print(fpr.shape)
#print([min(fpr>=TH),max(fpr>=TH)])
#print(tpr.shape)
#print([min(fpr),max(tpr)])
cm=metrics.confusion_matrix(Data['y_actual']>=TH, Data['y_pred']>=TH)
cm=metrics.confusion_matrix(fpr>=TH, tpr>=TH)
print('Confusion Matrix = ')
print(cm)
#print(threshold)
#print(Data)
##


fp_rate = 0.025

fpr, tpr, threshold = metrics.roc_curve(Data['y_actual'],Data['y_pred'])
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b--', label = Fusion_Operatings[0]+' '+Method_Name+' AUC: %0.4f' % roc_auc,alpha=.6)
idx = next(x[0] for x in enumerate(fpr) if x[1] > fp_rate)
plt.scatter(fpr[idx],tpr[idx],marker='X',color = 'b',label = Fusion_Operatings[0]+' '+Method_Name+' TH: %0.3f' % threshold[idx])
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+Method_Name+' '+Fusion_Operatings[0]+' Receiver Operating Characteristic.png', bbox_inches='tight')
plt.show()


precision, recall, threshold = metrics.precision_recall_curve(Data['y_actual'],Data['y_pred'])
pr_auc = metrics.auc(recall, precision)
plt.plot(recall, precision, 'b--', label = 'AUC: %0.3f' % pr_auc,alpha=.6)
plt.legend(loc = 'upper right')
plt.title('Precision - Recall')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.grid()
plt.savefig(save_path+Method_Name+' '+Fusion_Operatings[0]+' Precision - Recall.png', bbox_inches='tight')
plt.show()



df = pd.DataFrame()
df['y_actual']=Data['y_actual']
df['y_pred']=Data['y_pred']
print('pearson corr. =')
print(df.corr(method='pearson'))







from mpl_toolkits.basemap import Basemap
extent = [-18.75, 13, 28.25, 60]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Init_testfiles=['12H_2018-06-04_36_EPS_DF',
 '12H_2018-06-28_00_EPS_DF',
 '07H_2018-06-24_19_EPS_DF',
 '18H_2018-06-12_06_EPS_DF',
 '01H_2018-06-16_01_EPS_DF',
 '02H_2018-06-04_26_EPS_DF',
 '14H_2018-06-16_02_EPS_DF',
 '20H_2018-06-28_32_EPS_DF',
 '07H_2018-06-16_07_EPS_DF',
 '18H_2018-06-16_30_EPS_DF',
 '03H_2018-06-08_03_EPS_DF',
 '22H_2018-06-04_22_EPS_DF',
 '01H_2018-06-08_25_EPS_DF',
 '10H_2018-06-28_10_EPS_DF',
 '04H_2018-06-24_04_EPS_DF',
 '02H_2018-06-28_26_EPS_DF',
 '16H_2018-06-24_16_EPS_DF',
 '22H_2018-06-24_10_EPS_DF',
 '12H_2018-06-20_36_EPS_DF',
 '19H_2018-06-28_19_EPS_DF',
 '21H_2018-06-04_33_EPS_DF',
 '04H_2018-06-20_28_EPS_DF',
 '14H_2018-06-20_26_EPS_DF',
 '23H_2018-06-28_35_EPS_DF',
 '15H_2018-06-08_15_EPS_DF',
 '09H_2018-06-04_09_EPS_DF',
 '10H_2018-06-08_22_EPS_DF',
 '17H_2018-06-20_29_EPS_DF',
 '12H_2018-06-24_00_EPS_DF',
 '03H_2018-06-08_15_EPS_DF',
 '21H_2018-06-04_09_EPS_DF',
 '21H_2018-06-20_21_EPS_DF',
 '09H_2018-06-28_33_EPS_DF',
 '08H_2018-06-08_32_EPS_DF',
 '14H_2018-06-08_02_EPS_DF',
 '08H_2018-06-04_20_EPS_DF',
 '13H_2018-06-12_25_EPS_DF',
 '19H_2018-06-28_07_EPS_DF',
 '14H_2018-06-16_26_EPS_DF',
 '15H_2018-06-28_03_EPS_DF',
 '16H_2018-06-16_04_EPS_DF',
 '06H_2018-06-16_06_EPS_DF',
 '08H_2018-06-24_08_EPS_DF',
 '21H_2018-06-08_33_EPS_DF',
 '15H_2018-06-16_27_EPS_DF',
 '06H_2018-06-12_06_EPS_DF',
 '08H_2018-06-08_20_EPS_DF',
 '01H_2018-06-20_01_EPS_DF',
 '23H_2018-06-04_11_EPS_DF',
 '20H_2018-06-24_20_EPS_DF']
testfiles=[]
for file in Init_testfiles:
    testfiles.append('E:\\Data\\IsoBar_Fused_Files_V2\\'+file+'.h5')



for i in range(10,20):
    print(testfiles[i])
    Y_Actual=Data['y_actual'][Data['Lon']>=extent[0] and Data['Lon']<=extent[1] and Data['Lat']>=extent[2] and Data['Lat']<=extent[3]]
    fig = plt.figure(figsize=(20,15))
    plt.subplot2grid((1,2),(0,0),colspan=1)
    m.imshow(df['y_actual'][i*25521:(i+1)*25521].values.reshape(128,128))
    m.drawcoastlines()
    m.drawcountries()
    plt.title('Actual')

    plt.subplot2grid((1,2),(0,1),colspan=1)
    m.imshow(df['y_pred'][i*25521:(i+1)*25521].values.reshape(128,128),cmap=plt.cm.jet,vmax= TH) #This Threshold is based on the value provider by the ROC CURVE
    m.drawcoastlines()
    m.drawcountries()
    
    fpr, tpr, TH = metrics.roc_curve(df['y_actual'][i*25521:(i+1)*25521].values,df['y_pred'][i*25521:(i+1)*25521].values)
    roc_auc = metrics.auc(fpr, tpr)
    
    precision, recall, threshold = metrics.precision_recall_curve(df['y_actual'][i*25521:(i+1)*25521].values,df['y_pred'][i*25521:(i+1)*25521].values)
    pr_auc = metrics.auc(recall, precision)
    
    corr = pd.concat([df['y_actual'][i*25521:(i+1)*25521],df['y_pred'][i*25521:(i+1)*25521]],axis=1).corr(method='pearson').iloc[1,0]
   
    
    plt.title(Method_Name+' '+Fusion_Operatings[0]+' \n ROC AUC: {0:0.3f} \n PR AUC: {1:0.3f} \n Pearson: {2:0.3f}'.format(roc_auc,pr_auc,corr))