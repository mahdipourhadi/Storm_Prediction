#%% Loading data and parameter initialization

##Commented lines:414,415,420,421

import pandas as pd
import glob
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import metrics


#Point to Result Files
#result_files = glob.glob('D:\postdoc_hadi\ISOBAR\Code\\NN_16_16\Results\Results_2018*')
#Methods_Name=["PV2","Aniel"]
Methods_Name=['Method-1','Fusion-Mean','Fusion-All','Fusion-Tops']
save_path='D:\postdoc_hadi\ISOBAR\Code\Fusion\Results2Comp\\'

for i,Met_Name in enumerate(Methods_Name):
    result_files = glob.glob('D:\postdoc_hadi\ISOBAR\Code\Fusion\Results2Comp\Results_'+ Met_Name +'\Results_2018*')
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
        TH0 = 0.0001#Baseline:0
        TH1 = 0.23#Aniel:1
        #10000*Distance from Ideal point [0,1] for TH from 0.2 to 0.3 with Step=0.01
        #16507 16294 16167 16141 16158 16234 16390 16624 16921 16264 17682
    elif i==1:
        Data2 = []
        for f in result_files:
            Data2.append(pd.read_pickle(f))
        Data2 = pd.concat(Data2)
        #"Data" contains all results
        fpr2, tpr2, threshold2 = metrics.roc_curve(Data2['y_actual'],Data2['y_pred'])
        roc_auc2 = metrics.auc(fpr2, tpr2)
        TH2 = 0.2043#ATM2021
        #10000*Distance from Ideal point [0,1] for TH from 0.2 to 0.3 with Step=0.01
        #16059 15739 15479 15290 15160 15121 15162 15285 15454 15719 16073
    elif i==2:
        Data3 = []
        for f in result_files:
            Data3.append(pd.read_pickle(f))
        Data3 = pd.concat(Data3)
        #"Data" contains all results
        fpr3, tpr3, threshold3 = metrics.roc_curve(Data3['y_actual'],Data3['y_pred'])
        roc_auc3 = metrics.auc(fpr3, tpr3)
        TH3 = 0.2033#0.1915
        #10000*Distance from Ideal point [0,1] for TH from 0.2 to 0.3 with Step=0.01
        #16059 15739 15479 15290 15160 15121 15162 15285 15454 15719 16073
    elif i==3:
        Data4 = []
        for f in result_files:
            Data4.append(pd.read_pickle(f))
        Data4 = pd.concat(Data4)
        #"Data" contains all results
        fpr4, tpr4, threshold4 = metrics.roc_curve(Data4['y_actual'],Data4['y_pred'])
        roc_auc4 = metrics.auc(fpr4, tpr4)        
        TH4 = 0.19




#%%Receiver Operating Characteristic Figure Plotting
fig = plt.figure(figsize=(5,4))




#print(roc_auc)

plt.plot(fpr0, tpr0, 'g--', label = 'Baseline AUC: %0.4f' %roc_auc0 + ', fpr:  %0.4f' %fpr0[np.argmax(threshold0<TH0)] + ', tpr:  %0.4f' %tpr0[np.argmax(threshold0<TH0)],alpha=.6)
plt.scatter(fpr0[np.argmax(threshold0<TH0)],tpr0[np.argmax(threshold0<TH0)],marker = 'x',s=100,color='k')
#plt.show()

print('Baseline')
print(roc_auc0)
print(fpr0[np.argmax(threshold0<TH0)],tpr0[np.argmax(threshold0<TH0)])

#plt.scatter(Threshold['fpr'].iloc[0],Threshold['tpr'].iloc[0],marker = 'x',s=100,color='k',label='Baseline Threshold = %0.4f' % Threshold['threshold'].iloc[0])

#Data['Baseline_CM'] = Data['Baseline'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)





plt.plot(fpr1, tpr1, 'k--', label = Methods_Name[0]+' Model AUC: %0.4f' % roc_auc1 + ', fpr:  %0.4f' %fpr1[np.argmax(threshold1<TH1)] + ', tpr:  %0.4f' %tpr1[np.argmax(threshold1<TH1)],alpha=.6)
plt.scatter(fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)],marker = 'x',s=100,color='g')
fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Methods_Name[0]+' Model')
print(roc_auc1)
print(fpr1[np.argmax(threshold1<TH1)],tpr1[np.argmax(threshold1<TH1)])
print(((fpr1[np.argmax(threshold1<TH1)])**2+(1-tpr1[np.argmax(threshold1<TH1)])**2)**0.5)



plt.plot(fpr2, tpr2, 'r--', label = Methods_Name[1]+' Model AUC: %0.4f' % roc_auc2 + ', fpr:  %0.4f' %fpr2[np.argmax(threshold2<TH2)] + ', tpr:  %0.4f' %tpr2[np.argmax(threshold2<TH2)],alpha=.6)
plt.scatter(fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)],marker = 'x',s=100,color='r')
fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Methods_Name[1]+' Model')
print(roc_auc2)
print(fpr2[np.argmax(threshold2<TH2)],tpr2[np.argmax(threshold2<TH2)])
print(((fpr2[np.argmax(threshold2<TH2)])**2+(1-tpr2[np.argmax(threshold2<TH2)])**2)**0.5)




plt.plot(fpr3, tpr3, 'b--', label = Methods_Name[2]+' Model AUC: %0.4f' % roc_auc3 + ', fpr:  %0.4f' %fpr3[np.argmax(threshold3<TH3)] + ', tpr:  %0.4f' %tpr3[np.argmax(threshold3<TH3)],alpha=.6)
plt.scatter(fpr3[np.argmax(threshold3<TH3)],tpr3[np.argmax(threshold3<TH3)],marker = 'x',s=100,color='b')
fpr3[np.argmax(threshold3<TH3)],tpr3[np.argmax(threshold3<TH3)]
print(Methods_Name[2]+' Model')
print(roc_auc3)
print(fpr3[np.argmax(threshold3<TH3)],tpr3[np.argmax(threshold3<TH3)])
print(((fpr3[np.argmax(threshold3<TH3)])**2+(1-tpr3[np.argmax(threshold3<TH3)])**2)**0.5)



plt.plot(fpr4, tpr4, 'c--', label = Methods_Name[3]+' Model AUC: %0.4f' % roc_auc4 + ', fpr:  %0.4f' %fpr4[np.argmax(threshold4<TH4)] + ', tpr:  %0.4f' %tpr4[np.argmax(threshold4<TH4)],alpha=.6)
plt.scatter(fpr4[np.argmax(threshold4<TH4)],tpr4[np.argmax(threshold4<TH4)],marker = 'x',s=100,color='r')
fpr4[np.argmax(threshold4<TH4)],tpr4[np.argmax(threshold4<TH4)]
#Accuracy2['ypred_CM'] = Accuracy2['y_pred'].apply(lambda x: 0 if x <Threshold['threshold'].iloc[0] else 1)
print(Methods_Name[3]+' Model')
print(roc_auc4)
print(fpr4[np.argmax(threshold4<TH4)],tpr4[np.argmax(threshold4<TH4)])
print(((fpr4[np.argmax(threshold4<TH4)])**2+(1-tpr4[np.argmax(threshold4<TH4)])**2)**0.5)

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
fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[Data1['Range']<=12]['y_actual'],Data1[Data1['Range']<=12]['Baseline'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_actual'],Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['Baseline'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>24)]['y_actual'],Data1[Data1['Range']>24]['Baseline'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.4f'  % roc_auc012,alpha=.6)


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

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_actual'],Data1[(Data1['Range']>12)&(Data1['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data1[(Data1['Range']>24)]['y_actual'],Data1[Data1['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.4f'  % roc_auc012,alpha=.6)


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

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data2[(Data2['Range']>12)&(Data2['Range']<=24)]['y_actual'],Data2[(Data2['Range']>12)&(Data2['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data2[(Data2['Range']>24)]['y_actual'],Data2[Data2['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.4f'  % roc_auc012,alpha=.6)


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










fig = plt.figure(figsize=(5,4))
fpr012, tpr012, threshold012 = metrics.roc_curve(Data3[Data3['Range']<=12]['y_actual'],Data3[Data3['Range']<=12]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data3[(Data3['Range']>12)&(Data3['Range']<=24)]['y_actual'],Data3[(Data3['Range']>12)&(Data3['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data3[(Data3['Range']>24)]['y_actual'],Data3[Data3['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.4f'  % roc_auc012,alpha=.6)


plt.title(Methods_Name[2]+' Model Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+Methods_Name[2]+' Model Range Sensitivity.png', bbox_inches='tight')
plt.show()










fig = plt.figure(figsize=(5,4))
fpr012, tpr012, threshold012 = metrics.roc_curve(Data4[Data4['Range']<=12]['y_actual'],Data4[Data4['Range']<=12]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'c--', label = 'Range less than 12 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data4[(Data4['Range']>12)&(Data4['Range']<=24)]['y_actual'],Data4[(Data4['Range']>12)&(Data4['Range']<=24)]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'b--', label = 'Range between 12 and 24 hours %0.4f'  % roc_auc012,alpha=.6)

fpr012, tpr012, threshold012 = metrics.roc_curve(Data4[(Data4['Range']>24)]['y_actual'],Data4[Data4['Range']>24]['y_pred'])
roc_auc012 = metrics.auc(fpr012, tpr012)

plt.plot(fpr012, tpr012, 'g--', label = 'Range between 24 and 36 hours %0.4f'  % roc_auc012,alpha=.6)


plt.title(Methods_Name[3]+' Model Range Sensitivity')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.savefig(save_path+Methods_Name[3]+' Model Range Sensitivity.png', bbox_inches='tight')
plt.show()












#%%Normalized Models Results Figure Plotting




Equivalet_Quaf=1#h0.sum()/h1.sum()


h1, b1 = np.histogram(Data1[Data1['y_actual']==1]['Baseline'],bins=30)
plt.bar(b1[:-1], h1.astype(np.float32) / h1.sum(), width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
h0, b0 = np.histogram(Data1[Data1['y_actual']==0]['Baseline'],bins=30)
plt.bar(b0[:-1], h0.astype(np.float32) / h0.sum(), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+'Baseline'+ ' Model Results')
plt.xlabel('Baseline'+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(100*max(max(h1.astype(np.float32) / h1.sum()),max(h0.astype(np.float32) / h0.sum())))/5)/20)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Normalized '+'Baseline'+ ' Model Results.png', bbox_inches='tight')
plt.show()

plt.bar(b1[:-1], h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf, width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
plt.bar(b0[:-1], h0.astype(np.float32) / (h0.sum()+h1.sum()), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Total Normalized '+'Baseline'+ ' Model Results')
plt.xlabel('Baseline'+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(1000*max(h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf))/5)/200)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Total Normalized '+'Baseline'+ ' Model Results.png', bbox_inches='tight')
plt.show()

h1, b1 = np.histogram(Data1[Data1['y_actual']==1]['Baseline'],bins=100000)
h0, b0 = np.histogram(Data1[Data1['y_actual']==0]['Baseline'],bins=100000)
TH0_Bin=np.argmax((h1.astype(np.float32) / h1.sum())>=(h0.astype(np.float32) / h0.sum()))/len(h1)
TH0_Bin=np.argmax((h1*Equivalet_Quaf)>=h0)/len(h1)








h1, b1 = np.histogram(Data1[Data1['y_actual']==1]['y_pred'],bins=30)
plt.bar(b1[:-1], h1.astype(np.float32) / h1.sum(), width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
h0, b0 = np.histogram(Data1[Data1['y_actual']==0]['y_pred'],bins=30)
plt.bar(b0[:-1], h0.astype(np.float32) / h0.sum(), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[0]+ ' Model Results')
plt.xlabel(Methods_Name[0]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(100*max(max(h1.astype(np.float32) / h1.sum()),max(h0.astype(np.float32) / h0.sum())))/5)/20)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Normalized '+Methods_Name[0]+ ' Model Results.png', bbox_inches='tight')
plt.show()


plt.bar(b1[:-1], h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf, width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
plt.bar(b0[:-1], h0.astype(np.float32) / (h0.sum()+h1.sum()), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Total Normalized '+Methods_Name[0]+ ' Model Results')
plt.xlabel(Methods_Name[0]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(1000*max(h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf))/5)/200)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Total Normalized '+Methods_Name[0]+ ' Model Results.png', bbox_inches='tight')
plt.show()


h1, b1 = np.histogram(Data1[Data1['y_actual']==1]['y_pred'],bins=100000)
h0, b0 = np.histogram(Data1[Data1['y_actual']==0]['y_pred'],bins=100000)
TH1_Bin=np.argmax((h1.astype(np.float32) / h1.sum())>=(h0.astype(np.float32) / h0.sum()))/len(h1)
TH1_Bin=np.argmax((h1*Equivalet_Quaf)>=h0)/len(h1)








h1, b1 = np.histogram(Data2[Data2['y_actual']==1]['y_pred'],bins=30)
plt.bar(b1[:-1], h1.astype(np.float32) / h1.sum(), width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
h0, b0 = np.histogram(Data2[Data2['y_actual']==0]['y_pred'],bins=30)
plt.bar(b0[:-1], h0.astype(np.float32) / h0.sum(), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[1]+ ' Model Results')
plt.xlabel(Methods_Name[1]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(100*max(max(h1.astype(np.float32) / h1.sum()),max(h0.astype(np.float32) / h0.sum())))/5)/20)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Normalized '+Methods_Name[1]+ ' Model Results.png', bbox_inches='tight')
plt.show()


plt.bar(b1[:-1], h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf, width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
plt.bar(b0[:-1], h0.astype(np.float32) / (h0.sum()+h1.sum()), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Total Normalized '+Methods_Name[1]+ ' Model Results')
plt.xlabel(Methods_Name[1]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(1000*max(h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf))/5)/200)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Total Normalized '+Methods_Name[1]+ ' Model Results.png', bbox_inches='tight')
plt.show()


h1, b1 = np.histogram(Data2[Data2['y_actual']==1]['y_pred'],bins=100000)
h0, b0 = np.histogram(Data2[Data2['y_actual']==0]['y_pred'],bins=100000)
TH2_Bin=np.argmax((h1.astype(np.float32) / h1.sum())>=(h0.astype(np.float32) / h0.sum()))/len(h1)
TH2_Bin=np.argmax((h1*Equivalet_Quaf)>=h0)/len(h1)







h1, b1 = np.histogram(Data3[Data3['y_actual']==1]['y_pred'],bins=30)
plt.bar(b1[:-1], h1.astype(np.float32) / h1.sum(), width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
h0, b0 = np.histogram(Data3[Data3['y_actual']==0]['y_pred'],bins=30)
plt.bar(b0[:-1], h0.astype(np.float32) / h0.sum(), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[2]+ ' Model Results')
plt.xlabel(Methods_Name[2]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(100*max(max(h1.astype(np.float32) / h1.sum()),max(h0.astype(np.float32) / h0.sum())))/5)/20)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Normalized '+Methods_Name[2]+ ' Model Results.png', bbox_inches='tight')
plt.show()


plt.bar(b1[:-1], h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf, width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
plt.bar(b0[:-1], h0.astype(np.float32) / (h0.sum()+h1.sum()), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Total Normalized '+Methods_Name[2]+ ' Model Results')
plt.xlabel(Methods_Name[2]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(1000*max(h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf))/5)/200)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Total Normalized '+Methods_Name[2]+ ' Model Results.png', bbox_inches='tight')
plt.show()


h1, b1 = np.histogram(Data3[Data3['y_actual']==1]['y_pred'],bins=100000)
h0, b0 = np.histogram(Data3[Data3['y_actual']==0]['y_pred'],bins=100000)
TH3_Bin=np.argmax((h1.astype(np.float32) / h1.sum())>=(h0.astype(np.float32) / h0.sum()))/len(h1)
TH3_Bin=np.argmax((h1*Equivalet_Quaf)>=h0)/len(h1)







h1, b1 = np.histogram(Data4[Data4['y_actual']==1]['y_pred'],bins=30)
plt.bar(b1[:-1], h1.astype(np.float32) / h1.sum(), width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
h0, b0 = np.histogram(Data4[Data4['y_actual']==0]['y_pred'],bins=30)
plt.bar(b0[:-1], h0.astype(np.float32) / h0.sum(), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Normalized '+Methods_Name[3]+ ' Model Results')
plt.xlabel(Methods_Name[3]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(100*max(max(h1.astype(np.float32) / h1.sum()),max(h0.astype(np.float32) / h0.sum())))/5)/20)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Normalized '+Methods_Name[3]+ ' Model Results.png', bbox_inches='tight')
plt.show()


plt.bar(b1[:-1], h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf, width=(b1[1]-b1[0]), alpha=.5,label='Convective Class', color='red')
plt.bar(b0[:-1], h0.astype(np.float32) / (h0.sum()+h1.sum()), width=(b0[1]-b0[0]), alpha=.7,label='Non-Convective Class', color='grey')
plt.legend()
plt.title('Total Normalized '+Methods_Name[3]+ ' Model Results')
plt.xlabel(Methods_Name[3]+' Model Prediction Score')
plt.ylabel('Density')
plt.ylim(0,math.ceil(math.ceil(1000*max(h1.astype(np.float32) / (h0.sum()+h1.sum())*Equivalet_Quaf))/5)/200)
plt.xlim(0,1)
plt.grid()
plt.savefig(save_path+'Total Normalized '+Methods_Name[3]+ ' Model Results.png', bbox_inches='tight')
plt.show()

h1, b1 = np.histogram(Data4[Data4['y_actual']==1]['y_pred'],bins=100000)
h0, b0 = np.histogram(Data4[Data4['y_actual']==0]['y_pred'],bins=100000)
TH4_Bin=np.argmax((h1.astype(np.float32) / h1.sum())>=(h0.astype(np.float32) / h0.sum()))/len(h1)
TH4_Bin=np.argmax((h1*Equivalet_Quaf)>=h0)/len(h1)






#%%Models Prediction Score Figure Plotting
Manual_Binarization=0
if Manual_Binarization==1:
    TH_Visual=0.62
    TH0_Bin=TH_Visual
    TH1_Bin=TH_Visual
    TH2_Bin=TH_Visual
    TH3_Bin=TH_Visual
    TH4_Bin=TH_Visual

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
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

idx=np.argmax(np.sum(np.sum(yAct_Array,axis=2),axis=1))


fig = plt.figure(figsize=(7,12))

m.contourf(x, y, 1-yAct_Array[idx,:,:], cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yAct_Array[idx,:,:], cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title('Target Image')
plt.savefig(save_path+'Target_Image.png', bbox_inches='tight')
plt.show()





cmap = plt.cm.jet
levels = np.arange(0, 1, 0.025)






fig = plt.figure(figsize=(7,12))

#m.contourf(x, y, 1-yAct_Array[idx,:,:], cmap=plt.cm.Greys, alpha=.7)

m.contourf(x, y, Baseline_Array[idx,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
#plt.xlabel(Methods_Name[0]+' Model Prediction Score')
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title('Baseline Model Prediction Score')
plt.savefig(save_path+'Baseline Model Prediction Score.png', bbox_inches='tight')
plt.show()








fig = plt.figure(figsize=(7,12))
m.contourf(x, y, 1-(Baseline_Array[idx,:,:]>=TH0_Bin), cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, Baseline_Array[idx,:,:]>=TH0_Bin, cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title('Baseline Model Prediction Score (Binary), TH = %0.5f' % TH0_Bin)
plt.savefig(save_path+'Baseline Model Prediction Score_Binary.png', bbox_inches='tight')
plt.show()

AS=np.zeros(5)
BAS=np.zeros(5)
F1=np.zeros(5)
PS=np.zeros(5)
JS=np.zeros(5)
CKS=np.zeros(5)

AS[0]=metrics.accuracy_score(yAct_Array[idx,:,:],Baseline_Array[idx,:,:]>=TH0_Bin)
#BAS[0]=metrics.balanced_accuracy_score(yAct_Array[idx,:,:].astype(int),(Baseline_Array[idx,:,:]>=TH0_Bin).astype(int))
F1[0]=metrics.f1_score(yAct_Array[idx,:,:],Baseline_Array[idx,:,:]>=TH0_Bin,average='weighted')
PS[0]=metrics.precision_score(yAct_Array[idx,:,:],Baseline_Array[idx,:,:]>=TH0_Bin, average='weighted')
JS[0]=metrics.jaccard_score(yAct_Array[idx,:,:],Baseline_Array[idx,:,:]>=TH0_Bin, average='weighted')
#CKS[0]=metrics.cohen_kappa_score(yAct_Array[idx,:,:],Baseline_Array[idx,:,:]>=TH0_Bin)








fig = plt.figure(figsize=(7,12))
#m.contourf(x, y, Baseline_Array[129,:,:], cmap=plt.cm.jet, alpha=0)
m.contourf(x, y, yPred_Array[idx,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
#plt.xlabel(Methods_Name[0]+' Model Prediction Score')
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[0]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[0]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()







fig = plt.figure(figsize=(7,12))
m.contourf(x, y, 1-(yPred_Array[idx,:,:]>=TH1_Bin), cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yPred_Array[idx,:,:]>=TH1_Bin, cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title(Methods_Name[0]+' Model Prediction Score (Binary), TH = %0.5f' % TH1_Bin)
plt.savefig(save_path+Methods_Name[0]+' Model Prediction Score_Binary.png', bbox_inches='tight')
plt.show()

AS[1]=metrics.accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin)
#BAS[1]=metrics.balanced_accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin)
F1[1]=metrics.f1_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin, average='weighted')
PS[1]=metrics.precision_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin, average='weighted')
JS[1]=metrics.jaccard_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin, average='weighted')
#CKS[1]=metrics.cohen_kappa_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH1_Bin)











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
cmap = plt.cm.jet
#m.contourf(x, y, Baseline_Array[129,:,:], cmap=plt.cm.jet, alpha=0)
levels = np.arange(0, 1, 0.025)
m.contourf(x, y, yPred_Array[idx,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
#plt.xlabel(Methods_Name[1]+' Model Prediction Score')
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[1]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[1]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()



fig = plt.figure(figsize=(7,12))
m.contourf(x, y, 1-(yPred_Array[idx,:,:]>=TH2_Bin), cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yPred_Array[idx,:,:]>=TH2_Bin, cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title(Methods_Name[1]+' Model Prediction Score (Binary), TH = %0.5f' % TH2_Bin)
plt.savefig(save_path+Methods_Name[1]+' Model Prediction Score_Binary.png', bbox_inches='tight')
plt.show()

AS[2]=metrics.accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin)
#BAS[2]=metrics.balanced_accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin)
F1[2]=metrics.f1_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin, average='weighted')
PS[2]=metrics.precision_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin, average='weighted')
JS[2]=metrics.jaccard_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin, average='weighted')
#CKS[2]=metrics.cohen_kappa_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH2_Bin)












Data3.reset_index(inplace=True,drop=True)

Data3['NN_Error'] = Data3['y_actual']-Data3['y_pred']
Data3['NN_Error'] = Data3['NN_Error']**2

Data3['B_Error'] = Data3['y_actual']-Data3['Baseline']
Data3['B_Error'] = Data3['B_Error']**2

Data3.head()




Data3.sort_values(['TimeSlot','Range','Lat','Lon'],ascending=True,inplace=True)
Index = Data3.groupby(['TimeSlot','Range']).count().reset_index()
Index = Index[['TimeSlot','Range']]
Count_Array = np.array(Data3['Count']).reshape(len(Index),141,181)
yAct_Array = np.array(Data3['y_actual']).reshape(len(Index),141,181)
Baseline_Array = np.array(Data3['Baseline']).reshape(len(Index),141,181)
yPred_Array = np.array(Data3['y_pred']).reshape(len(Index),141,181)
Err_Array = np.array(Data3['NN_Error']).reshape(len(Index),141,181)
BErr_Array = np.array(Data3['B_Error']).reshape(len(Index),141,181)
LatArray = np.array(Data3['Lat']).reshape(len(Index),141,181)
LonArray = np.array(Data3['Lon']).reshape(len(Index),141,181)




Index['ReleaseTime'] = Index['TimeSlot'] - Index['Range'].apply(lambda x: datetime.timedelta(hours=x))


##Index.loc[196]


GIF = Index.sort_values(['ReleaseTime','Range'],ascending = [True,True]).reset_index()


print(Data3['Lon'].min(), Data3['Lon'].max(), Data3['Lat'].min(), Data3['Lat'].max())


#extent = [ 6, 20, 36, 54] # [left, right, bottom, top]
extent = [-10, 7, 32, 49]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Lat = LatArray[0,:,:]
Lon = LonArray[0,:,:]

#m = Basemap(projection='merc',llcrnrlat=Lat.min(),urcrnrlat=Lat.max(),\
 #                           llcrnrlon=Lon.min(),urcrnrlon=Lon.max(),lat_ts=20,resolution='l')
x, y = m(Lon, Lat)


fig = plt.figure(figsize=(7,12))
cmap = plt.cm.jet
#m.contourf(x, y, Baseline_Array[129,:,:], cmap=plt.cm.jet, alpha=0)
levels = np.arange(0, 1, 0.025)
m.contourf(x, y, yPred_Array[idx,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
#plt.xlabel(Methods_Name[2]+' Model Prediction Score')
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[2]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[2]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(7,12))
m.contourf(x, y, 1-(yPred_Array[idx,:,:]>=TH3_Bin), cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yPred_Array[idx,:,:]>=TH3_Bin, cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title(Methods_Name[2]+' Model Prediction Score (Binary), TH = %0.5f' % TH3_Bin)
plt.savefig(save_path+Methods_Name[2]+' Model Prediction Score_Binary.png', bbox_inches='tight')
plt.show()

AS[3]=metrics.accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin)
#BAS[3]=metrics.balanced_accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin)
F1[3]=metrics.f1_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin, average='weighted')
PS[3]=metrics.precision_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin, average='weighted')
JS[3]=metrics.jaccard_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin, average='weighted')
#CKS[3]=metrics.cohen_kappa_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH3_Bin)










Data4.reset_index(inplace=True,drop=True)

Data4['NN_Error'] = Data4['y_actual']-Data4['y_pred']
Data4['NN_Error'] = Data4['NN_Error']**2

Data4['B_Error'] = Data4['y_actual']-Data4['Baseline']
Data4['B_Error'] = Data4['B_Error']**2

Data4.head()




Data4.sort_values(['TimeSlot','Range','Lat','Lon'],ascending=True,inplace=True)
Index = Data4.groupby(['TimeSlot','Range']).count().reset_index()
Index = Index[['TimeSlot','Range']]
Count_Array = np.array(Data4['Count']).reshape(len(Index),141,181)
yAct_Array = np.array(Data4['y_actual']).reshape(len(Index),141,181)
Baseline_Array = np.array(Data4['Baseline']).reshape(len(Index),141,181)
yPred_Array = np.array(Data4['y_pred']).reshape(len(Index),141,181)
Err_Array = np.array(Data4['NN_Error']).reshape(len(Index),141,181)
BErr_Array = np.array(Data4['B_Error']).reshape(len(Index),141,181)
LatArray = np.array(Data4['Lat']).reshape(len(Index),141,181)
LonArray = np.array(Data4['Lon']).reshape(len(Index),141,181)




Index['ReleaseTime'] = Index['TimeSlot'] - Index['Range'].apply(lambda x: datetime.timedelta(hours=x))


##Index.loc[196]


GIF = Index.sort_values(['ReleaseTime','Range'],ascending = [True,True]).reset_index()


print(Data4['Lon'].min(), Data4['Lon'].max(), Data4['Lat'].min(), Data4['Lat'].max())


#extent = [ 6, 20, 36, 54] # [left, right, bottom, top]
extent = [-10, 7, 32, 49]
m = Basemap(projection='cyl', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='l')

Lat = LatArray[0,:,:]
Lon = LonArray[0,:,:]

#m = Basemap(projection='merc',llcrnrlat=Lat.min(),urcrnrlat=Lat.max(),\
 #                           llcrnrlon=Lon.min(),urcrnrlon=Lon.max(),lat_ts=20,resolution='l')
x, y = m(Lon, Lat)



fig = plt.figure(figsize=(7,12))
cmap = plt.cm.jet
#m.contourf(x, y, Baseline_Array[129,:,:], cmap=plt.cm.jet, alpha=0)
levels = np.arange(0, 1, 0.025)
m.contourf(x, y, yPred_Array[idx,:,:], cmap=cmap,levels=levels, alpha=.6)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#plt.scatter(x,y,color='k',s=.2)
#plt.clim(0,.9)
plt.colorbar(ticks = [0,.15,.3,.45,.6,.75,.9],pad=.03,orientation='horizontal',extend='both')
#plt.xlabel(Methods_Name[3]+' Model Prediction Score')
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.2)
plt.title(Methods_Name[3]+' Model Prediction Score')
plt.savefig(save_path+Methods_Name[3]+' Model Prediction Score.png', bbox_inches='tight')
plt.show()



fig = plt.figure(figsize=(7,12))
m.contourf(x, y, 1-(yPred_Array[idx,:,:]>=TH4_Bin), cmap=plt.cm.Greys, alpha=.6)
m.contourf(x, y, yPred_Array[idx,:,:]>=TH4_Bin, cmap=plt.cm.Reds, alpha=.3)
m.drawcoastlines(color = 'w',linewidth=1.2)
m.drawcountries(color = 'w',linewidth=1.2)
#parallels = np.arange(36,54,2)
#meridians = np.arange(5,20,5)
#m.drawparallels(parallels)
#m.drawmeridians(meridians)
plt.scatter(x,y,color='k',s=.75)
plt.title(Methods_Name[3]+' Model Prediction Score (Binary), TH = %0.5f' % TH4_Bin)
plt.savefig(save_path+Methods_Name[3]+' Model Prediction Score_Binary.png', bbox_inches='tight')
plt.show()

AS[4]=metrics.accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin)
#BAS[4]=metrics.balanced_accuracy_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin)
F1[4]=metrics.f1_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin, average='weighted')
PS[4]=metrics.precision_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin, average='weighted')
JS[4]=metrics.jaccard_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin, average='weighted')
#CKS[4]=metrics.cohen_kappa_score(yAct_Array[idx,:,:],yPred_Array[idx,:,:]>=TH4_Bin)
