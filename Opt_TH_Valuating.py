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
Fusion_Operatings=['All']
Method_Name='Fusion'
#result_files = glob.glob('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_2018*16')
result_files = glob.glob('NN_16_16\\'+'Fusion_'+Fusion_Operatings[0]+'\\Results_2018*')
#result_files = glob.glob(r'D:\postdoc_hadi\ISOBAR\Code\Fusion\Results2Comp\Results_Fusion\Results_2018*')

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
plt.savefig(Method_Name+'vsBaseline_Acc_TPR_FPR.png', bbox_inches='tight')
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
