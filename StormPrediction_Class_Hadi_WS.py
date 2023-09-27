import pandas as pd
import numpy as np
import keras
from sklearn import utils
from itertools import compress



class Scaler_Generator(keras.utils.Sequence):
    def __init__(self, filenames, parameters):
        self.filenames = filenames
        self.parameters = parameters

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        Train = []
        for i, ID in enumerate(self.filenames[idx: (idx + 1)]):
#            DF = pd.read_pickle(ID)
            DF = pd.read_hdf(ID)
            hour_param = DF['TimeSlot'][0].hour
            DF['Hour'] = hour_param
            Train.append(DF)
        Train = pd.concat(Train,sort=False)
        Train = Train[self.parameters]

        return Train.astype(float)


class Data_Generator(keras.utils.Sequence):
    def __init__(self, filenames, num_files, num_members, parameters, scaler):
        self.filenames = filenames
        self.num_files = num_files
        self.num_members = num_members
        self.batch_size = 25521*self.num_files*self.num_members #25521 corresponds to number of datapoints per map
        self.files_range = np.arange(0, len(self.filenames), self.num_files)
        self.member_range = np.arange(1, 2, self.num_members)
#        self.member_range = np.arange(0, 50, self.num_members)

        self.file_index = pd.DataFrame(np.append(self.files_range[1:],len(self.filenames)),self.files_range).reset_index()
        self.file_index.columns = ['Start','End']

        self.member_index = pd.DataFrame([2],self.member_range).reset_index()
#        self.member_index = pd.DataFrame(np.append(self.member_range[1:],50),self.member_range).reset_index()
        self.member_index.columns = ['Start','End']

        self.Index = []
        for i in range(0,len(self.member_index)):
            DF = self.file_index
            DF['StartM'] = self.member_index.Start[i]
            DF['EndM'] = self.member_index.End[i]
            self.Index.append(DF.copy())
        
        self.Index = pd.concat(self.Index).reset_index(drop=True)

        self.batch_size = len(self.Index)

        self.parameters = parameters

        self.scaler = scaler


    def __len__(self):
#        return int((np.ceil(len(self.filenames) / float(self.num_files))).astype(np.int) * 50/self.num_members)
        return int((np.ceil(len(self.filenames) / float(self.num_files))).astype(np.int) * 1/self.num_members)



    def on_epoch_end(self):
        'Shuffles files and indexes after each epoch'
        self.filenames = utils.shuffle(self.filenames)
        self.Index = utils.shuffle(self.Index)


    def __getitem__(self, idx):

        Train = []
        for i, ID in enumerate(self.filenames[self.Index.Start[idx]:self.Index.End[idx]]):
            DF = pd.read_hdf(ID)
#            DF = pd.read_pickle(ID)
            hour_param = DF['TimeSlot'][0].hour
#            DF = DF[(DF['Member'] > self.Index.StartM[idx]) & (DF['Member'] < self.Index.EndM[idx] + 1)]
#            DF = DF[(DF['Member'] >= self.Index.StartM[idx]) & (DF['Member'] < self.Index.EndM[idx] + 1)]
            DF['Hour'] = hour_param
            Train.append(DF)

        Train = pd.concat(Train,sort=False)

        Train.reset_index(inplace=True, drop=True)

        Train = utils.shuffle(Train)

        yTrain = Train[['Binary']]
        yTrain = yTrain.iloc[:, 0]
        xTrain = Train[self.parameters]
        
#        Fixed_Columns_idx=xTrain.std()==0
#        Fixed_Columns=xTrain.loc[:,Fixed_Columns_idx]
#        xTrain.loc[:,xTrain.std()==0]=np.random.normal(0,1,size=Fixed_Columns.shape)
#        self.scaler.fit(xTrain.astype(float))
        xTrain = self.scaler.transform(xTrain.astype(float))
#        xTrain = self.scaler.fit_transform(xTrain.astype(float))
#        xTrain[:,Fixed_Columns_idx]=Fixed_Columns
#        joblib.dump(self.scaler, 'Storm_Model_Scaler')
#        joblib.dump(self.scaler, 'Storm_Model_Scaler')
        xTrain = pd.DataFrame(xTrain)
#        breakpoint()
        return xTrain.values, yTrain.values


class Test_Generator(keras.utils.Sequence):
    def __init__(self, filenames, parameters, model, scaler):
        self.filenames = filenames
        self.parameters = parameters
        self.model = model
        self.scaler = scaler

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        Test = []
        for i, ID in enumerate(self.filenames[idx: (idx + 1)]):
            #DF = pd.read_pickle(ID)
            DF = pd.read_hdf(ID)
            hour_param = DF['TimeSlot'][0].hour
            DF['Hour'] = hour_param
            Test.append(DF)

        Test = pd.concat(Test,sort=False)

        Test.reset_index(inplace=True, drop=True)
        
        Test['Baseline'] = ((Test.totalxMean >= 44) & (Test.crrMean > 0)).astype(int)


        y_actual = Test[['Binary']]
        y_actual = y_actual.iloc[:, 0]

        xTest = Test[self.parameters]

        xTest = self.scaler.transform(xTest.astype(float))
#        xTest = self.scaler.fit_transform(xTest.astype(float))
        
        xTest = pd.DataFrame(xTest)

        y_pred = self.model.predict(xTest)

        Test = Test[['Lat','Lon','TimeSlot','Range','Member','Baseline','Count']]

        Test['y_actual'] = y_actual
        Test['y_pred'] = y_pred
        
        return Test

#
#
class Permutation(keras.utils.Sequence):

    def __init__(self, filenames, parameters, model, scaler, param):
        self.filenames = filenames
        self.parameters = parameters
        self.model = model
        self.scaler = scaler
        self.param = param

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        Test = []
        for i, ID in enumerate(self.filenames[idx: (idx + 1)]):
            DF = pd.read_pickle(ID)
            hour_param = DF['TimeSlot'][0].hour
            DF['Hour'] = hour_param
            Test.append(DF)

        Test = pd.concat(Test,sort=False)

        Test.reset_index(inplace=True, drop=True)
        
        Test['Baseline'] = ((Test.totalxMean >= 44) & (Test.crrMean > 0)).astype(int)


        y_actual = Test[['Binary']]
        y_actual = y_actual.iloc[:, 0]

        xTest = Test[self.parameters]
        i = self.parameters.index(self.param)
        permX  = pd.concat(
                [xTest.iloc[:,:i],
                 pd.DataFrame(np.random.permutation(xTest[self.param].values)),
                 xTest.iloc[:,i+1:]],axis=1)
        
        
        
        xTest = self.scaler.transform(permX.astype(float))
        xTest = pd.DataFrame(xTest)

        y_pred = self.model.predict(xTest)

        Test = Test[['Lat','Lon','TimeSlot','Range','Member','Any','Baseline']]

        Test['y_actual'] = y_actual
        Test['y_pred'] = y_pred

        return Test



def Get_files(files,dates):
    times = list(set([i.split('/')[-1][:14] for i in files]))
    date_times = list(compress(times, pd.Series(times).apply(lambda x: any([i in x for i in dates]))))
    date_files = []
    for t in date_times:
        date_files.append(list(np.array(files)[np.array(list(t in file for file in files))]))

    datefiles = []
    for sublist in date_files:
        for item in sublist:
            datefiles.append(item)

    return datefiles

def get_RDT_date(a):
    x = a.split('/')[-1]
    return x[4:8]+x[9:11]+x[12:14]


