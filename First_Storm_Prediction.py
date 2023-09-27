{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "executionInfo": {
     "elapsed": 465543,
     "status": "ok",
     "timestamp": 1602482633540,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "mr8bbrKZKIH3",
    "outputId": "4e74ec73-0c0a-4bd1-e00f-550dd420f702"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mounted at /content/drive\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "drive.mount(\"/content/drive\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 102
    },
    "executionInfo": {
     "elapsed": 16617,
     "status": "ok",
     "timestamp": 1602482652744,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "K0pGljlXqXhn",
    "outputId": "8dd4f8cb-b9cd-4081-b04c-ef8d83eb0efb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2.30000000e+01 9.99968750e+02 1.45500039e+04 6.28200000e+03\n",
      " 2.92365625e+04 4.96831884e+01 4.61940527e+01 1.18999481e-01\n",
      " 1.28898621e-01 2.19876830e+07 2.49288960e+07 1.00001323e+00\n",
      " 5.43188477e+01 1.64134430e+02 7.62176514e-03 4.17470932e-03\n",
      " 2.49653793e+02 1.37288788e+02 2.63460010e+04]\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import pandas\n",
    "import pandas_datareader\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "#load train data\n",
    "data0 = {'Train_Files_Name':['23H_2018-05-25_23_Data','23H_2018-05-21_23_Data','23H_2018-06-01_23_Data','00H_2018-06-08_00_Data','15H_2018-07-27_15_Data']}#Insert manually all train data name\n",
    "df0=pd.DataFrame(data0, columns=['Train_Files_Name'])\n",
    "for ind in df0.index:\n",
    "  File_Name=df0['Train_Files_Name'][ind]\n",
    "  ROOT_DIR=os.path.abspath(os.path.join('drive','My Drive','Database_Manuel','Sample_Data',File_Name))\n",
    "  I=pandas.read_pickle(ROOT_DIR)\n",
    "  In=I[[\"Range\",\"cin\",\"hcct\",\"cape\",\"sp\",\"tcw\",\"tcwv\",\"lsp\",\"cp\",\"sshf\",\"slhf\",\"tcc\",\"2t\",\t\"2d\"\t,\"crr\",\t\"lsrr\",\t\"kx\",\t\"totalx\",\t\"z\"]]\n",
    "  Tar=I[\"target\"]\n",
    "  if ind == 0:\n",
    "    Input_Train=In.to_numpy()\n",
    "    Target_Train=Tar.to_numpy()\n",
    "  else:\n",
    "    Input_Train=np.concatenate((Input_Train,In.to_numpy()))\n",
    "    Target_Train=np.concatenate((Target_Train,Tar.to_numpy()))\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "#load test data\n",
    "data1 = {'Test_Files_Name':['22H_2018-08-12_22_Data']}#Insert manually all test data name\n",
    "df1=pd.DataFrame(data1, columns=['Test_Files_Name'])\n",
    "for ind in df1.index:\n",
    "  File_Name=df1['Test_Files_Name'][ind]\n",
    "  ROOT_DIR=os.path.abspath(os.path.join('drive','My Drive','Database_Manuel','Sample_Data',File_Name))\n",
    "  I=pandas.read_pickle(ROOT_DIR)\n",
    "  In=I[[\"Range\",\"cin\",\"hcct\",\"cape\",\"sp\",\"tcw\",\"tcwv\",\"lsp\",\"cp\",\"sshf\",\"slhf\",\"tcc\",\"2t\",\t\"2d\"\t,\"crr\",\t\"lsrr\",\t\"kx\",\t\"totalx\",\t\"z\"]]\n",
    "  Tar=I[\"target\"]\n",
    "  if ind == 0:\n",
    "    Input_Test=In.to_numpy()\n",
    "    Target_Test=Tar.to_numpy()\n",
    "  else:\n",
    "    Input_Test=np.concatenate((Input_Test,In.to_numpy()))\n",
    "    Target_Test=np.concatenate((Target_Test,Tar.to_numpy()))\n",
    "\n",
    "print(Input_Train.max(axis=0)-Input_Train.min(axis=0))\n",
    "#print(np.exp(Input_Train.min(axis=0)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "executionInfo": {
     "elapsed": 762859,
     "status": "ok",
     "timestamp": 1602483445659,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "iMRuPF9FNIQV",
    "outputId": "a673cac0-6ccf-4b5a-a459-277144738e35"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Normalization_Param \n",
      " [[0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]\n",
      " [0. 0. 0. 0.]]\n",
      "ind= 0\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 1\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 2\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 3\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 4\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 5\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 6\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 7\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 8\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 9\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 10\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 11\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 12\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 13\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 14\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 15\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 16\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 17\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 18\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n"
     ]
    }
   ],
   "source": [
    "#Train data Normalizing\n",
    "\n",
    "def Log_Normalizing( Input_Feature ):\n",
    "   \"Apply log func on feature to decrease the range-change\"\n",
    "   Min_Val=Input_Feature.min(axis=0)\n",
    "   tmp=Input_Feature-Min_Val+1\n",
    "   a=np.log(tmp.max(axis=0))\n",
    "   if a != 0:\n",
    "     Input_Feature=2*np.log(tmp)/a-1\n",
    "     print('Log_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=np.log(tmp)-1\n",
    "     print('Log_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature,Min_Val,a\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Exp_Normalizing( Input_Feature ):\n",
    "   \"Apply exp func on feature to increase the range-change\"\n",
    "   Min_Val=Input_Feature.min(axis=0)\n",
    "   tmp=Input_Feature-Min_Val\n",
    "   a=np.exp(Input_Feature.max(axis=0)-Min_Val)-1\n",
    "   if a != 0:\n",
    "     Input_Feature=2*(np.exp(Input_Feature-Min_Val)-1)/a-1\n",
    "     print('Exp_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=2*(np.exp(Input_Feature-Min_Val)-1)-1\n",
    "     print('Exp_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature,Min_Val,a\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Linear_Normalizing( Input_Feature ):\n",
    "   \"Apply linear func on feature to decrease the range-change\"\n",
    "   Min_Val=Input_Feature.min(axis=0)\n",
    "   tmp=Input_Feature-Min_Val\n",
    "   a=tmp.max(axis=0)\n",
    "   if a != 0:\n",
    "     Input_Feature=2*tmp/a-1\n",
    "     print('Linear_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=tmp-1\n",
    "     print('Linear_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature,Min_Val,a\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Mean0_Var1_Normalizing( Input_Feature ):\n",
    "   \"Apply Mean0_Var1 on feature to decrease the range-change\"\n",
    "   from statistics import mean\n",
    "   from statistics import stdev\n",
    "   Mean_Val=mean(Input_Feature)\n",
    "   Std_Val=stdev(Input_Feature)\n",
    "   if Std_Val != 0:\n",
    "     Input_Feature=(Input_Feature-Mean_Val)/Std_Val\n",
    "     print('Mean0_Var1_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=Input_Feature-Mean_Val\n",
    "     print('Mean0_Var1_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature,Mean_Val,Std_Val\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "data2 = {'Normalization_Func':['Lin','Log','Log','Log','Log','Lin','Lin','Exp','Exp','Log','Log','Exp','Lin','Lin','Exp','Exp','Lin','Lin','Log']}#Insert manually Normalization Function for each feature\n",
    "df2=pd.DataFrame(data2, columns=['Normalization_Func'])\n",
    "Normalization_Param=np.zeros([df2.index.stop,4])\n",
    "print('Normalization_Param \\n',Normalization_Param)\n",
    "for ind in df2.index:\n",
    "  print('ind=', ind)\n",
    "  Normalization_Type=df2['Normalization_Func'][ind]\n",
    "  if Normalization_Type == 'Log':\n",
    "    Input_Train[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1]=Log_Normalizing(Input_Train[:,ind])\n",
    "  else:\n",
    "    if Normalization_Type == 'Exp':\n",
    "      Input_Train[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1]=Exp_Normalizing(Input_Train[:,ind])\n",
    "    else:\n",
    "      if Normalization_Type == 'Lin':\n",
    "        Input_Train[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1]=Linear_Normalizing(Input_Train[:,ind])\n",
    "  Input_Train[:,ind],Normalization_Param[ind,2],Normalization_Param[ind,3]=Mean0_Var1_Normalizing(Input_Train[:,ind])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 391
    },
    "executionInfo": {
     "elapsed": 659,
     "status": "ok",
     "timestamp": 1602484508619,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "QrkWLf3IW6gs",
    "outputId": "a10ee8b7-2216-453c-aabb-048963b7c576"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Normalization_Func  ...  Normalizing_Param4\n",
      "0                 Lin  ...            0.778540\n",
      "1                 Log  ...            0.501128\n",
      "2                 Log  ...            0.868426\n",
      "3                 Log  ...            0.512706\n",
      "4                 Log  ...            0.041843\n",
      "5                 Lin  ...            0.268816\n",
      "6                 Lin  ...            0.286421\n",
      "7                 Exp  ...            0.027960\n",
      "8                 Exp  ...            0.028928\n",
      "9                 Log  ...            0.026206\n",
      "10                Log  ...            0.022849\n",
      "11                Exp  ...            0.782948\n",
      "12                Lin  ...            0.275052\n",
      "13                Lin  ...            0.066153\n",
      "14                Exp  ...            0.033743\n",
      "15                Exp  ...            0.019924\n",
      "16                Lin  ...            0.121586\n",
      "17                Lin  ...            0.118198\n",
      "18                Log  ...            0.172799\n",
      "\n",
      "[19 rows x 5 columns]\n"
     ]
    }
   ],
   "source": [
    "Data_Train_Normalization = {'Normalization_Func':['Lin','Log','Log','Log','Log','Lin','Lin','Exp','Exp','Log','Log','Exp','Lin','Lin','Exp','Exp','Lin','Lin','Log'],\n",
    "         'Normalizing_Param1':Normalization_Param[:,0],\n",
    "         'Normalizing_Param2':Normalization_Param[:,1],\n",
    "         'Normalizing_Param3':Normalization_Param[:,2],\n",
    "         'Normalizing_Param4':Normalization_Param[:,3]}\n",
    "df_Data_Train_Normalization=pd.DataFrame(Data_Train_Normalization, columns=['Normalization_Func','Normalizing_Param1','Normalizing_Param2','Normalizing_Param3','Normalizing_Param4'])\n",
    "print(df_Data_Train_Normalization)\n",
    "ROOT_DIR=os.path.abspath(os.path.join('drive','My Drive','Database_Manuel','Sample_Data','Data_Train_Normalization.csv'))\n",
    "df_Data_Train_Normalization.to_csv(ROOT_DIR)#, index=False)\n",
    "#data = pd.read_csv(ROOT_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "executionInfo": {
     "elapsed": 1321,
     "status": "ok",
     "timestamp": 1602484558992,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "v_45HAUdmMD9",
    "outputId": "1276b3d1-f412-4d9f-f151-678b573f858d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ind= 0\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 1\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 2\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 3\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 4\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 5\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 6\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 7\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 8\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 9\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 10\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 11\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:7: RuntimeWarning: invalid value encountered in log\n",
      "  import sys\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 12\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 13\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 14\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 15\n",
      "Exp_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 16\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 17\n",
      "Linear_Normalizing\n",
      "Mean0_Var1_Normalizing\n",
      "ind= 18\n",
      "Log_Normalizing\n",
      "Mean0_Var1_Normalizing\n"
     ]
    }
   ],
   "source": [
    "#Test data Normalizing\n",
    "\n",
    "def Log_Test_Normalizing( Input_Feature, Min_Val, a ):\n",
    "\n",
    "   tmp=Input_Feature-Min_Val+1\n",
    "   if a != 0:\n",
    "     Input_Feature=2*np.log(tmp)/a-1\n",
    "     print('Log_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=np.log(tmp)-1\n",
    "     print('Log_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Exp_Test_Normalizing( Input_Feature, Min_Val, a ):\n",
    "\n",
    "   tmp=Input_Feature-Min_Val\n",
    "   if a != 0:\n",
    "     Input_Feature=2*(np.exp(Input_Feature-Min_Val)-1)/a-1\n",
    "     print('Exp_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=2*(np.exp(Input_Feature-Min_Val)-1)-1\n",
    "     print('Exp_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Linear_Test_Normalizing( Input_Feature, Min_Val, a ):\n",
    "\n",
    "   tmp=Input_Feature-Min_Val\n",
    "   if a != 0:\n",
    "     Input_Feature=2*tmp/a-1\n",
    "     print('Linear_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=tmp-1\n",
    "     print('Linear_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def Mean0_Var1_Test_Normalizing( Input_Feature, Mean_Val, Std_Val ):\n",
    "\n",
    "   if Std_Val != 0:\n",
    "     Input_Feature=(Input_Feature-Mean_Val)/Std_Val\n",
    "     print('Mean0_Var1_Normalizing')\n",
    "   else:\n",
    "     Input_Feature=Input_Feature-Mean_Val\n",
    "     print('Mean0_Var1_Normalizing: devide to zero can not be performed!')\n",
    "   return Input_Feature\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "for ind in df2.index:\n",
    "  print('ind=', ind)\n",
    "  Normalization_Type=df2['Normalization_Func'][ind]\n",
    "  if Normalization_Type == 'Log':\n",
    "    Input_Test[:,ind]=Log_Test_Normalizing(Input_Test[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1])\n",
    "  else:\n",
    "    if Normalization_Type == 'Exp':\n",
    "      Input_Test[:,ind]=Exp_Test_Normalizing(Input_Test[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1])\n",
    "    else:\n",
    "      if Normalization_Type == 'Lin':\n",
    "        Input_Test[:,ind]=Linear_Test_Normalizing(Input_Test[:,ind],Normalization_Param[ind,0],Normalization_Param[ind,1])\n",
    "  Input_Test[:,ind]=Mean0_Var1_Test_Normalizing(Input_Test[:,ind],Normalization_Param[ind,2],Normalization_Param[ind,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 374
    },
    "executionInfo": {
     "elapsed": 728699,
     "status": "ok",
     "timestamp": 1602489993842,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "hZOShXLIswEU",
    "outputId": "39c83365-2394-4dd2-ebf3-d0603355b63f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "49846/49846 [==============================] - 72s 1ms/step - loss: 0.1382 - accuracy: 0.9443\n",
      "Epoch 2/10\n",
      "49846/49846 [==============================] - 74s 1ms/step - loss: 0.1264 - accuracy: 0.9483\n",
      "Epoch 3/10\n",
      "49846/49846 [==============================] - 73s 1ms/step - loss: 0.1232 - accuracy: 0.9496\n",
      "Epoch 4/10\n",
      "49846/49846 [==============================] - 72s 1ms/step - loss: 0.1215 - accuracy: 0.9503\n",
      "Epoch 5/10\n",
      "49846/49846 [==============================] - 73s 1ms/step - loss: 0.1205 - accuracy: 0.9507\n",
      "Epoch 6/10\n",
      "49846/49846 [==============================] - 73s 1ms/step - loss: 0.1199 - accuracy: 0.9511\n",
      "Epoch 7/10\n",
      "49846/49846 [==============================] - 72s 1ms/step - loss: 0.1197 - accuracy: 0.9512\n",
      "Epoch 8/10\n",
      "49846/49846 [==============================] - 71s 1ms/step - loss: 0.1192 - accuracy: 0.9513\n",
      "Epoch 9/10\n",
      "49846/49846 [==============================] - 73s 1ms/step - loss: 0.1190 - accuracy: 0.9515\n",
      "Epoch 10/10\n",
      "49846/49846 [==============================] - 73s 1ms/step - loss: 0.1188 - accuracy: 0.9516\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x7f6f56a86f98>"
      ]
     },
     "execution_count": 14,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from keras.utils import to_categorical\n",
    "from keras import models\n",
    "from keras import layers\n",
    "\n",
    "\n",
    "Train_Labels = to_categorical(Target_Train)\n",
    "Test_Labels = to_categorical(Target_Test)\n",
    "network = models.Sequential()\n",
    "network.add(layers.Dense(95, activation='tanh', input_shape=(19,)))\n",
    "network.add(layers.Dense(57, activation='tanh'))\n",
    "network.add(layers.Dense(19, activation='tanh'))\n",
    "network.add(layers.Dense(10, activation='tanh'))\n",
    "network.add(layers.Dense(5, activation='tanh'))\n",
    "network.add(layers.Dense(2, activation='softmax'))\n",
    "network.compile(optimizer='rmsprop',\n",
    "                loss='categorical_crossentropy',\n",
    "                metrics=['accuracy'])\n",
    "network.fit(Input_Train, Train_Labels, epochs=10, batch_size=128)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 85
    },
    "executionInfo": {
     "elapsed": 245328,
     "status": "ok",
     "timestamp": 1602493145943,
     "user": {
      "displayName": "Hadi Mahdipour",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjkcNufEQM-E8yP6nxIfCPrwkowkNMMhCK2zTd8Kg=s64",
      "userId": "04589609695017354496"
     },
     "user_tz": -210
    },
    "id": "k8ZlFfaMxj8r",
    "outputId": "d8f11dcb-1379-46a2-ad5f-ef492c999ee3"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "199383/199383 [==============================] - 202s 1ms/step - loss: 0.1184 - accuracy: 0.9522\n",
      "0.11844507604837418 0.9522455930709839\n",
      "39877/39877 [==============================] - 39s 989us/step - loss: nan - accuracy: 0.9120\n",
      "nan 0.911983072757721\n"
     ]
    }
   ],
   "source": [
    "Train_Loss, Train_Acc = network.evaluate(Input_Train, Train_Labels)\n",
    "print(Train_Loss,Train_Acc)\n",
    "Test_Loss, Test_Acc = network.evaluate(Input_Test, Test_Labels)\n",
    "print(Test_Loss,Test_Acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "uMr7JxqnNDDl"
   },
   "outputs": [],
   "source": [
    "#npArray = np.arange(1, 20, 2)\n",
    "#print(npArray)\n",
    "#print(npArray[5:])\n",
    "#print(npArray[:4])\n",
    "#tmp=Input_Val[:,1:5]\n",
    "#tmp\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Wvfwtt8qMMEJ"
   },
   "outputs": [],
   "source": [
    "#import keras\n",
    "#keras.__version__"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyM5PnkpKjWnt3GKg39c52wf",
   "collapsed_sections": [],
   "name": "First_Storm_Prediction.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
