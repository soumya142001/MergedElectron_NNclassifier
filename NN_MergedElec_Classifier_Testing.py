######################################################################################################################

# This is a testing code. The model is saved as .h5 file and here we test how the model performs on random data

######################################################################################################################


import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import os
import warnings
warnings.filterwarnings('ignore')
import sys

#Function Definition for Scaling any dataframe variable from -1 to 1    
# Scaling of variables

def Scaling(X,ub,lb,columns):
    ran = np.subtract(ub,lb)
    for i in range(len(ran)):
        colname = columns[i]
        scale   = ran[i]
        Y       = X.loc[:, X.columns == colname]
        Y       = 2*(Y/scale)-1
        X.loc[:, X.columns == colname] = Y
    

#Importing the root files using uproot
fileJPsi = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_JPsi.root")
fileZ = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_Z.root")
#fileQCD = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_QCD.root")

#Extracting the trees from the root files
tree_name = "myVariables"
treeJPsi = fileJPsi[tree_name]
treeZ = fileZ[tree_name]
#treeQCD = fileQCD[tree_name]


#Extracting the branches and creating the datframe
branchesJPsi = treeJPsi.keys()
branchesZ = treeZ.keys()
#branchesQCD = treeQCD.keys()

awkJPsi = treeJPsi.arrays(branchesJPsi)
awkZ = treeZ.arrays(branchesZ)
#awkQCD = treeQCD.arrays(branchesQCD)
df_jpsi = pd.DataFrame(awkJPsi.to_list())
df_z = pd.DataFrame(awkZ.to_list())
#df_qcd = pd.DataFrame(awkQCD.to_list())

#List of variables will be needing a lot for object selection

listofvars = ['ngsf','ngsftrack','ngsftrack_dR0p1','gendR']


#Training Variables used for Training the NN

trainingvars = ['gsfErawOvertrackP','gsfEcorrOvertrackP','gsfhcalOverEcal','gsfdeltaErawEcorrOverEcorr',
'gsf_PtdiffgsfgsftrackOverPtgsf','gsfsigmaEtaEta','gsfsigmaIetaIeta','gsfsigmaIphiIphi','gsfr9',
'gsfdeltaEtaSuperClusterTrackAtVtx','gsfdeltaEtaSeedClusterTrackAtCalo',
'gsfdeltaPhiSuperClusterTrackAtVtx','gsfdeltaPhiSeedClusterTrackAtCalo']

lowerbound   = [1,1,0,0,-2,0,0,0,-0.04,-0.04,-0.04,-0.04]
upperbound   = [50,50,0.1,0.6,1,0.03,0.04,0.05,0.04,0.04,0.04,0.04]
lowerbound   = np.array(lowerbound)
upperbound   = np.array(upperbound)

                
#Selecting the dataset on which training should be performed and scaling its variables                
df_jpsi = df_jpsi[(df_jpsi['ngsf']==1)]
X = df_jpsi[trainingvars].apply(lambda col: col.explode()).reset_index(drop=True)

Scaling(X,upperbound,lowerbound,trainingvars)


#Converting all objects to tensorflow objects
import tensorflow as tf

X = tf.convert_to_tensor(X, dtype=tf.float32)

#Importing the Model
#modelname = "/home/soumya/CMSSW_files/NTuples/mergedelectron_model.h5"
modelname = "Models/mymodel_v24.h5"
model = tf.keras.models.load_model(modelname)
model.load_weights(modelname)
model.summary()
y = model.predict(X)

print(len(y))
print(len(y[y>0.8]))

plt.hist(y,bins=np.arange(0,1,0.01),color='green',histtype='stepfilled',weights=np.ones_like(np.array(y))/len(np.array(y)))
plt.xlabel('NN score',fontsize=20)
plt.ylabel('Probability',fontsize=20)
plt.show()




