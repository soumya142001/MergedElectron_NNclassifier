import uproot
import numpy as np
import awkward as ak
import awkward_pandas as akpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Function Definition for Scaling any dataframe variable from lowerbound to upperbound    
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
#filename = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_Z.root")
filename = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_HNL_trilepton_M2_V0p01_e.root")
#Extracting the trees from the root files
tree_name = 'myVariables'
tree     = filename[tree_name]

#Extracting the branches and creating the datframe
branches = tree.keys()
awk = tree.arrays(branches)
#awk = tree.arrays(filter_name='gsf*')
df = pd.DataFrame(awk.to_list())


#Training Variables used for Training the NN

trainingvars = ['gsfErawOvertrackP','gsfEcorrOvertrackP','gsfhcalOverEcal','gsfdeltaErawEcorrOverEcorr',
'gsf_PtdiffgsfgsftrackOverPtgsf','gsfsigmaEtaEta','gsfsigmaIetaIeta','gsfsigmaIphiIphi','gsfr9',
'gsfdeltaEtaSuperClusterTrackAtVtx','gsfdeltaEtaSeedClusterTrackAtCalo',
'gsfdeltaPhiSuperClusterTrackAtVtx','gsfdeltaPhiSeedClusterTrackAtCalo']

listofvars = trainingvars+['gsfEventno']

lowerbound   = [1,1,0,0,-2,0,0,0,-0.04,-0.04,-0.04,-0.04]
upperbound   = [50,50,0.1,0.6,1,0.03,0.04,0.05,0.04,0.04,0.04,0.04]
lowerbound   = np.array(lowerbound)
upperbound   = np.array(upperbound)

#Selecting the dataset on which training should be performed and scaling its variables
X_df = df[listofvars].apply(lambda col: col.explode()).reset_index(drop=True)

X = X_df[trainingvars]

Scaling(X,upperbound,lowerbound,trainingvars)

#Converting all objects to tensorflow objects
import tensorflow as tf

X = tf.convert_to_tensor(X, dtype=tf.float32)

#Importing the Model
modelname = "Models/mymodel_v25.h5"
model = tf.keras.models.load_model(modelname)
model.load_weights(modelname)

y = model.predict(X)
#y=np.ones(len(X))

#Adding the NNscore branch to the dataframe
X_df['NNscore'] = y

# Grouping events by event number so that again from object level we can go to event level (Original Dataframe)
grouped_data = X_df.groupby('gsfEventno')['NNscore'].apply(list).reset_index(name='Grouped_Score')
awk_score = ak.Array(grouped_data['Grouped_Score'].tolist())

df['NNscore'] = awk_score

print(df[['gsfEventno','trkdR0p1_Mtrktrk','NNscore']])

#Defining the Original Branches as a dictionary
#orig_branches = tree.arrays(filter_name='gsf*', library='ak')

orig_branches = df.filter(regex='^gsf|NN|trk').to_dict('list') #Searching for original branches using regex command
#trk_branches = df.filter(regex='^trk').to_dict('list')

#del orig_branches['ngsf'],orig_branches['ngsftrack'],orig_branches['ngenelectrons'],orig_branches['gsfEventno'],orig_branches['ngsftrack_dR0p1']


filename.close()


#Opening file in the update mode to add the branch
file_updated = uproot.update("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_HNL_trilepton_M2_V0p01_e.root")

#branch = {'Pt' : gsfPt, 'Eta' : gsfEta, 'Phi' : gsfPhi, 'seedtrackPt' : gsfseedtrackPt, 'seedtrackEta' : gsfseedtrackEta, 'NNscore' : NNscore}
#file_updated['Mergedelectron_NNScore'] = {'gsf' : ak.zip(branch)}
#file_updated['Mergedelectron_NNScore'] = {'Gsf' : ak.zip(orig_branches), 'Gsftrack' : ak.zip(trk_branches)}
#file_updated['Mergedelectron_NNScore'] = df

file_updated['Mergedelectron_NNScore'] = {'Gsf' : ak.zip(orig_branches)}

file_updated.close()









