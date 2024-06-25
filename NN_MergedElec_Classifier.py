#############################################################################################

# This is a classification code that differentiates merged electrons from single electrons. 
# Source of merged electrons is J/Psi and for single electrons its Z sample

# To run this code on the terminal :
#python NN_MergedElec_Classifier.py <PDF name> <Model Name>
#For eg. python NN_MergedElec_Classifier.py output.pdf mymodel.h5

#############################################################################################


#Importing the Modules
import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Importing the OS
import os
import warnings
warnings.filterwarnings('ignore')
import sys

#Output PDF
outputname = 'Output_PDF/'+sys.argv[1]
modelname = sys.argv[2]

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

# Scaling of variables
def Scaling(X,ub,lb,columns):
    ran = np.subtract(ub,lb)
    for i in range(len(ran)):
        colname = columns[i]
        scale   = ran[i]
        Y       = X.loc[:, X.columns == colname]
        Y       = 2*(Y/scale)-1
        X.loc[:, X.columns == colname] = Y


#Function for pT and eta reweighting
def pTetaweight(pTbins,etabins,df1,df2):
    df1['pTetaweight'] = 1
    df2['pTetaweight'] = 1
    for i in range(len(etabins)-1):
        for j in range(len(pTbins)-1):
            eta1 = df1['gsfEta']
            pT1 = df1['gsfPt']
            eta2 = df2['gsfEta']
            pT2 = df2['gsfPt']
            n1 = len(df1[(eta1 < etabins[i+1]) & (eta1 > etabins[i]) & (pT1 < pTbins[j+1]) & (pT1 > pTbins[j])])
            n2 = len(df2[(eta2 < etabins[i+1]) & (eta2 > etabins[i]) & (pT2 < pTbins[j+1]) & (pT2 > pTbins[j])])
            df2.loc[(eta2 < etabins[i+1]) & (eta2 > etabins[i]) & (pT2 < pTbins[j+1]) & ( pT2 > pTbins[j]),'pTetaweight'] = n1/n2 
            
#Function for pT reweighting
def pTweight(pTbins,df1,df2):
    df1['pTweight'] = 1
    df2['pTweight'] = 1
    for i in range(len(pTbins)-1):
        pT1 = df1['gsfPt']
        pT2 = df2['gsfPt']
        n1 = len(df1[(pT1 < pTbins[i+1]) & (pT1 > pTbins[i])])
        n2 = len(df2[(pT2 < pTbins[i+1]) & (pT2 > pTbins[i])])
        df2.loc[(pT2 < pTbins[i+1]) & (pT2 > pTbins[i]),'pTweight'] = n1/n2
        
#Function for eta reweighting     
def etaweight(etabins,df1,df2):
    df1['etaweight'] = 1
    df2['etaweight'] = 1
    for i in range(len(etabins)-1):
        eta1 = df1['gsfEta']
        eta2 = df2['gsfEta']
        n1 = len(df1[(eta1 < etabins[i+1]) & (eta1 > etabins[i])])
        n2 = len(df2[(eta2 < etabins[i+1]) & (eta2 > etabins[i])])
        df2.loc[(eta2 < etabins[i+1]) & (eta2 > etabins[i]),'etaweight'] = n1/n2 


#Importing the root files using uproot
fileJPsi = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_JPsi.root")
fileZ = uproot.open("/home/soumya/CMSSW_files/NTuples/Analyzer/myTree_rootfiles/myTree_Z.root")

#Extracting the trees from the root files
tree_name = "myVariables"
treeJPsi = fileJPsi[tree_name]
treeZ = fileZ[tree_name]

#Extracting the branches and creating the datframe
branchesJPsi = treeJPsi.keys()
branchesZ = treeZ.keys()

awkJPsi = treeJPsi.arrays(branchesJPsi)
awkZ = treeZ.arrays(branchesZ)
df_jpsi = pd.DataFrame(awkJPsi.to_list())
df_z = pd.DataFrame(awkZ.to_list())

#df_jpsi = treeJPsi.arrays(library='pd')
#df_z = treeZ.arrays(library='pd')


#Declaring some branches from Tree, many of which will be later used for training. Right now these branches will be helpful in exploding the dataframe.
listofvars = ['gsfPt', 'gsfEta', 'gsfPhi', 'gsfseedtrackPt', 'gsfseedtrackEta', 'gsfseedtrackPhi', 'gsfseedtrackP', 'gsfsigmaEtaEta', 
              'gsfsigmaIetaIeta', 'gsfsigmaIphiIphi', 'gsfr9', 'gsfeSuperClusterOverP', 'gsfeSeedClusterOverP', 'gsfdeltaEtaSuperClusterTrackAtVtx', 
              'gsfdeltaEtaSeedClusterTrackAtCalo', 'gsfdeltaEtaEleClusterTrackAtCalo', 'gsfdeltaEtaSeedClusterTrackAtVtx',
              'gsfdeltaPhiSuperClusterTrackAtVtx', 'gsfdeltaPhiSeedClusterTrackAtCalo', 'gsfdeltaPhiEleClusterTrackAtCalo', 'gsfdeltaErawEcorrOverEcorr', 
              'gsfhcalOverEcal', 'gsfhcalOverEcalBc','gsfrawEnergy','gsfcorrectedEcalEnergy','gsfErawOvertrackP','gsfEcorrOvertrackP','gsf_PtdiffgsfgsftrackOverPtgsf']

#Selecting dataframes for Training 
df_z = df_z[df_z['ngsf']==2]
df_jpsi = df_jpsi[df_jpsi['ngsf']==1]


#exploding the dataframe so that we look at object level and not event level
df_z = df_z[listofvars].apply(lambda col: col.explode()).reset_index(drop=True)
df_jpsi = df_jpsi[listofvars].apply(lambda col: col.explode()).reset_index(drop=True)

print(df_z)

#Defining pT and eta bins
pTbins = [0,10,20,50,100,250,1000]
etabins = [-3,-2,-1,0,1,2,3]

pTetaweight(pTbins,etabins,df_jpsi,df_z)

#Storing the 2D histograms and their info
hz, xedges, yedges, image = plt.hist2d(df_z['gsfEta'],df_z['gsfPt'],[etabins,pTbins],weights=df_z['pTetaweight'],alpha=0.5)
hj, xedges1, yedges1, image1 = plt.hist2d(df_jpsi['gsfEta'],df_jpsi['gsfPt'],[etabins,pTbins],weights=df_jpsi['pTetaweight'],alpha=0.5)

#Plotting the difference of two histograms. It should be very close to 0 for the two to match
plot = plt.pcolormesh(xedges, yedges, (hz-hj).T)
plt.colorbar(plot)
plt.savefig(pp, format='pdf')
plt.close()


trainingvars = ['gsfErawOvertrackP','gsfEcorrOvertrackP','gsfhcalOverEcal','gsfdeltaErawEcorrOverEcorr','gsf_PtdiffgsfgsftrackOverPtgsf','gsfsigmaEtaEta','gsfsigmaIetaIeta','gsfsigmaIphiIphi',
                'gsfr9','gsfdeltaEtaSuperClusterTrackAtVtx','gsfdeltaEtaSeedClusterTrackAtCalo','gsfdeltaPhiSuperClusterTrackAtVtx','gsfdeltaPhiSeedClusterTrackAtCalo']
lowerbound   = [1,1,0,0,-2,0,0,0,-0.04,-0.04,-0.04,-0.04]
upperbound   = [50,50,0.1,0.6,1,0.03,0.04,0.05,0.04,0.04,0.04,0.04]

#Giving Labels to the Signal-0 and Background-1 (Signal - J/Psi and Bkg - Z)
df_jpsi['label']=1
df_z['label']=0
        
listofvars = listofvars+['pTetaweight']        
 
data = pd.concat([df_jpsi,df_z])

X = data[listofvars]
y = data['label']

#Splitting the data into training and testing datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

#Stroring pT eta weights for training
weight_test  = X_test['pTetaweight']
weight_train = X_train['pTetaweight']

X_train = X_train[trainingvars]
X_test  = X_test[trainingvars]

#Scaling the variables from by putting hard cutoff on upper bound and lower bound by hand
Scaling(X_train,upperbound,lowerbound,trainingvars)
Scaling(X_test,upperbound,lowerbound,trainingvars)

n_features = X_train.shape[1] 

#Defining the Model
model = Sequential()
model.add(Dense(256, activation='relu', kernel_initializer='he_normal',input_dim=n_features))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Converting all objects to tensorflow objects (for training)
import tensorflow as tf

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
weight_test = tf.convert_to_tensor(weight_test, dtype=tf.float32)
weight_train = tf.convert_to_tensor(weight_train, dtype=tf.float32)
        
#Training the model
history = model.fit(X_train,y_train,epochs=50,batch_size=512,validation_data=(X_test,y_test),verbose=1,sample_weight = weight_train)       
        
model.summary()
model.save('Models/'+modelname)   

#Plotting the Loss function
plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 10])
#plt.yscale('log')
plt.legend(loc='upper right')
#plt.savefig('loss_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()   

#predicted value of y for training and testing from the model
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

#getting the label value of signal(JPsi i.e 1) and background(Z i.e 0) for test data
true_val=1
y_test_pred_signal = [y_test_pred[i] for i in range(len(y_test)) if y_test[i]==true_val]

true_val=0
y_test_pred_bkg = [y_test_pred[i] for i in range(len(y_test)) if y_test[i]==true_val]

#getting the label value of signal(JPsi i.e 1) and background(Z i.e 0) for training data
true_val=1
y_train_pred_signal = [y_train_pred[i] for i in range(len(y_train)) if y_train[i]==true_val]

true_val=0
y_train_pred_bkg = [y_train_pred[i] for i in range(len(y_train)) if y_train[i]==true_val]

#Plotting the NN scores for testing and training data
plt.hist(np.array(y_test_pred_signal),label='test score sig',color='red',alpha=0.7,bins=np.arange(0,1,0.01),histtype='stepfilled',weights=np.ones_like(np.array(y_test_pred_signal))/len(np.array(y_test_pred_signal)))
plt.hist(np.array(y_test_pred_bkg),label='test score bkg',color='green',alpha=0.7,bins=np.arange(0,1,0.01),histtype='stepfilled',weights=np.ones_like(np.array(y_test_pred_bkg))/len(np.array(y_test_pred_bkg)))

plt.hist(np.array(y_train_pred_signal),label='train score sig',color='red',linewidth=2,bins=np.arange(0,1,0.01),histtype='step',weights=np.ones_like(np.array(y_train_pred_signal))/len(np.array(y_train_pred_signal)))
plt.hist(np.array(y_train_pred_bkg),label='train score bkg',color='green',linewidth=2,bins=np.arange(0,1,0.01),histtype='step',weights=np.ones_like(np.array(y_train_pred_bkg))/len(np.array(y_train_pred_bkg)))
plt.xlabel('Predicted Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.legend()
plt.savefig(pp, format='pdf')
plt.close()

#ROC curve for testing
fpr,tpr,_ = roc_curve(y_test,y_test_pred)
auc_score_test = auc(fpr,tpr)

#ROC curve for Training

fpr1,tpr1,_ = roc_curve(y_train,y_train_pred)
auc_score_train = auc(fpr1,tpr1)

#Plotting the ROC
plt.figure(figsize=(8,8))
plt.plot(fpr,tpr,color='blue', label='Testing ROC (AUC = %0.4f)' % auc_score_test)
plt.plot(fpr1,tpr1,color='red', label='Training ROC (AUC = %0.4f)' % auc_score_train)
plt.legend(loc='lower right')
plt.title(f'ROC Curve',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlim(0.,1.)
plt.ylim(0.,1.)
plt.show()
plt.savefig(pp, format='pdf')
plt.close()

pp.close()

print(f'All done. Model is saved as {modelname}')
        



