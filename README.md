# MergedElectron_NNclassifier
This repository contains Neural Network codes to classify merged and single electrons (Specific to this work). Also, it contains Python code to add any column present in the pandas dataframe as a branch in a tree to a root file. 
The steps and use of files are outlined below:
1. The file `NN_MergedElec_Classifier.py` contains the training code for the NN. The instruction on how to run the code is given as comments on top of the file. The output of the above code is a trained model saved as a .h5 file and a PDF file containing the plots of loss as a function of epochs, The NN score plot, and the ROC curve.
2. `NN_MergedElec_Classifier_Testing.py` contains the NN code for testing the model. The model saved as a .h5 file in the previous step is used in this code.
3. The models are stored in `Models/`directory. The output to the NN training is stored in the `Output_PDF/` directory. The `Output_PDF/` directory contains the performance for each trained model with each PDF having 2D histograms to show that pT-eta reweighting is done correctly, a loss function plot, an NN score plot, and an ROC curve. You are free to choose the model whose performance you think is good enough!
4. The list of variables used for training, the NN architecture, the number of epochs, and the upper and lower bounds of values of variables used for scaling are given in the file `training_variables.txt`
5. `NNscore_branchadder.py` adds the branch which contains the NNscore to the actual root file. But this code can be used as a general code for adding any column of the pandas dataframe as a branch to the root file. The exciting thing about this code is that it preserves the awkward structure of the actual tree!

   
