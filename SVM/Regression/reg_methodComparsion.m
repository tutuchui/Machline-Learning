load('ANNResult.mat')
load('polyResult.mat')
load('rbfResult.mat')
load('LinearResult.mat')

A_P = ttest2(ANNResult,polySVMResult);
A_L = ttest2(ANNResult,LinearSVMResult);
A_R = ttest2(ANNResult,rbfSVMResult);
P_R = ttest2(polySVMResult,rbfSVMResult);
P_L = ttest2(polySVMResult,LinearSVMResult);
R_L = ttest2(rbfSVMResult,LinearSVMResult);