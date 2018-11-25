load('ANNResult.mat')
load('DecisionTreeResult.mat')
load('polyResult.mat')
load('rbfResult.mat')
load('LinearResult.mat')

A_P = ttest2(ANNResult,polySVMResult);
A_L = ttest2(ANNResult,LinearSVMResult);
A_D = ttest2(ANNResult,DecisionTreeResult);
A_R = ttest2(ANNResult,rbfSVMResult);
D_P = ttest2(DecisionTreeResult,polySVMResult);
D_R = ttest2(DecisionTreeResult,rbfSVMResult);
D_L = ttest2(ANNResult,LinearSVMResult);
P_R = ttest2(polySVMResult,rbfSVMResult);
P_L = ttest2(polySVMResult,LinearSVMResult);
R_L = ttest2(rbfSVMResult,LinearSVMResult);