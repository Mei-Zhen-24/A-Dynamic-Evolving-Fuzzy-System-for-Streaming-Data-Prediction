clc;clear;close all
%% 读取数据
load Data_Exp1.mat
DataTrain = [TrainInput,TrainOutput];
DataTest = [TestInput,TestOutput];
%% 定义超参数
xi0 = 1;
omega0 = 0.3;
zeta = 0.95;
chi0 = 0.05;
rhoMax = 0.9;
rhoMin = 0.7;
%% 开始
YTestPre = DEFS(DataTrain, DataTest, xi0, omega0, zeta, chi0, rhoMax, rhoMin);