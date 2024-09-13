%% Copyright (c) 2024, Zhen Mei

%% All rights reserved. Please read the "license.txt" for license terms.

%% This code is A Dynamic Evolving Fuzzy System for Streaming Data Prediction Algorithm described in:
%==========================================================================================================
%% Z. Mei, T. Zhao and X. Gu, "A Dynamic Evolving Fuzzy System for Streaming Data Prediction," in IEEE Transactions on Fuzzy Systems, vol. 32, no. 8, pp. 4324-4337, Aug. 2024, doi: 10.1109/TFUZZ.2024.3395643.
%==========================================================================================================
%% Please cite the paper above if this code helps.

%% For any queries about the code, please contact Dr. ZhenMei
%% meizhen185023@163.com

%% Programmed by ZhenMei

function YTestPre = DEFS(DataTrain, DataTest, xi0, omega0, zeta, chi0, rhoMax, rhoMin)
tic
%% �õ�ѵ�����ݵ�ά��
[TTrain, DimN] = size(DataTrain);
%% ��ʼ��Ԥ�����
% YTrainPre = zeros(TTrain,1);
% ErroTrain = zeros(TTrain,1);
%% Ĭ�ϳ�����
% xi0 = 0.5;
% omega0 = 0.5;
% zeta = 0.95;
% chi0 = 0.05;
% rhoMax = 0.9;
% rhoMin = 0.7;
%% �������
M = DimN-1;
theta0 = 1000;
sigma0 = 1e-10;
gamma = 0.5;
varpi = 0;
xi = xi0;
epsilon = 3;
Delta1 = 2;
Delta2 = 5;
t1 = 0;
%% �ۺϾ��������ز��� 
% Please refer to the paper ``An approach to online identification of Takagi-Sugeno fuzzy models``
alphaX = 1/(1+M);
alphaY = 1-alphaX;
VarThetaX = sum(DataTrain(1,1:end-1).*DataTrain(1,1:end-1),2);
VarThetaY = DataTrain(1,end).*DataTrain(1,end);
SigmaX = VarThetaX;
SigmaY = VarThetaY;
BetaX = DataTrain(1,1:end-1);
BetaY = DataTrain(1,end);
%% ʹ�õ�һ�����ݵ��ʼ��ģ��ϵͳ
C = 1;
N = 1;
m = DataTrain(1,1:M);
a = zeros(1,M+1);
SigmaInv = 1/sigma0*eye(M);
Theta = eye(M+1)*theta0;
v = 1;
eta = 1;
%% ʹ�õ�һ�����ݵ���º��ֵ
t = 1;
RIUse = 1;
datain = DataTrain(t,1:end-1);
dataout = DataTrain(t,end);
dataEx = [1,datain];
Ypre = dataEx*a(RIUse,:)';
ErroTrain = dataout - Ypre;
KIn = Theta(:,:,RIUse)*dataEx'*(1/(zeta+dataEx*Theta(:,:,RIUse)*dataEx'));
A1 = a(RIUse,:)'+ KIn*ErroTrain;
Theta(:,:,RIUse) = 1/zeta*( Theta(:,:,RIUse) - KIn*dataEx*Theta(:,:,RIUse));
a(RIUse,:)=A1';
%% ѵ���׶�
for t = 2:TTrain
    if C >= 50
        disp('��')
    end
    %% �õ����ݵ�
    datain = DataTrain(t,1:M);
    dataout = DataTrain(t,end);
    %% ����������
    Mu = zeros(C,1);
    for RIUse = 1:C
        Mu(RIUse) = exp( -1/2*(datain - m(RIUse,:)) * SigmaInv(:,:,RIUse) * (datain - m(RIUse,:))' )+eps;
    end
    Lamada = Mu./sum(Mu);
    %% ���������ݵ�Ǳ��ֵ
    VarThetaX = sum(datain.*datain,2);
    VarThetaY = dataout.*dataout;
    VX = sum(datain.*BetaX,2);
    VY = sum(dataout.*BetaY,2);
    PT = (t-1)/( (t-1)*(1+alphaX*VarThetaX+alphaY*VarThetaY) + alphaX*(SigmaX-2*VX) + alphaY*(SigmaY-2*VY)+eps );
    %% ����sigma0
    sigma0 = (t-1)^2/t^2*sigma0+4*(1-PT)/(PT*t^2);
    %% ���²���
    SigmaX = SigmaX + VarThetaX;
    SigmaY = SigmaY + VarThetaY;
    BetaX = BetaX + datain;
    BetaY = BetaY + dataout;
    %% ���㵱ǰ���ݵ㵽�������ĵ���С����
    Psi = sqrt(sum((datain - m).*(datain - m),2));
    Psi = Psi';
    OmegaT = PT *min(Psi);
    %% ��þֲ����ģ�ͣ�
    YI = Lamada'*dataout;
    dataEx=[1,datain];
    ErrorI = YI-dataEx*a'.*Lamada';
    ErrorI2 = ErrorI.^2;
    Delta = max(ErrorI2);
    %% �������ݵ�������
    Phi = exp(-abs(ErrorI2)/(sum(abs(ErrorI2))+eps)).*(sum(Psi)./(Psi+eps));
    [~,Kappa] = max(Phi);
    %% �ж��Ƿ���������
    if (Delta>=(varpi(Kappa)+epsilon*xi(Kappa))||OmegaT>=omega0)&&(C<60)&&t1>Delta1
        C = C+1;
        N = [N,1];
        m(C,:) = datain;
        a(C,:) = zeros(1,M+1);
        SigmaInv(:,:,C) = 1/sigma0 * eye(M) ;
        Theta(:,:,C) = theta0 * eye(M+1);
        xi = [xi;xi0];
        varpi = [varpi; 3*Delta;];
        eta = [eta;1];
        v = [v;t];
        Mu = [Mu;1];
        t1 = 0;
    else
        %% ���¾��뵱ǰ������ļ�Ⱥ
        N(Kappa) = N(Kappa)+1;
        SN = N(Kappa);
        m(Kappa,:) = m(Kappa,:) + (datain - m(Kappa,:))/SN;
        FM = (SN-1)^3+SN*(SN-1)*(datain - m(Kappa,:))*SigmaInv(:,:,Kappa)*(datain - m(Kappa,:))';
        SigmaInv(:,:,Kappa) = (SN/(SN-1))*SigmaInv(:,:,Kappa)-(SN^2)*SigmaInv(:,:,Kappa)*(datain - m(Kappa,:))'*(datain - m(Kappa,:))*SigmaInv(:,:,Kappa)/FM;
        xi(Kappa) = (N(Kappa)-1)/N(Kappa)*xi(Kappa)+(N(Kappa)-1)/N(Kappa)^2*(ErrorI2(Kappa)-varpi(Kappa)).^2;
        varpi(Kappa) = varpi(Kappa)+(ErrorI2(Kappa)-varpi(Kappa))/N(Kappa);
    end
    %% �ж��Ƿ�Ҫ�ϲ�����
    if (mod(t,Delta2)==0) && t1 > Delta1
        RhoT = (-(rhoMax - rhoMin)./(1+exp(-10*C/30+5))+rhoMax)^M;
        MuJK = zeros(C,C);
        for i = 1:C
            for j = 1:C
                MuJK(i,j) = exp( -1/2* (m(i,:) - m(j,:)) * SigmaInv(:,:,j) * (m(i,:) - m(j,:))' )+eps;
            end
        end
        MuJK(abs(MuJK-1)<1e-10)=0;
        NuJK = (triu(MuJK)+tril(MuJK)')/2;
        [MaxVaule1,MaxIndex1] = max(NuJK);
        [MaxVaule2,MaxIndex2] = max(MaxVaule1);
        if MaxVaule2 > RhoT
            MergerJ = min(MaxIndex2,MaxIndex1(MaxIndex2));
            MergerK = max(MaxIndex2,MaxIndex1(MaxIndex2));
            %% ���úϲ����¹������
            N = [N,N(MergerJ)+N(MergerK)];
            m(C+1,:) = (m(MergerJ,:)*N(MergerJ)+m(MergerK,:)*N(MergerK))/(N(MergerJ)+N(MergerK));
            SigmaInv(:,:,C+1) =inv( (inv(SigmaInv(:,:,MergerJ))*N(MergerJ)+inv(SigmaInv(:,:,MergerK))*N(MergerK))/(N(MergerJ)+N(MergerK)) );
            a(C+1,:) = (a(MergerJ,:)*N(MergerJ)+a(MergerK,:)*N(MergerK))/(N(MergerJ)+N(MergerK));
            Theta(:,:,C+1) = (Theta(:,:,MergerJ)*N(MergerJ)+Theta(:,:,MergerK)*N(MergerK))/(N(MergerJ)+N(MergerK));
            xi = [xi;(xi(MergerJ)*N(MergerJ)+xi(MergerK)*N(MergerK))/(N(MergerJ)+N(MergerK))];
            varpi = [varpi;(varpi(MergerJ)*N(MergerJ)+varpi(MergerK)*N(MergerK))/(N(MergerJ)+N(MergerK)) ];
            v= [v; t];
            eta = [eta;1];
            MuNow = exp( -1/2* (datain - m(C+1,:)) *SigmaInv(:,:,C+1)*(datain - m(C+1,:))')+eps;
            Mu = [Mu; MuNow];
            %% ɾ�����ϲ��Ĺ������
            MonDelIndex = [MergerJ,MergerK];
            v(MonDelIndex) = [];
            N(MonDelIndex) = [];
            a(MonDelIndex,:) = [];
            Theta(:,:,MonDelIndex) = [];
            SigmaInv(:,:,MonDelIndex) = [];
            m(MonDelIndex,:) = [];
            Mu(MonDelIndex)=[];
            eta(MonDelIndex)=[];
            varpi(MonDelIndex) = [];
            xi(MonDelIndex) = [];
            C = C-1;
        end
    end
    %% ����Ч��ֵ
    if t1 > Delta1
        Lamada = Mu./sum(Mu);
        eta = eta + (Lamada - eta)./(t*ones(length(v),1)-v+1);
        Chi = eta./(eps+sqrt(varpi));
        MonDelIndex = find( Chi  < chi0);
        if ~isempty(MonDelIndex) && C > length(MonDelIndex)
            %% ɾ������
            v(MonDelIndex) = [];
            N(MonDelIndex) = [];
            a(MonDelIndex,:) = [];
            Theta(:,:,MonDelIndex) = [];
            SigmaInv(:,:,MonDelIndex) = [];
            m(MonDelIndex,:) = [];
            Mu(MonDelIndex)=[];
            Lamada(MonDelIndex)=[];
            eta(MonDelIndex)=[];
            varpi(MonDelIndex) = [];
            xi(MonDelIndex) = [];
            C = C-length(MonDelIndex);
        end
    end
    %% �������
    t1 = t1+1;
    Mu = zeros(C,1);
    for RIUse = 1:C
        Mu(RIUse) = exp( -1/2* (datain - m(RIUse,:)) * SigmaInv(:,:,RIUse) * (datain - m(RIUse,:))' )+eps;
    end
    AddMu = sum(Mu);
    [MuStar,RStar] = sort(Mu,'descend');
    CStar = find((cumsum(MuStar)-gamma*AddMu)>=0);
    RUse = RStar(1:CStar);
    MuUse = Mu(RUse);
    Lamada = MuUse./(sum(MuUse)+eps);
    dataEx=[1,datain];
    Ypre = dataEx*a(RUse,:)'*Lamada;
    ErroTrain = dataout - Ypre;
    %% ʹ�ô��������ӵľֲ���С�����㷨
    for RI = 1:length(RUse)
        RIUse = RUse(RI);
        KIn = Theta(:,:,RIUse)*dataEx'*(Lamada(RI) /( zeta + Lamada(RI)*dataEx*Theta(:,:,RIUse)*dataEx'));
        A1 = a(RIUse,:)'+KIn*ErroTrain;
        Theta(:,:,RIUse) =1/zeta*( Theta(:,:,RIUse) - KIn*dataEx*Theta(:,:,RIUse));
        a(RIUse,:)=A1';
    end
end
%% ���Խ׶�
TTest = length(DataTest);
ErrorTest = zeros(TTest,1);
YTestPre = zeros(TTest,1);
Mu = zeros(C,1);
for t = 1:TTest
    datain = DataTest(t,1:M);
    dataout = DataTest(t,end);
    for RIUse = 1:C
        Mu(RIUse) = exp( -1/2 * (datain - m(RIUse,:)) * SigmaInv(:,:,RIUse) * (datain - m(RIUse,:))' )+eps;
    end
    %% �õ�����Ҫ��Ĺ���
    AddMu = sum(Mu);
    [MuStar,RStar] = sort(Mu,'descend');
    CStar = find((cumsum(MuStar)-gamma*AddMu)>0);
    RUse = RStar(1:CStar);
    MuUse = Mu(RUse);
    Lamada = MuUse./(sum(MuUse)+eps);
    dataEx=[1,datain];
    YTestPre(t) = dataEx*a(RUse,:)'*Lamada;
    ErrorTest(t) = dataout - YTestPre(t);
end
toc
%% ������
RMSE = sqrt(sum(ErrorTest.^2)/TTest);
NDEI = RMSE/std(DataTest(:,end));
disp([' RMSE is ' num2str(RMSE)]);
disp([' NDEI is ' num2str(NDEI)]);
disp([' Model Rule Number is ' num2str(C)])