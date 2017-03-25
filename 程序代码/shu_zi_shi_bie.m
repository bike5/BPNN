clear all;
%网络创建
load data;
[R,Q]=size(number);
P=number;
T=targets;
% NodeNum=2;
NodeNum=12; %隐含层节点
% NodeNum=16;
% NodeNum=300;

% BTF='traingd';%梯度下降法 标准BP算法
% BTF='traingdm';%动量梯度下降算法
% BTF='trainlm';%L-M优化算法  数值优化算法  
BTF='traingdx'; %变学习率动量梯度下降算法   启发式学习算法

TF1='logsig';%隐含层传递函数
TF2='logsig';%输出层传递函数
% TF1='tansig';%隐含层传递函数
% TF2='tansig';%输出层传递函数
% TF1='logsig';%隐含层传递函数
% TF2='tansig';%输出层传递函数  %稍慢
% TF1='tansig';%隐含层传递函数
% TF2='logsig';%输出层传递函数  %稍快

BLF='learngdm';
TypeNum=10;%输出层节点
net=newff(minmax(P),[NodeNum,TypeNum],{TF1,TF2},BTF,BLF);
net.LW{1,1}=net.LW{1,1}*0.01;%调整权值  
net.b{1}=net.b{1}*0.01;     %1表示在隐层和输出层之间的阈值
net.LW{2,1}=net.LW{2,1}*0.01;%调整权值     %2表示隐层和输出层之间的权值 1表示第一个输入向量
net.b{2}=net.b{2}*0.01;%调整阈值   %使得初始权值足够小，加快学习速率
%网络训练
net.trainParam.goal=0.001;  %学习目标
net.trainParam.epochs=5000;  %迭代次数
net=train(net,P,T);
%网络测试
A=sim(net,P);%此时输出并不是标准的0-1值，需要使用compet函数进行调整
AA=compet(A)


%有噪声输入来训练网络,共四组输入，两组未加噪两组加噪，10次训练
netn=net;
netn.trainParam.goal=0.001;
netn.trainParam.epochs=6000;
T=[targets targets targets targets];
for i=1:10
    P=[number,number,(number+randn(R,Q)*0.3),(number+randn(R,Q)*0.2)];
    netn=train(netn,P,T);
end
%使用加入不同程度噪声的输入向量进行网络性能测量，训练100次
noise_range=0:0.05:0.5;
T=targets;
max_test=100;
for i=1:11
    error1(i)=0;
    error2(i)=0;
    for j=1:max_test
        %未加噪输入进行网络性能测试
       P=number+randn(R,Q)*noise_range(i);
       A=sim(net,P);
       AA=compet(A);%compet竞争传递函数使得矩阵每一列中最大的为1，其他的为零，这样可以将网络的训练输出
       error1(i)=error1(i)+sum(sum(abs(AA-T)))/2
       %加噪输入进行网络性能测试
       An=sim(netn,P);
       AAn=compet(An);
       error2(i)=error2(i)+sum(sum(abs(AAn-T)))/2  %sum（）对矩阵的列求和
    end
end
figure;
plot(noise_range,error1,'--',noise_range,error2,'-')
title('网络识别误差'),xlabel('噪声程度'),ylabel('误差性能')
legend('无噪声训练网络','有噪声训练网络')

%对污染数字进行识别  以噪声指标为0.5的噪声信号对数字进行污染 
err=0.4;
figure;
for i=1:10
       number_noise=number(:,i)+randn(R,1)*err;%对某一列加噪
       subplot(4,5,i);
       plotchar(number_noise);%画出污染数字
      % 画出识别结果
%        A3=sim(netn,number_noise);
%        A3=compet(A3);
%        answer=find(A3==1);
%        subplot(4,5,i+10);
%        plotchar(number(:,answer));
end  
