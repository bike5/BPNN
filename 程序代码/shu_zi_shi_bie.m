clear all;
%���紴��
load data;
[R,Q]=size(number);
P=number;
T=targets;
% NodeNum=2;
NodeNum=12; %������ڵ�
% NodeNum=16;
% NodeNum=300;

% BTF='traingd';%�ݶ��½��� ��׼BP�㷨
% BTF='traingdm';%�����ݶ��½��㷨
% BTF='trainlm';%L-M�Ż��㷨  ��ֵ�Ż��㷨  
BTF='traingdx'; %��ѧϰ�ʶ����ݶ��½��㷨   ����ʽѧϰ�㷨

TF1='logsig';%�����㴫�ݺ���
TF2='logsig';%����㴫�ݺ���
% TF1='tansig';%�����㴫�ݺ���
% TF2='tansig';%����㴫�ݺ���
% TF1='logsig';%�����㴫�ݺ���
% TF2='tansig';%����㴫�ݺ���  %����
% TF1='tansig';%�����㴫�ݺ���
% TF2='logsig';%����㴫�ݺ���  %�Կ�

BLF='learngdm';
TypeNum=10;%�����ڵ�
net=newff(minmax(P),[NodeNum,TypeNum],{TF1,TF2},BTF,BLF);
net.LW{1,1}=net.LW{1,1}*0.01;%����Ȩֵ  
net.b{1}=net.b{1}*0.01;     %1��ʾ������������֮�����ֵ
net.LW{2,1}=net.LW{2,1}*0.01;%����Ȩֵ     %2��ʾ����������֮���Ȩֵ 1��ʾ��һ����������
net.b{2}=net.b{2}*0.01;%������ֵ   %ʹ�ó�ʼȨֵ�㹻С���ӿ�ѧϰ����
%����ѵ��
net.trainParam.goal=0.001;  %ѧϰĿ��
net.trainParam.epochs=5000;  %��������
net=train(net,P,T);
%�������
A=sim(net,P);%��ʱ��������Ǳ�׼��0-1ֵ����Ҫʹ��compet�������е���
AA=compet(A)


%������������ѵ������,���������룬����δ����������룬10��ѵ��
netn=net;
netn.trainParam.goal=0.001;
netn.trainParam.epochs=6000;
T=[targets targets targets targets];
for i=1:10
    P=[number,number,(number+randn(R,Q)*0.3),(number+randn(R,Q)*0.2)];
    netn=train(netn,P,T);
end
%ʹ�ü��벻ͬ�̶��������������������������ܲ�����ѵ��100��
noise_range=0:0.05:0.5;
T=targets;
max_test=100;
for i=1:11
    error1(i)=0;
    error2(i)=0;
    for j=1:max_test
        %δ������������������ܲ���
       P=number+randn(R,Q)*noise_range(i);
       A=sim(net,P);
       AA=compet(A);%compet�������ݺ���ʹ�þ���ÿһ��������Ϊ1��������Ϊ�㣬�������Խ������ѵ�����
       error1(i)=error1(i)+sum(sum(abs(AA-T)))/2
       %������������������ܲ���
       An=sim(netn,P);
       AAn=compet(An);
       error2(i)=error2(i)+sum(sum(abs(AAn-T)))/2  %sum�����Ծ���������
    end
end
figure;
plot(noise_range,error1,'--',noise_range,error2,'-')
title('����ʶ�����'),xlabel('�����̶�'),ylabel('�������')
legend('������ѵ������','������ѵ������')

%����Ⱦ���ֽ���ʶ��  ������ָ��Ϊ0.5�������źŶ����ֽ�����Ⱦ 
err=0.4;
figure;
for i=1:10
       number_noise=number(:,i)+randn(R,1)*err;%��ĳһ�м���
       subplot(4,5,i);
       plotchar(number_noise);%������Ⱦ����
      % ����ʶ����
%        A3=sim(netn,number_noise);
%        A3=compet(A3);
%        answer=find(A3==1);
%        subplot(4,5,i+10);
%        plotchar(number(:,answer));
end  
