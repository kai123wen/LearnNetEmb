% https://linqs.soe.ucsc.edu/data
% ���Ƶ����ݼ�

% Ӧ����ֻҪ����A ������ ����B ����ô(A,B) 1

% load data
net = load('citeseer-undirected.mat');
A = net.network;
group = net.group;
% disp(group)
disp(A)

% �õ����ĵ�ƪ��
N = size(A,1);

% ��A���� �����
% disp(sum(A,2))

% S = sparse(1:10,1:10,2);
% disp(S) 
% ���ɽ����(1,1)        2
%            (2,2)        2

% �õ�������1./sqrt(sum(A,2))�����Ƶ�λ����

D = sparse(1:N,1:N,1./sqrt(sum(A,2)));

A = D*A*D; % symmetric transition matrix

A = (A + A*A + A*A*A + A*A*A*A)/4;

% symmetric PPMI
D = sparse(1:N,1:N,1./sqrt(sum(A,2)));

A = D*A*D;

% disp(A);

network = max(log(A) - log(1/N), 0);
network = sparse(network);
% disp(network)
save('citeseer-PPMI-4.mat', 'network', 'group');
