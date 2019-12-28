% https://linqs.soe.ucsc.edu/data
% 相似的数据集

% 应该是只要论文A 引用了 论文B ，那么(A,B) 1

% load data
net = load('citeseer-undirected.mat');
A = net.network;
group = net.group;
% disp(group)
disp(A)

% 得到论文的篇数
N = size(A,1);

% 对A进行 行求和
% disp(sum(A,2))

% S = sparse(1:10,1:10,2);
% disp(S) 
% 生成结果：(1,1)        2
%            (2,2)        2

% 得到的是以1./sqrt(sum(A,2))的类似单位矩阵

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
