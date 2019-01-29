function [ acc ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels
% 
% n = length(Xt);
% Xt1 = Xt(1:n/2);
% Xt2 = Xt(n/2+1:end);

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);

%distance from X training data Xt
D = pdist2(X',Xt','euclidean');
%sort rows of D in ascending order. nearest describes where in D the sorted values are? 
[~,nearest] = sort(D,2,'ascend');
labelsOut = mode(Lt(nearest(:,1:k)),2);

cM = calcConfusionMatrix(labelsOut,Lt);
accuracy(1) = calcAccuracy(cM);

D = pdist2(Xt',X','euclidean');
%sort rows of D in ascending order. nearest describes where in D the sorted values are? 
[~,nearest] = sort(D,2,'ascend');
labelsOut = mode(Lt(nearest(:,1:k)),2);

cM = calcConfusionMatrix(labelsOut,Lt);
accuracy(2) = calcAccuracy(cM);

acc = mean(accuracy);
end

