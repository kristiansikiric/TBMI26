function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);

%Distance between features
D = pdist2(X',Xt','euclidean');
%Sort the distances
[~,nearest] = sort(D,2,'ascend');
labelsOut = mode(Lt(nearest(:,1:k)),2);


end

