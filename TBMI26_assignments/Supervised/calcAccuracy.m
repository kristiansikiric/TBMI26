function [ acc ] = calcAccuracy( cM )
%CALCACCURACY Takes a confusion matrix amd calculates the accuracy

acc = sum(diag(cM))/sum(sum(cM)); % Replace with your own code

end

