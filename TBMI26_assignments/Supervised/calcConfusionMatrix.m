function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

% Add your own code here

for i = 1:numClasses
    ind = Ltrue == i; %extract true class
    for j = 1:numClasses
        cM(i,j)=sum(Lclass(ind) == j); 
    end
end


end

