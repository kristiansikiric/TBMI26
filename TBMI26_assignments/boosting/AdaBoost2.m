%% Hyper-parameters
%  You will need to change these. Start with a small number and increase
%  when your algorithm is working.

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
%  Note that the data sets are shuffled each time you run the script.
%  This is to prevent a solution that is tailored to specific images.

load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do NOT modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

D = ones(1,size(xTrain,2))*1/size(xTrain,2);
%H = zeros(1,size(xTrain,2));
threshold = 0;
params = zeros(4,nbrWeakClassifiers);
bestC = 0;
bestX = 0;
bestP = 0;
for i = 1:nbrWeakClassifiers
    Emin = inf;
    for x = 1 : size(xTrain,1) 
        for t = 1 : size(xTrain,2)
            P = 1;
            C = WeakClassifier(xTrain(x,t),P,xTrain(x,:)');
            E = WeakClassifierError(C,D,yTrain);
            if E > 0.5
                P = P*-1;
                E = 1-E;
            end
            if(Emin > E)
                Emin = E;
                threshold = xTrain(x,t);
                bestC = C*P;
                bestX = x;
                bestP = P;
            end
        end
    end
    alpha = 0.5*log((1-Emin)/Emin);
    params(:,i) = [bestX; threshold; bestP; alpha]; 
    D = D.*exp(-alpha*yTrain.*bestC');
    D = D/sum(D);
end
%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.

res = zeros(length(xTest),1);
for w = 1 : size(params,2)
    feat = params(1,w);
    WC = WeakClassifier(params(2,w),params(3,w),xTest(feat,:)');
    tot = WC*params(4,w);
    res = res + tot;
    acc(w) = 1-sum(sign(res)' ~= yTest)/length(yTest);
end
class = sign(res);

res = zeros(length(xTrain),1);
for w = 1 : size(params,2)
    feat = params(1,w);
    WC = WeakClassifier(params(2,w),params(3,w),xTrain(feat,:)');
    tot = WC*params(4,w);
    res = res + tot;
    accTr(w) = 1-sum(sign(res)' ~= yTrain)/length(yTrain);
end
classTr = sign(res);


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
% errors = zeros(1,nbrWeakClassifiers);
% for i = 1 : nbrWeakClassifiers
%     class = evaluate(i,xTest,params); 
%     errors(i) = 1-(sum(class ~= yTest)/length(yTest));
% end

figure(4);
1-sum(class' ~= yTest)/length(yTest)
plot(1:nbrWeakClassifiers,acc);
hold on;
plot(1:nbrWeakClassifiers,accTr);
legend("Test","Train")
%% Plot some of the misclassified faces and non-faces from the test set
%  Use the subplot command to make nice figures with multiple images.

classDiff = class'-yTest;
classDiff = find(classDiff);
figure(5)

colormap gray;
subplot(231)
imagesc(testImages(:,:,210))
subplot(232)
imagesc(testImages(:,:,4568))
subplot(233)
imagesc(testImages(:,:,2902))
subplot(234)
imagesc(testImages(:,:,7448))
subplot(235)
imagesc(testImages(:,:,11199))
subplot(236)
imagesc(testImages(:,:,12151))

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
figure(6);
colormap gray;
subplot(231)
imagesc(haarFeatureMasks(:,:,1))
subplot(232)
imagesc(haarFeatureMasks(:,:,10))
subplot(233)
imagesc(haarFeatureMasks(:,:,20))
subplot(234)
imagesc(haarFeatureMasks(:,:,30))
subplot(235)
imagesc(haarFeatureMasks(:,:,40))
subplot(236)
imagesc(haarFeatureMasks(:,:,50))
