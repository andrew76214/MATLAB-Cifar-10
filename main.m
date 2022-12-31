clear
clc

[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load('cifar10Data');
size(trainingImages);
numImageCategories = 10;
categories(trainingLabels)

net = alexnet;
layersTransfer = net.Layers(1:end-3);

finalLayers = [ fullyConnectedLayer(numImageCategories)
                softmaxLayer
                classificationLayer ];

 layers = [ layersTransfer
            finalLayers
            ];

 analyzeNetwork(layers)

 inputSize = net.Layers(1).InputSize;

 % augimdsTrain = augmentedImageDatastore(inputSize(1:2), trainingImages);
 % augimdsTest = augmentedImageDatastore(inputSize(1:2), testImages);

 imageSize = [224 224 3];
 pixelRange = [-4 4];

 imageAugmenter = imageDataAugmenter(RandXReflection=true, RandXTranslation=pixelRange, RandYTranslation=pixelRange);

 augimdsTrain = augmentedImageDatastore(imageSize, trainingImages, trainingLabels, ... % DataAugmentation=imageAugmenter, ...
                                        ColorPreprocessing="gray2rgb");
 augimdsTest = augmentedImageDatastore(imageSize, testImages, testLabels, ...
                                        ColorPreprocessing="gray2rgb");

 opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'Shuffle','every-epoch',...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {augimdsTest, testLabels}, ...
    'ExecutionEnvironment','gpu',...
    'Verbose', true,...
    'Plots','training-progress');

cifar10Net = trainNetwork(augimdsTrain, layers, opts);

[YTest, probs] = classify(cifar10Net, augimdsTest);
accuracy = sum(YTest == testLabels)/numel(testLabels);

disp("accuracy: " + accuracy*100 + "%")

figure(Units="normalized",Position=[0.2 0.2 0.4 0.4]);
cm = confusionchart(testLabels, YTest);
cm.Title = "Confusion Matrix for Validation Data";
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";

figure
idx = randperm(size(testImages,4),9);
for i = 1:numel(idx)
    subplot(3,3,i)
    imshow(testImages(:,:,:,idx(i)));
    prob = num2str(100*max(probs(idx(i),:)),3);
    predClass = char(YTest(idx(i)));
    title(predClass + ", " + prob + "%")
end