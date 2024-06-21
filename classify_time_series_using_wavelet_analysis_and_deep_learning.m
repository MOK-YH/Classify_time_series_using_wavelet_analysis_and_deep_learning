% 데이터 다운로드, 압축 풀기
unzip(fullfile(tempdir,"physionet_ECG_data-main.zip"),tempdir)

unzip(fullfile(tempdir,"physionet_ECG_data-main","ECGData.zip"), ...
    fullfile(tempdir,"physionet_ECG_data-main"))
load(fullfile(tempdir,"physionet_ECG_data-main","ECGData.mat"))

parentDir = tempdir;
dataDir = "data";
helperCreateECGDirectories(ECGData,parentDir,dataDir)

% helperfunction으로 helpPlotReps 수행
helperPlotReps(ECGData)

% 시간-주파수 표현 만들기
% 스케일로그램(시간-주파수 표현) 세부 항목
Fs = 128;
fb = cwtfilterbank(SignalLength=1000, ...
    SamplingFrequency=Fs, ...
    VoicesPerOctave=12);
sig = ECGData.Data(1,1:1000);
[cfs,frq] = wt(fb,sig);
t = (0:999)/Fs;
figure
pcolor(t,frq,abs(cfs))
set(gca,"yscale","log")
shading interp
axis tight
title("Scalogram")
xlabel("Time (s)")
ylabel("Frequency (Hz)")

% 스케일로그램 plot
helperCreateRGBfromTF(ECGData,parentDir,dataDir)

% 훈련 데이터와 검증 데이터로 나누기 
allImages = imageDatastore(fullfile(parentDir,dataDir), ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

% Train 80, Validation 20으로 두 그룹 나누기
[imgsTrain,imgsValidation] = splitEachLabel(allImages,0.8,"randomized");
disp("Number of training images: "+num2str(numel(imgsTrain.Files)))

disp("Number of validation images: "+num2str(numel(imgsValidation.Files)))

% GoogLeNet
% 지원 패키지 불러오기
net = imagePretrainedNetwork("googlenet");

% 신경망에서 그래프 추출
numberOfLayers = numel(net.Layers);
figure("Units","normalized","Position",[0.1 0.1 0.8 0.8])
plot(net)
title("GoogLeNet Layer Graph: "+num2str(numberOfLayers)+" Layers")

% 첫 번째  Layer를 검사
net.Layers(1)

% 파라미터 수정하기
net.Layers(end-3:end)

% 과적합 방지 위한 Drop out layer
newDropoutLayer = dropoutLayer(0.6,"Name","new_Dropout");
net = replaceLayer(net,"pool5-drop_7x7_s1",newDropoutLayer);

% 새로운 완전 연결 계층으로 교체 
numClasses = numel(categories(imgsTrain.Labels));
newConnectedLayer = fullyConnectedLayer(numClasses,"Name","new_fc", ...
    "WeightLearnRateFactor",5,"BiasLearnRateFactor",5);
net = replaceLayer(net,"loss3-classifier",newConnectedLayer);

% 교체 확인
net.Layers(end-3:end)

% 훈련 옵션 지정
options = trainingOptions("sgdm", ...
    MiniBatchSize=15, ...
    MaxEpochs=20, ...
    InitialLearnRate=1e-4, ...
    ValidationData=imgsValidation, ...
    ValidationFrequency=10, ...
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics="accuracy");

% 신경망 훈련
trainedGN = trainnet(imgsTrain,net,"crossentropy",options);

% GoogLeNet 정확도 평가
classNames = categories(imgsTrain.Labels);
scores = minibatchpredict(trainedGN,imgsValidation);
YPred = scores2label(scores,classNames);
accuracy = mean(YPred==imgsValidation.Labels);
disp("GoogLeNet Accuracy: "+num2str(100*accuracy)+"%")

% 활성화 결과 생성
trainedGN.Layers(1:5)

% 필터 가중치 시각화
wghts = trainedGN.Layers(2).Weights;
wghts = rescale(wghts);
wghts = imresize(wghts,8);
figure
I = imtile(wghts,GridSize=[8 8]);
imshow(I)
title("First Convolutional Layer Weights")

% ARR 활성화 영역 확인
convLayer = "conv1-7x7_s2";

imgClass = "ARR";
imgName = "ARR_10.jpg";
imarr = imread(fullfile(parentDir,dataDir,imgClass,imgName));

trainingFeaturesARR = predict(trainedGN,single(imarr),Outputs=convLayer);
sz = size(trainingFeaturesARR);
trainingFeaturesARR = reshape(trainingFeaturesARR,[sz(1) sz(2) 1 sz(3)]);
figure
I = imtile(rescale(trainingFeaturesARR),GridSize=[8 8]);
imshow(I)
title(imgClass+" Activations")

% 원본 영상과 비교
imgSize = size(imarr);
imgSize = imgSize(1:2);
[~,maxValueIndex] = max(max(max(trainingFeaturesARR)));
arrMax = trainingFeaturesARR(:,:,:,maxValueIndex);
arrMax = rescale(arrMax);
arrMax = imresize(arrMax,imgSize);
figure
I = imtile({imarr,arrMax});
imshow(I)
title("Strongest "+imgClass+" Channel: "+num2str(maxValueIndex))

% SqueezeNet
% 불러오기
netsqz = imagePretrainedNetwork("squeezenet");

% 크기 확인
disp("Number of Layers: "+num2str(numel(netsqz.Layers)))

netsqz.Layers(1)

% SqueezeNet 신경망 파라미터 수정
% end 5개 계층검사
netsqz.Layers(end-4:end)

% Drop out 계층 확률 0.6으로 교체
tmpLayer = netsqz.Layers(end-5);
newDropoutLayer = dropoutLayer(0.6,"Name","new_dropout");
netsqz = replaceLayer(netsqz,tmpLayer.Name,newDropoutLayer);

% 새로운 계층의 학습률 인자 높임
numClasses = numel(categories(imgsTrain.Labels));
tmpLayer = netsqz.Layers(end-4);
newLearnableLayer = convolution2dLayer(1,numClasses, ...
        "Name","new_conv", ...
        "WeightLearnRateFactor",10, ...
        "BiasLearnRateFactor",10);
netsqz = replaceLayer(netsqz,tmpLayer.Name,newLearnableLayer);

% 변경 확인
netsqz.Layers(end-4:end)

% SqueezeNet 위한 RGB 데이터 준비
augimgsTrain = augmentedImageDatastore([227 227],imgsTrain);
augimgsValidation = augmentedImageDatastore([227 227],imgsValidation);

% 훈련 옵션 설정, 훈련
ilr = 3e-4;
miniBatchSize = 10;
maxEpochs = 15;
valFreq = floor(numel(augimgsTrain.Files)/miniBatchSize);
opts = trainingOptions("sgdm", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=maxEpochs, ...
    InitialLearnRate=ilr, ...
    ValidationData=augimgsValidation, ...
    ValidationFrequency=valFreq, ...
    Verbose=1, ...
    Plots="training-progress", ...
    Metrics="accuracy");

trainedSN = trainnet(augimgsTrain,netsqz,"crossentropy",opts);

% 정확도 평가
scores = minibatchpredict(trainedSN,augimgsValidation);
YPred = scores2label(scores,classNames);
accuracy = mean(YPred==imgsValidation.Labels);
disp("SqueezeNet Accuracy: "+num2str(100*accuracy)+"%")