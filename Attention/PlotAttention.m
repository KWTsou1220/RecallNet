clear;
clc;

%%

a1 = csvread('../Prediction1/Test1/attention_a1.csv');
a2 = csvread('../Prediction1/Test1/attention_a2.csv');
c1 = csvread('../Prediction1/Test1/attention_c1.csv');
c2 = csvread('../Prediction1/Test1/attention_c2.csv');


%%
time_step  = 20;
batch_size = 50;
time_batch = time_step*batch_size;
num_speaker = 77;
audpath = strcat('../../Enhancement/Test1/Target1/');

sequence = cell(6, 1);
start_idx = 1;
for idx=1:6
    sentencePath = strcat(audpath, strcat(int2str(idx), '/'));
    sentenceFile = dir(sentencePath);
    sentenceFile = sentenceFile(3:end);
    sequence{idx} = cell(25, 1);
    for jdx=1:25
        load(strcat(sentencePath, sentenceFile(jdx).name));
        sequence{idx}{jdx} = size(target1, 2);
        start_idx = start_idx+sequence{idx}{jdx};
    end
    remain = mod(start_idx, time_batch);
    if remain ~= 0
        start_idx = start_idx + time_batch-remain + 1;
    end
end


