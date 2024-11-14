---
geometry: margin=30mm
title: "Assignment Week 9"
author: "Student Name: Patrick Farmer       Student Number: 20501828"
date: "Date: 23-10-2024"
---

![](https://www.tcd.ie/media/tcd/site-assets/images/tcd-logo.png)

\clearpage

## I(a)

The zip file was download and extracted. I then ran gpt.py but this took an unreasonable length of time to run as it was running on the CPU. Since I have an AMD GPU I had to instal ROCm to run the code on the GPU. I then ran the code again and it ran much faster.\
Now that the environment was setup I changed the code to take the file path as a command line argument and ran the code pointing at the input_childSpeech_trainingSet.txt data set. A print message was also added to print the vocabulary size of the
dataset wich was 40. This was run for the two other datasets as well. The childSpeechTest dataset also had a vocabulary size of 40 while the shakespeare dataset had a vocabulary size of 65.

## I(b)

The parameters of the model that I changed were:
- Block size: 256 -> 64
- Max Iterations: 5000 -> 3000
- Evaluation Interval: 500 -> 300
- Learning Rate: 3e-4 -> 2e-4
- Embedding Size: 384 -> 128
- Attention Heads: 6 -> 4
- Transformer Layers: 6 -> 4
- Dropout Rate: 0.2 -> 0.3

Block size is the maximum context length. Which is the amount of previous tokens that the model will consider when predicting the next token.
* The block size was reduced 4 times to 64. This was done to reduce the parameter count of the model, I chose to reduce this significantly as looking at the data set I saw that each line was quite short and independant of the next and previous lines so a large context size is unnecessary.\
Max Iterations is the maximum number of iterations that the model will train for.
* The max iterations was reduced to 3000 as it was found that the simpler model did not need as many iterations to train and reducing this would reduce the training time and level of overfitting.\
Evaluation Interval is the number of iterations between evaluations against the validation set.
* The evaluation interval was reduced to 300 to match the decrease in the max iterations and have the same count of evaluations.\
Learning Rate is the rate at which the optimiser will update the weights of the model.\
* The learning rate was reduced to 2e-4 to stabilise training as the model was simplified and a smaller learning rate makes more sense.\
Embedding Size is the size of the embedding layer, which is the layer that converts the input tokens into a dense vector.
* The embedding size was reduced to 128 to reduce the parameter count of the model. This was done as the vocabulary size of the data set was quite small and the embedding size did not need to be as large.\
Attention Heads is the number of heads in the multi-head attention layer. This is the number of different attention mechanisms that the model will use.
* The attention heads was reduced to 4 to reduce the parameter count of the model. This was done as the model was simplified and the number of heads did not need to be as large.\
Transformer Layers is the number of transformer layers in the model.
* The transformer layers was reduced to 4 to reduce the parameter count of the model. This was done as the model was simplified and if the model was too deep it would be more likely to overfit.\
Dropout Rate is the rate of the dropout layer. This is the percentage (in decimal form) of the units that will be dropped during training.
* The dropout rate was increased to 0.3 to reduce the level of overfitting. This was done as the model was simplified and the dropout rate was increased to reduce the level of overfitting.

## I(c)

## I(d)

## I(e)

## II(a)

## II(b)

## Appendices

### I(a)

### I(b)

### I(c)

### I(d)

### I(e)

### II(a)

### II(b)
