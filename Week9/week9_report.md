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

The parameter count of this model was 0.80772 M parameters\
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

Two other sets of hyperparameters that I tried were as follows:

### Hyperparameters set 2
- Block size: 128
- Max Iterations: 3000
- Evaluation Interval: 300
- Learning Rate: 2e-4
- Embedding Size: 128
- Attention Heads: 3
- Transformer Layers: 3
- Dropout Rate: 0.1

### Hyperparameters set 3
- Block size: 256
- Max Iterations: 3000
- Evaluation Interval: 300
- Learning Rate: 2e-4
- Embedding Size: 172
- Attention Heads: 2
- Transformer Layers: 2
- Dropout Rate: 0.1

The the three models were trained on the childSpeechTraining dataset and the parameter count and loss function for each model was as follows:
### Model 1
```log
0.80772 M parameters
Vocabulary size: 40
step 0: train loss 3.6883, val loss 3.6886
step 300: train loss 0.4891, val loss 0.4938
step 600: train loss 0.3920, val loss 0.3949
step 900: train loss 0.3792, val loss 0.3822
step 1200: train loss 0.3767, val loss 0.3788
step 1500: train loss 0.3744, val loss 0.3785
step 1800: train loss 0.3707, val loss 0.3745
step 2100: train loss 0.3722, val loss 0.3760
step 2400: train loss 0.3711, val loss 0.3748
step 2700: train loss 0.3698, val loss 0.3716
step 2999: train loss 0.3697, val loss 0.3719
```

### Model 2
```log
0.615592 M parameters
Vocabulary size: 40
step 0: train loss 3.7671, val loss 3.7663
step 300: train loss 0.4805, val loss 0.4838
step 600: train loss 0.3643, val loss 0.3675
step 900: train loss 0.3560, val loss 0.3586
step 1200: train loss 0.3510, val loss 0.3547
step 1500: train loss 0.3454, val loss 0.3500
step 1800: train loss 0.3455, val loss 0.3493
step 2100: train loss 0.3433, val loss 0.3488
step 2400: train loss 0.3437, val loss 0.3472
step 2700: train loss 0.3440, val loss 0.3494
step 2999: train loss 0.3425, val loss 0.3468
```

### Model 3
```log
0.769912 M parameters
Vocabulary size: 40
step 0: train loss 3.7828, val loss 3.7833
step 500: train loss 0.4572, val loss 0.4624
step 1000: train loss 0.3463, val loss 0.3504
step 1500: train loss 0.3357, val loss 0.3400
step 2000: train loss 0.3341, val loss 0.3395
step 2500: train loss 0.3319, val loss 0.3378
step 3000: train loss 0.3299, val loss 0.3370
step 3500: train loss 0.3290, val loss 0.3350
step 4000: train loss 0.3281, val loss 0.3362
step 4500: train loss 0.3260, val loss 0.3373
step 4999: train loss 0.3247, val loss 0.3362
```

The three models had very few differences between them. The only observable difference in the output of the models is that model 1 has more typos than the other two. The sentence structure, grammar and content is very similar between the output of the three models though. The loss function of the three models is also very similar with model 3 having the best and model 1 having the worst. This backs up the observation that model 1 has more typos than the other two models. Model 2 has the lowest parameter counts and almost as good a loss function as model 3. For a scenario where the model needs to be as small as possible model 2 would be the best choice.\
The three models do not overfit significantly, the loss of the validation set is very close to the loss of the training set for all three models. The models were tested with a reduced level of regularisation to see if the models would perform better but there was no improvement in the loss function of the models and the models were reverted to the above hyperparameters.\

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
