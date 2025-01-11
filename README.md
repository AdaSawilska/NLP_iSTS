# NLP_iSTS

## Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Datasets](#datasets)
-  [Training model](#training-model)
-  [Testing model with SemEval scripts](#testing-model-with-semeval-scripts)



## Introduction
This project focuses on leveraging the [RoBERTa](https://arxiv.org/abs/1907.11692) model to assess interpretable semantic textual similarity (iSTS) between pairs of sentences. The objective is to compute how similar two given sentences, s1 and s2, are by assigning a similarity score. While this score is beneficial for various applications, it does not reveal which specific parts of the sentences share equivalent or closely related meanings.

Interpretable STS aims to bridge this gap by providing explanations for why sentences are considered similar or different. This is achieved by aligning corresponding chunks between the two sentences and assigning each alignment a similarity score and a relational label, offering a more detailed understanding of the connection between the sentences.

In addition to utilizing gold standard chunks for alignment, this project also incorporates sentence-level context to enhance the interpretability and accuracy of the similarity assessment. This approach draws inspiration from the [SemEval](https://alt.qcri.org/semeval2020/) competition, which emphasizes comprehensive semantic analysis.

## Requirements
```bash
pip install -r requirements.txt
```
## Datasets
All needed datasets are on [site](https://alt.qcri.org/semeval2016/task2/).
Then you have to prepare training and validation sets.

## Training model 
To train model with default parameters you need to  change the directiories for your training and validation sets. Then run following command 
```bash
python main.py
```

## Testing model with SemEval scripts
There are 3 perl scripts created by competition organizers:
- evalF1_no_penalty.pl
- evalF1_penalty.pl
- wellformed.pl

Obtained model was tested separately on each testing dataset and the results are shown below:

Results Using evalF1_no_penalty.pl
|                   |     „News Headlines”    |     „Images Captions”    |     „Answers-Students”    |
|-------------------|-------------------------|--------------------------|---------------------------|
|     F1   Ali      |     0.9929              |     0.9843               |     0.9812                |
|     F1   Type     |     0.7768              |     0.7731               |     0.8414                |
|     F1   Score    |     0.9387              |     0.9331               |     0.9341                |
|     F1 Typ+Sco    |     0.7574              |     0.7531               |     0.8211                |

Results Using evalF1_penalty.pl
|                   |     „News Headlines”    |     „Images Captions”    |     „Answers-Students”    |
|-------------------|-------------------------|--------------------------|---------------------------|
|     F1   Ali      |     0.9929              |     0.9843               |     0.9812                |
|     F1   Type     |     0.7768              |     0.7731               |     0.8414                |
|     F1   Score    |     0.9387              |     0.9331               |     0.9341                |
|     F1 Typ+Sco    |     0.8267              |     0.8008               |     0.8506                |




