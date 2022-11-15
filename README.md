# PersonaClassifier

This repository contains the source code and trained model for RoBERTa finetuned on DNLI dataset. 

<!--See more details on our [project page](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/)-->

The repository is developed based on [D3](https://github.com/caoyu-noob/D3).



## Setup & Installation (TL;DR)



#### Environment
1. python == 3.7.0
2. torch==1.5.0
3. transformers==3.1.0
4. spacy==2.2.4
5. fairseq==0.9.0 (I downloaded the source code into the root directory)
6. sentencepiece==0.1.94


## Pipeline details

#### Prepare models
At first, we have to get all trained models we need for data manipulation in experiments.
You need go to `./data_manipulation/prepare_model`.

##### NLI model for evaluating persona consistency
You need to download [DialogueNLI dataset](https://wellecks.github.io/dialogue_nli/)
and put it under this directory. Also, download large size [RoBERTa MNLI model](https://huggingface.co/roberta-large-mnli)
and put it under this directory, renaming the document as `roberta_mnli/`.

Then you can train the NLI model using this dataset using script `train_nli_model.py`.

After obtain the trained best model, you need to renamed the file `best_model.bin` as `pytorch_model.bin` for the following 
use. Define the path that saves the trained NLI model for persona consistency as `PERSONA_NLI`.

We also provide our trained [NLI model](https://drive.google.com/file/d/1QnT8V2Yj4Zl2yW2rnQIi2p56I_wbN3Ee/view?usp=sharing) 
for downloading.


#### Calculate consistency score

```bash
bash consistency.sh # pipeline
or
python cal_consistency_score.py 
```


#### Predict persona label
```bash
bash cal_persona_label.py
```


### Data Preparation
See example data in ./data

