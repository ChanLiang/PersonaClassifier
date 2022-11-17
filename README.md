# PersonaClassifier

This repository contains the source code and trained model for RoBERTa finetuned on DNLI dataset. 

The repository is developed based on [D3](https://github.com/caoyu-noob/D3).



## Setup & Installation (TL;DR)

#### Environmental requirements
Note: The script below may not be sufficient and missing packages need to be configured manually.
1. python == 3.7.0
2. torch==1.5.0
3. transformers==3.1.0
4. spacy==2.2.4
5. fairseq==0.9.0 (I downloaded the source code into the root directory)
6. sentencepiece==0.1.94


## Pipeline details

#### 1. Prepare models

Just download the finetuned [NLI model](https://drive.google.com/file/d/1QnT8V2Yj4Zl2yW2rnQIi2p56I_wbN3Ee/view?usp=sharing) and put it to ./persona_nli .

Note: This model is a [RoBERTa large MNLI model](https://huggingface.co/roberta-large-mnli) finetuned on the [DialogueNLI dataset](https://wellecks.github.io/dialogue_nli/).

#### 2. Data Preparation
##### Evaluating persona consistency
See example data in ./data/consistency_calculation

##### Predicting persona label
See example data in ./data/persona_labeling


#### 3. Evaluating persona consistency

```bash
bash consistency_pipeline.sh
or
python cal_consistency_score.py 
```


#### 4. Predicting persona label
```bash
bash persona_label_pipeline.sh
or
bash cal_persona_label.py --params...
bash get_persona_labeled_dataset.py --params...
```

#### 5. Counting the persona label
```bash
python count_label.py output_file
```

## Some counting results
Interestingly, I found that the model is quite sure that 50% responses don't use any persona as its predicted class distribution is 'sharp'---- the probability of the predicted class is more than an order of magnitude larger than the other two classes.

```python
(D3) bash-4.2$ python count_label.py predictions/test/output-wo-th
[3979, 925, 849, 746, 698, 315]
[0.53, 0.12, 0.11, 0.099, 0.092, 0.042]

(D3) bash-4.2$ python count_label.py predictions/train/output-wo-th
[33159, 8914, 7741, 6983, 6224, 2698]
[0.50, 0.134, 0.118, 0.106, 0.095, 0.041]

(D3) bash-4.2$ python count_label.py predictions/valid/output-wo-th
[3818, 1129, 958, 821, 752, 323]
[0.49, 0.14, 0.12, 0.11, 0.096, 0.041]
```




