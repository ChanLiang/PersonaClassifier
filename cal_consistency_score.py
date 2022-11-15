import json
import pickle
from sys import argv

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample
import numpy as np


def get_dataloader(input_examples, tokenizer, device, batch_size=256):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def read_data(ori_f, pred_f):
    with open(ori_f, 'r') as f:
        data = json.load(f)
        
    preds = []
    with open(pred_f, 'r') as f:
        for line in f:
            preds.append(line.strip())
    assert len(data) == len(preds)
    # print ('len(data) = ', len(data)) # 7801

    examples = []
    cnt = 0
    for (persona_list, history, response, persona_label), hyp in zip(data, preds):
        for persona in persona_list:
            # InputExample: A single training/test example for simple sequence classification.
            examples.append(InputExample(str(cnt), persona, hyp, '0'))
            cnt += 1
    # print ('len(examples) = ', len(examples)) # 34971 = hyp x 5
    return examples, len(preds)


# The original train file
input_file = argv[1]
# The output file that saves the NLI logits given the train samples
output_file = './result_files/' + argv[2].strip()
# load tokenizer and model (single gpu is enough)
# output_file = './result_files/baseline_normal_ord.bin'

print ('load model...')
NLI_MODEL_PATH = './persona_nli'
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# make prediction on persona data
input_examples, num = read_data(
'/misc/kfdata01/kf_grp/lchen/D3/data_manipulation/data_distillation/predictions/test/output', 
# f"/misc/kfdata01/kf_grp/lchen/ACL23/Nov/decoding_results/debug_baseline_GP2-pretrain-step-7000_normal_ord_top10_top0.9_T0.9_2022-11-08_pred_response"
input_file
)

train_dataloader = get_dataloader(input_examples, tokenizer, device, batch_size=512)
all_logits = None
with torch.no_grad():
    # for batch in tqdm(train_dataloader):
    for batch in train_dataloader:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        outputs = model(**inputs)
        if all_logits is None:
            all_logits = outputs[0].cpu().detach()
        else: # [n, 3], 每个batch直接cat到第一个维度上 
            all_logits = torch.cat((all_logits, outputs[0].cpu().detach()), dim=0)

# with open(output_file, 'rb') as f:
#     all_logits = torch.tensor(pickle.load(f))

results = torch.argmax(all_logits, dim=1) # [n]
# print (all_logits.shape, results.shape) # torch.Size([34971, 3]) torch.Size([34971])

# calculate consistence score as https://aclanthology.org/P19-1542.pdf
# label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
cnt = 0
for i, res in enumerate(results):
    cnt = cnt + (res - 1)
print ('consistence score is ', cnt/7512)
with open(output_file + '.txt', 'w') as f:
    f.write(str(cnt/7512))

all_logits = all_logits.numpy()
with open(output_file + '.bin', 'wb') as f:
    pickle.dump(all_logits, f)


