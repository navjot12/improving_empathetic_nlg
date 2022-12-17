import numpy as np
import sys
import torch

from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# 32 Indicies
emotion_to_idx = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3,
                  'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
                  'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11,
                  'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
                  'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19,
                  'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
                  'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27,
                  'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

# 16 Indicies
emotion16_to_idx = {'lonely': 0, 'guilty': 1, 'embarrassed': 1, 'ashamed': 1,
                    'jealous': 2, 'grateful': 3, 'content': 3, 'surprised': 4,
                    'caring': 5, 'disappointed': 6, 'disgusted': 6, 'angry': 7,
                    'annoyed': 7, 'furious': 7, 'prepared': 8, 'anticipating': 8,
                    'apprehensive': 8, 'hopeful': 9, 'confident': 9, 'sad': 10,
                    'devastated': 10, 'trusting': 11, 'faithful': 11, 'proud': 12,
                    'impressed': 12, 'excited': 13, 'joyful': 13, 'sentimental': 14,
                    'nostalgic': 14, 'afraid': 15, 'terrified': 15, 'anxious': 15}

context = np.load(
    "../MoEL/empathetic-dialogue/sys_dialog_texts.test.npy", allow_pickle=True)
emotion = np.load("../MoEL/empathetic-dialogue/sys_emotion_texts.test.npy")

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', truncation_side='right')

model = BertForSequenceClassification.from_pretrained("model")

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenized_context = []
tokenized_masks = []
for c in tqdm(context):
    tokens = tokenizer(" ".join(c),
                       padding='max_length', max_length=512,
                       truncation=True, return_tensors='pt')
    tokenized_context.append(tokens['input_ids'])
    tokenized_masks.append(tokens['attention_mask'])

tokens = torch.stack(tokenized_context).squeeze()
masks = torch.stack(tokenized_masks).squeeze()

data = TensorDataset(tokens, masks)
dataloader = DataLoader(data, batch_size=10)

outputs = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        output = model(batch[0].to(device),
                       token_type_ids=None, attention_mask=batch[1].to(device))

        logits = output[0].detach().cpu().numpy()
        outputs.append(np.argmax(logits, axis=1))

y_true = np.vectorize(emotion_to_idx.get)(emotion)[0:11]
y_pred = outputs[0]
print("Accuracy: {}".format(np.sum(y_true == y_pred)/len(y_true)))
print("Precision, Recall, F1-Score: {}".format(
    precision_recall_fscore_support(y_true, y_pred, average='micro')))
