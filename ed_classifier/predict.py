import numpy as np
import sys
import torch

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', truncation_side='right')

max_length = 512
emotion_to_idx = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
                  'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
idx_to_emotion = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud', 4: 'angry', 5: 'sad', 6: 'grateful',
                  7: 'lonely', 8: 'impressed', 9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                  13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful', 17: 'prepared', 18: 'guilty',
                  19: 'furious', 20: 'nostalgic', 21: 'jealous', 22: 'anticipating', 23: 'embarrassed',
                  24: 'content', 25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                  29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

for dataset in ["dev", "test", "train"]:
    input_data = np.load("sys_dialog_texts.{}.npy".format(
        dataset), allow_pickle=True)
    tokenized_context = []
    tokenized_masks = []
    for c in tqdm(input_data):
        tokens = tokenizer(" ".join(c),
                           padding='max_length', max_length=max_length,
                           truncation=True, return_tensors='pt')
        tokenized_context.append(tokens['input_ids'])
        tokenized_masks.append(tokens['attention_mask'])

    outputs = []

    model = BertForSequenceClassification.from_pretrained("model")
    for token, mask in tqdm(zip(tokenized_context, tokenized_masks)):
        output = model(token, token_type_ids=None, attention_mask=mask)

        logits = output[0].detach().numpy()
        outputs.append(idx_to_emotion[np.argmax(logits, axis=1).item()])

    np.save("sys_emotioncls_texts.{}.npy".format(dataset), np.stack(outputs))
