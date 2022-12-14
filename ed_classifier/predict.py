import numpy as np
import sys
import torch

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', truncation_side='right')

max_length = 512

# Old dictionaries
# emotion_to_idx = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
#                   'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
# idx_to_emotion = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud', 4: 'angry', 5: 'sad', 6: 'grateful',
#                   7: 'lonely', 8: 'impressed', 9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
#                   13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful', 17: 'prepared', 18: 'guilty',
#                   19: 'furious', 20: 'nostalgic', 21: 'jealous', 22: 'anticipating', 23: 'embarrassed',
#                   24: 'content', 25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
#                   29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

# Modified dictionaries
emotion_to_idx = {'lonely': 0, 'guilty': 1, 'embarrassed': 1, 'ashamed': 1, 'jealous': 2, 'grateful': 3, 'content': 3, 'surprised': 4, 'caring': 5, 'disappointed': 6, 'disgusted': 6, 'angry': 7, 'annoyed': 7, 'furious': 7, 'prepared': 8, 'anticipating': 8,
                  'apprehensive': 8, 'hopeful': 9, 'confident': 9, 'sad': 10, 'devastated': 10, 'trusting': 11, 'faithful': 11, 'proud': 12, 'impressed': 12, 'excited': 13, 'joyful': 13, 'sentimental': 14, 'nostalgic': 14, 'afraid': 15, 'terrified': 15, 'anxious': 15}
idx_to_emotion = {0: 'lonely', 1: 'guilty/embarrassed/ashamed', 2: 'jealous', 3: 'grateful/content', 4: 'surprised', 5: 'caring', 6: 'disappointed/disgusted', 7: 'angry/annoyed/furious',
                  8: 'prepared/anticipating/apprehensive', 9: 'hopeful/confident', 10: 'sad/devastated', 11: 'trusting/faithful', 12: 'proud/impressed', 13: 'excited/joyful', 14: 'sentimental/nostalgic', 15: 'afraid/terrified/anxious'}
mod_emotion_to_idx = {'lonely': 0, 'guilty/embarrassed/ashamed': 1, 'jealous': 2, 'grateful/content': 3, 'surprised': 4, 'caring': 5, 'disappointed/disgusted': 6, 'angry/annoyed/furious': 7,
                      'prepared/anticipating/apprehensive': 8, 'hopeful/confident': 9, 'sad/devastated': 10, 'trusting/faithful': 11, 'proud/impressed': 12, 'excited/joyful': 13, 'sentimental/nostalgic': 14, 'afraid/terrified/anxious': 15}

model = BertForSequenceClassification.from_pretrained("model")

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    tokens = torch.stack(tokenized_context).squeeze()
    masks = torch.stack(tokenized_masks).squeeze()

    data = TensorDataset(tokens, masks)
    dataloader = DataLoader(data, batch_size=10)

    outputs = []

    for batch in tqdm(dataloader):
        output = model(batch[0].to(device),
                       token_type_ids=None, attention_mask=batch[1].to(device))

        logits = output[0].detach().cpu().numpy()
        outputs.append(np.vectorize(idx_to_emotion.get)
                       (np.argmax(logits, axis=1)))

    np.save("sys_emotioncls_texts.{}.npy".format(
        dataset), np.concatenate(outputs))
