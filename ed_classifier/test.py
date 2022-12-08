import numpy as np
import sys
import torch

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# Run with python3 test.py [dir-of-model]

arg = sys.argv[1]

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', truncation_side='right')

max_length = 512
emotion_to_idx = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
                  'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

# Training data

context = np.load(
    "../MoEL/empathetic-dialogue/sys_dialog_texts.test.npy", allow_pickle=True)
emotion = np.load("../MoEL/empathetic-dialogue/sys_emotion_texts.test.npy")

# Use tokenizer to generate tokens from empathetic context
tokenized_context = []
tokenized_masks = []
for c in tqdm(context):
    tokens = tokenizer(" ".join(c),
                       padding='max_length', max_length=max_length,
                       truncation=True, return_tensors='pt')
    tokenized_context.append(tokens['input_ids'])
    tokenized_masks.append(tokens['attention_mask'])

# Create token torch tensors
test_tokens = torch.stack(tokenized_context).squeeze()
test_masks = torch.stack(tokenized_masks).squeeze()
test_Y = torch.tensor([emotion_to_idx[e] for e in emotion])

# Create testing dataloaders
test_data = TensorDataset(test_tokens, test_masks, test_Y)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=10)

model = BertForSequenceClassification.from_pretrained(arg)
print("loaded model {}".format(arg))

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.eval()

eval_loss, eval_accuracy = 0, 0
for batch in tqdm(test_dataloader):
    with torch.no_grad():
        outputs = model(batch[0].to(device), token_type_ids=None,
                        attention_mask=batch[1].to(device))
        logits = outputs[0].detach().cpu().numpy()
        labels = batch[2].to('cpu').numpy()

        eval_accuracy += np.sum(
            np.argmax(logits, axis=1).flatten() == labels.flatten()) / len(labels.flatten())

print("Accuracy: {}".format(eval_accuracy/len(test_dataloader)))
