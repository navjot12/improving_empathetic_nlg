import numpy as np
import sys
import torch

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# Run with python3 main.py [dir-of-model] [num-of-epochs]

arg = sys.argv[1] if len(sys.argv) > 1 else None
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', truncation_side='right')

max_length = 512

# Original indices
# emotion_to_idx = {'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
#                   'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}

# Modified indices
emotion_to_idx = {'lonely': 0, 'guilty': 1, 'embarrassed': 1, 'ashamed': 1, 'jealous': 2, 'grateful': 3, 'content': 3, 'surprised': 4, 'caring': 5, 'disappointed': 6, 'disgusted': 6, 'angry': 7, 'annoyed': 7, 'furious': 7, 'prepared': 8, 'anticipating': 8,
                  'apprehensive': 8, 'hopeful': 9, 'confident': 9, 'sad': 10, 'devastated': 10, 'trusting': 11, 'faithful': 11, 'proud': 12, 'impressed': 12, 'excited': 13, 'joyful': 13, 'sentimental': 14, 'nostalgic': 14, 'afraid': 15, 'terrified': 15, 'anxious': 15}
num_labels = 16

print("Loading data")

# Training data

context = np.load(
    "../MoEL/empathetic-dialogue/sys_dialog_texts.train.npy", allow_pickle=True)
emotion = np.load("../MoEL/empathetic-dialogue/sys_emotion_texts.train.npy")

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
train_tokens = torch.stack(tokenized_context).squeeze()
train_masks = torch.stack(tokenized_masks).squeeze()
train_Y = torch.tensor([emotion_to_idx[e] for e in emotion])

# Create training dataloaders
train_data = TensorDataset(train_tokens, train_masks, train_Y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=10)

# Development data

context = np.load(
    "../MoEL/empathetic-dialogue/sys_dialog_texts.dev.npy", allow_pickle=True)
emotion = np.load("../MoEL/empathetic-dialogue/sys_emotion_texts.dev.npy")

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
dev_tokens = torch.stack(tokenized_context).squeeze()
dev_masks = torch.stack(tokenized_masks).squeeze()
dev_Y = torch.tensor([emotion_to_idx[e] for e in emotion])

# Create development dataloaders
dev_data = TensorDataset(dev_tokens, dev_masks, dev_Y)
dev_sampler = SequentialSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=10)

if arg:
    model = BertForSequenceClassification.from_pretrained(arg)
    print("loaded model {}".format(arg))
else:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=16,
        output_attentions=False,
        output_hidden_states=False,)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("training")
for epoch in range(0, epochs):
    print("Epoch: {} of {}".format(epoch, epochs))
    loss = 0
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader)):
        model.zero_grad()

        output = model(batch[0].to(device), token_type_ids=None,
                       attention_mask=batch[1].to(device),
                       labels=batch[2].to(device))

        loss += output[0].item()
        output[0].backward()
        optimizer.step()

    model.save_pretrained("./modified_epoch_{}".format(epoch))

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    for batch in tqdm(dev_dataloader):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None,
                            attention_mask=batch[1].to(device))
        logits = outputs[0].detach().cpu().numpy()
        labels = batch[2].to('cpu').numpy()

        eval_accuracy += np.sum(
            np.argmax(logits, axis=1).flatten() == labels.flatten()) / len(labels.flatten())

    print("Training loss: {} \tAccuracy: {}".format(
        loss, eval_accuracy/len(dev_dataloader)))
