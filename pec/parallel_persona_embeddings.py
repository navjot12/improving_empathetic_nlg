#!/usr/bin/env python
# coding: utf-8

# In[21]:


from embedding_creator_helper import *

from datasets import concatenate_datasets
from multiprocessing import Process, Queue, cpu_count
import sys
import time


# In[ ]:


def batch_dataset(dataset, num_proc=cpu_count() - 1):
    num_per_proc = round(len(dataset) / num_proc) + 1

    batched_dataset = []
    for speaker_idx in range(0, len(dataset), num_per_proc):
        batched_dataset.append(dataset.select(
                                   range(speaker_idx,
                                         min(speaker_idx+num_per_proc, len(dataset)))))
    
    return batched_dataset

def dataset_func(dataset, model_name, model, tokenizer, idx, q):
    print('Launched process', idx)
    q.put((idx, dataset.map(lambda example: {'persona_embedding': model_forward_pass(model_name,
                                                                                     model,
                                                                                     tokenizer,
                                                                                     example['personas'])})))

if __name__ == '__main__':
    split, model_name, path = sys.argv[1:4]
    print('> Split: %s, Path: %s' % (split, path))

    # Load persona sentences
    dataset = load_process_dataset(split)
    print('> Dataset after filtering and processing:', dataset)

    # Load BERT based model with pretrained weights to create persona embeddings
    tokenizer, model = get_tokenizer_and_model(model_name)
    print('\n> %s tokenizer and model loaded.' % model_name)

    # Put model in eval mode
    model.eval()

    # Split into batches
    batched_datasets = batch_dataset(dataset)
    print('\n> %s batches created.' % len(batched_datasets))

    # Create process queue
    queue = Queue()
    processes = [Process(target=dataset_func,
                         args=(batched_datasets[idx],
                               model_name,
                               model,
                               tokenizer,
                               idx,
                               queue)) for idx in range(len(batched_datasets))]

    for p in processes:
        print('Starting', p)
        p.start()

    time.sleep(10)
    for p in processes:
        print(p)

    time.sleep(10)
    for p in processes:
        print(p)

    for p in processes:
        print('Joining', p)
        p.join()

    q_get = []
    while not queue.empty():
        q_get.append(queue.get())
    
    print('q_get:', q_get)
    
    # Sort all datasets by idx and get rid of idx
    batched_datasets = [idx_tuple[1] for idx_tuple in sorted(q_get, key=lambda x: x[0])]
    
    dataset = concatenate_datasets(batched_datasets)
    
    print(dataset)

    if not path.endswith('/'):
        path += '/'

    path += split + '-' + model_name + '-' + str(len(dataset)) + '-personas-concat'

    dataset.save_to_disk(path)

    print('\n > Serialized dataset at %s' % path)


# In[ ]:


'''
import torch
from transformers import RobertaTokenizer, RobertaModel, LongformerTokenizer,  LongformerModel

def model_forward_pass(model_name, model, tokenizer, cat_persona_sentences):
    with torch.no_grad():
        outputs = None
        
        if model_name == 'Roberta':
            inputs = tokenizer(cat_persona_sentences, truncation=True, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
        
        elif model_name == 'Longformer':
            # batch of size 1
            input_ids = torch.tensor(tokenizer.encode(cat_persona_sentences)).unsqueeze(0)

            # global attention mask to attend locally within a persona sentence
            # and globally among special tokens.
            global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
            for ix in range(len(input_ids[0])):
                if input_ids[0][ix] in [0, 2]:
                    global_attention_mask[0][ix] = 1.0

            outputs = model(input_ids,
                            global_attention_mask=global_attention_mask,
                            output_hidden_states=True)

        else:
            raise Exception('model_name argument not in [Roberta, Longformer]')
    
    # Get last four layers.
    last_four_layers = [outputs.hidden_states[i] for i in (-1, -2, -3, -4)]

    # Cast layers to a tuple and concatenate over the last dimension
    cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)

    # Take the mean of the concatenated vector over the token dimension
    return torch.mean(cat_hidden_states, dim=1).squeeze()
    

def dataset_func(dataset, idx, q):
    q.put((idx, dataset.map(lambda example: {'persona_embedding': model_forward_pass(model_name,
                                                                                     model,
                                                                                     tokenizer,
                                                                                     example['personas'])})))
'''

