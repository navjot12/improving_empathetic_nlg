#!/usr/bin/env python
# coding: utf-8

# In[1]:


from embedding_creator_helper import *

from multiprocessing import cpu_count, set_start_method
import sys
import time
# In[5]:


if __name__ == '__main__':
    split, start_idx, end_idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print('> split: %s, start_idx: %s, end_idx: %s' % (split, start_idx, end_idx))

    # Load persona sentences
    dataset = load_process_dataset(split)
    print('> Dataset after filtering and processing:', dataset)

    # Slice dataset
    end_idx = min(end_idx, len(dataset))
    dataset = dataset.select(range(start_idx, end_idx))
    print('> Sliced dataset from %s to %s' % (start_idx, end_idx))

    # Map dataset
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    dataset = dataset.map(lambda example: {'persona_embedding':
                                           model_forward_pass(model, tokenizer, example['personas'])})

    print(dataset)
    
    path = split + '-' + str(start_idx) + '-' + str(end_idx) + '-personas'
    
    dataset.save_to_disk(path)
    
    print('\n > Serialized dataset at %s' % path)


# In[ ]:




