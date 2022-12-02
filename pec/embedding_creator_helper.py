from datasets import load_dataset
from multiprocessing import cpu_count
import torch
from transformers import LongformerTokenizer,  LongformerModel

def load_process_dataset(split):
    # Load data from HuggingFace
    dataset = load_dataset("pec", "all")[split]
    print('--- Initial PEC %s dataset:' % split, dataset)

    # Filter conversations with only 1 context speaker
    dataset = dataset.filter(lambda example: len(example['context_speakers']) == 1)
    print('--- Dataset after filtering for conversations with only 1 context speaker:', len(dataset))

    # Filter conversations with at least 20 words in context
    dataset = dataset.filter(lambda example: len(' '.join(example['context']).split()) > 20)
    print('--- Dataset after filtering for conversations with at least 20 words in context utterance/s:', len(dataset))

    # Filter conversations with at least 25 persona sentences
    dataset = dataset.filter(lambda example: len(example['personas']) > 25)
    print('--- Dataset after filtering for conversations with at least 25 persona sentences:', len(dataset))

    # Filter conversations with at least 10 words in response utterance
    dataset = dataset.filter(lambda example: len(example['response'].split()) > 10)
    print('--- Dataset after filtering for conversations with at least 10 words in response utterance:', len(dataset))
    
    def persona_input_for_roberta(persona_sentences):
        return ' </s> '.join(['<s> ' + sentence + ' </s>' \
                         for sentence in persona_sentences])

    dataset = dataset.map(lambda example: {'personas': persona_input_for_roberta(example['personas'])})

    return dataset


def model_forward_pass(model, tokenizer, cat_persona_sentences):
    with torch.no_grad():
        
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

    
    # Get last four layers.
    last_four_layers = [outputs.hidden_states[i] for i in (-1, -2, -3, -4)]

    # Cast layers to a tuple and concatenate over the last dimension
    cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)

    # Take the mean of the concatenated vector over the token dimension
    return torch.mean(cat_hidden_states, dim=1).squeeze()
    
