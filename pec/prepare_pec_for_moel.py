import numpy as np
from datasets import load_from_disk

def transform(data, mode):
    contexts = [context[0].split(' . ') for context in data['context']]
    np.save('pec_dialog_texts.' + mode + '.npy', contexts)

    np.save('pec_target_texts.' + mode + '.npy', data['response_speaker'])

    np.save('pec_persona_embedding.' + mode + '.npy', data['persona_embedding'])
    
    print('Transformed', mode)


if __name__ == '__main__':
    train = load_from_disk('pec-train')
    val = load_from_disk('pec-val')
    test = load_from_disk('pec-test')
    
    transform(train, 'train')
    transform(val, 'dev')
    transform(test, 'test')
