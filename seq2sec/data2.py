from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
from enum import Enum
import torch

# Secondary structure codes

# Consensus btw dssp, stride and pross
# SS3
#original_choices = ['H', 'E', 'C']
#original_choices_label = [0, 1, 2]

#SS4
#choices = ['H', 'E', 'C', 'T']
#choices_label = [0, 1, 2, 3]

# Consensus btw dssp and stride
#SS8
# choices_ss8 = ['H', 'G', 'I', 'E', 'C', 'T', 'B', 'S']
# choices_ss8_label = [0, 1, 2, 3, 4, 5, 6, 7]



class AA(Enum):
    before = 0
    A = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    K = 9
    L = 10
    M = 11
    N = 12
    P = 13
    Q = 14
    R = 15
    S = 16
    T = 17
    V = 18
    Y = 19
    W = 20
    after = 21


MAX_LENGTH = 4000
PAD = 50
INPUT_CODE = 22

class SSDataset(Dataset):
    '''single task'''
    def __init__(self, json_config, context='training'):
        with open(json_config, 'r') as f:
            config = json.load(f)

            self.path = config['path']
            # for now only the first task will be used 
            self.tasks = config['label']
            self.context = context
            self.examples = self._read(config[context])

            f.close()
        

    def __len__(self):
        return len(self.examples['seq_res'])

    def __getitem__(self, idx):
        ex = {}
        for k in self.examples:
            if k == 'seq_res':
                # ex[k] = self._encode_aa(self.examples[k][idx])
                ex[k] = self._pad_input(self.examples[k][idx])
            else:
                ex[k] = self._pad_label(self.examples[k][idx])

        return ex


    # def __getitem__(self, idx):
    #     ex = self.examples[idx]
    #     # read dataframe
    #     df = pd.read_feather(self.path + ex['id'] + '.fth')
    #     # only use the specific chain 
    #     df = df.loc[df['res_chain'] == ex['chain']]

    #     try: 
    #         input_data = self._encode_aa(df['seq_res'].values)
    #         # again...only the first task is used
    #         output_data = self._pad_label(df[self.tasks[0]].values)
    #     except ValueError as e:
    #         print("ID: {} -> Error: {}", ex, e.value)

    #     return (input_data, output_data)

    def _read(self, examples_from_context):
        examples = {'seq_res': []}
        for t in self.tasks:
            examples[t] = []

        for e in examples_from_context:
            # print(e)
            df = pd.read_feather(self.path + e['id'] + '.fth')
            # only use the specific chain 
            df = df.loc[df['res_chain'] == e['chain']] 
            for k in examples.keys():
                if k == 'seq_res':
                    examples[k].append(self._seq2int(df[k].values))
                else:
                    examples[k].append(df[k].values)

        return examples

    # def _encode_aa(self, seq):
    #     code = torch.zeros((INPUT_CODE, MAX_LENGTH+(2*PAD)), requires_grad=False)

    #     for i in range(PAD):
    #         code[AA['before'].value, i] = 1

    #     for i, aa in enumerate(seq):
    #         code[AA[aa].value, i+PAD] = 1
        
    #     for i in range(PAD+len(seq), (2*PAD)+len(seq)):
    #         code[AA['after'].value, i] = 1

    #     # after (pad)+len(seq)+(pad) the values are zero

    #     return code

    def _seq2int(self, seq):
        code = np.zeros(len(seq), dtype=np.int)
        for i, aa in enumerate(seq):
            code[i] = AA[aa].value
        return code

    def _pad_input(self, x):
        idx = np.pad(x, (PAD,MAX_LENGTH+(2*PAD)-len(x)-PAD), 'constant', constant_values=(AA['before'].value, AA['after'].value))
        m = torch.zeros((INPUT_CODE, MAX_LENGTH+(2*PAD)), requires_grad=False)
        m[idx, np.arange(MAX_LENGTH+(2*PAD))] = 1
        return m


    def _pad_label(self, y):
        return torch.from_numpy(np.pad(y, (PAD,MAX_LENGTH+(2*PAD)-len(y)-PAD), 'constant', constant_values=(-1, -1))).type(torch.int64)

        # code = torch.ones((MAX_LENGTH+ (2*PAD)), dtype=torch.int64) * -1 #-1 is the code to ignore
        
        # for i, v in enumerate(torch.as_tensor(y)):
        #     code[i+PAD] = v

        # return code


# read len of examples and keep fn and len

# create sets of examples until a max size (these sets will be the new examples)

# len -> number of sets

# getitem -> read files and concatenate

if __name__ == "__main__":
    import time
    t0 = time.process_time()
    d = SSDataset('../data/config/data_cath95.json')
    # print(d.examples)
    print("Length", len(d))
    print("Time elapsed: ", time.process_time() - t0)

    t1 = time.process_time()
    print("Start epoch")
    for i in range(len(d)):
        a = d[i]
        # print(a)
        # for i in a:
        #     print(a[i].shape)
    print("Time elapsed: ", time.process_time() - t1)
    

    # d = SSDataset('../data/config/data_cath95.json')
    # print(d.path)
    # print(d.tasks)
    # print(len(d.examples))
    # for i in range(len(d.examples)):
    #     a = d[i]
    #     # print(i)
    #     # print(a[0].shape, a[1].shape)