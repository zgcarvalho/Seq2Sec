from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
from enum import Enum
import torch


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
    W = 19
    Y = 20
    after = 21


MAX_LENGTH = 4000
PAD = 50
INPUT_CODE = 22


class SSDataset(Dataset):
    """SSDataset contain the data to train the Seq2Sec network.
    """

    def __init__(self, json_config, context='training'):
        with open(json_config, 'r') as f:
            config = json.load(f)
            self.path = config['path']
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
                ex[k] = self._pad_input(self.examples[k][idx])
            elif k == 'buriedI_abs':
                ex[k] = self._pad_abs(self.examples[k][idx])
            else:
                ex[k] = self._pad_label(self.examples[k][idx])
        return ex

    def _read(self, examples_from_context):
        examples = {'seq_res': []}
        for t in self.tasks:
            examples[t] = []
        for e in examples_from_context:
            df = pd.read_feather(self.path + e['id'] + '.fth')
            df = df.loc[df['res_chain'] == e['chain']]
            for k in examples.keys():
                if k == 'seq_res':
                    examples[k].append(self._seq2int(df[k].values))
                else:
                    examples[k].append(df[k].values)
        return examples

    @staticmethod
    def _seq2int(seq):
        code = np.zeros(len(seq), dtype=np.int)
        for i, aa in enumerate(seq):
            code[i] = AA[aa].value
        return code

    @staticmethod
    def _pad_input(x):
        idx = np.pad(x, (PAD, MAX_LENGTH + 2 * PAD - len(x) - PAD),
            'constant', constant_values=(AA['before'].value, AA['after'].value)
            )
        m = torch.zeros((INPUT_CODE, MAX_LENGTH + 2 * PAD), requires_grad=False
            )
        m[idx, np.arange(MAX_LENGTH + 2 * PAD)] = 1
        return m

    @staticmethod
    def _pad_label(y):
        return torch.from_numpy(np.pad(y, (PAD, MAX_LENGTH + 2 * PAD - len(
            y) - PAD), 'constant', constant_values=(-1, -1))).type(torch.int64)

    @staticmethod
    def _pad_abs(y):
        return torch.from_numpy(np.pad(y, (PAD, MAX_LENGTH + 2 * PAD - len(
            y) - PAD), 'constant', constant_values=(np.nan, np.nan))).type(torch.float)


if __name__ == '__main__':
    import time
    t0 = time.process_time()
    d = SSDataset('../data/config/data_cath95.json')
    print('Length', len(d))
    print('Time elapsed: ', time.process_time() - t0)
    t1 = time.process_time()
    print('Start epoch')
    for i in range(len(d)):
        a = d[i]
    print('Time elapsed: ', time.process_time() - t1)
    t2 = time.process_time()
    print('Start batch')
    loader = torch.utils.data.DataLoader(d, batch_size=16, shuffle=True,
        num_workers=4)
    for i, sample in enumerate(loader):
        print(i, sample['seq_res'].size(), sample['ss_cons_3_label'].size())
    print('Time elapsed: ', time.process_time() - t2)
