import torch
from seq2sec.data import AA, PAD, INPUT_CODE
from sys import argv
import numpy as np

class Protein(object):
    def __init__(self, fasta_file):
        s = self._read_fasta_file(fasta_file)
        assert(self._is_ok(s))
        self.seq = s 
        # self.one_hot = self._encode_aa(s) 
        self.prediction = None
        self.steps = None
        

    def predict_with(self, model_func):
        y_pred = model_func(self._encode_aa(self.seq))
        # remove sequence padding
        self.prediction = np.transpose(np.squeeze(y_pred)[:,:,PAD:-PAD][-1,:,:])
        # transpose axis to create C matrix LxD where C is the number of classes, L is the length 
        # of the protein sequence and D is depth of the resnet (number of blocks)
        self.steps = np.transpose(np.squeeze(y_pred)[:,:,PAD:-PAD], axes=[1,2,0])
        

    def _read_fasta_file(self,fn):
        with open(fn, 'r') as f:
            seq = ""
            for l in f.readlines():
                if l.startswith('>'): 
                    continue
                else:
                    seq += l.strip()
        return seq.upper()

    def _is_ok(self,sequence):
        for i in sequence:
            if not i in AA.__members__:
                print("Error: {} is not a valid aminoacid code")
                return False
        return True

    def _encode_aa(self, seq):
        code = torch.zeros((1, INPUT_CODE, len(seq)+(2*PAD)), requires_grad=False)

        for i in range(PAD):
            code[0, AA['before'].value, i] = 1

        for i, aa in enumerate(seq):
            code[0, AA[aa].value, i+PAD] = 1
        
        for i in range(PAD+len(seq), (2*PAD)+len(seq)):
            code[0, AA['after'].value, i] = 1

        # after (pad)+len(seq)+(pad) the values are zero

        return code


if __name__ == "__main__":
    # load model and set to eval
    net = torch.load('./teste_4.pth') 
    net.eval()

    # read fasta
    # x = torch.randn((1,22,10))
    p = Protein("../fasta_sequences/start2fold/STF0001.fasta")
    p.predict_with(net.predict)
    print(p.prediction)
    print(p.prediction.shape)
    print(p.steps.shape)
    # x = p.one_hot
    # print(x)
    # print(x.shape)
    # p.prediction = net.predict(x)
    # print(pred)
    # print(pred.shape)

    # apply model to fasta and create a dataframe
