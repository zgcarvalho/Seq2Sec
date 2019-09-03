import torch
from seq2sec.data import AA, PAD, INPUT_CODE
from sys import argv
import numpy as np
from seq2sec.model import load

MAX_SAS = {
    'A':131.700,
    'C':172.000,
    'D':166.900,
    'E':193.600,
    'F':244.100,
    'G':103.600,
    'H':213.700,
    'I':206.200,
    'K':228.300,
    'L':209.700,
    'M':226.300,
    'N':169.500,
    'P':165.000,
    'Q':197.800,
    'R':255.500,
    'S':141.900,
    'T':164.400,
    'V':180.000,
    'W':285.600,
    'Y':252.000
}


class Protein(object):

    def __init__(self, fasta_file):
        s = self._read_fasta_file(fasta_file)
        assert self._is_ok(s)
        self.seq = s
        self.prediction = {}
        self.probabilities = {}
        self.steps = {}

        self.buried = []
        self.ss3 = []
        self.ss3_steps = []

    def predict_with(self, model):
        # convert string to one hot encoding and add padding
        pred = model.predict(self._pad_input(self._seq2int(self.seq)))
        for k in pred:
            if k == 'buriedI_abs':
                self.prediction[k] = np.squeeze(pred[k][-1][:, :, PAD:-PAD])
                self.steps[k] = np.transpose(np.squeeze(pred[k])[ :, PAD:-PAD])
            else:
                self.probabilities[k] = np.transpose(np.squeeze(pred[k])[:, :, PAD:-PAD][(-1), :, :])
                self.prediction[k] = self._prob2ss(self.probabilities[k])
                self.steps[k] = np.transpose(np.squeeze(pred[k])[:, :, PAD:-PAD], axes=[1, 2, 0])

    def sample_with(self, model, n=20):
        model.train()
        samples = [model.predict(self._pad_input(self._seq2int(self.seq))) for i in range(n)]
        self.buried = np.array([np.squeeze(s['buriedI_abs'][-1][:, :, PAD:-PAD]) for s in samples])
        self.ss3 = np.array([np.transpose(np.squeeze(s['ss_cons_3_label'])[:, :, PAD:-PAD][(-1), :, :]) for s in samples])
        self.ss3_steps = np.array([np.transpose(np.squeeze(s['ss_cons_3_label'])[:, :, PAD:-PAD], axes=[1, 2, 0]) for s in samples])

    def sample2_with(self, model, n=20):
        # model.train()
        samples = [model.predict2(self._pad_input(self._seq2int(self.seq))) for i in range(n)]
        self.buried = np.array([np.squeeze(s['buriedI_abs'][-1][:, :, PAD:-PAD]) for s in samples])
        
        self.ss3 = np.array([np.transpose(np.squeeze(s['ss_cons_3_label'])[:, :, PAD:-PAD][(-1), :, :]) for s in samples])
        self.ss3_steps = np.array([np.transpose(np.squeeze(s['ss_cons_3_label'])[:, :, PAD:-PAD], axes=[1, 2, 0]) for s in samples])

    def plot_probs(self):
        pass

    def plot_steps(self):
        pass

    @staticmethod
    def _prob2ss(prob):
        n_c = prob.shape[1]
        c = ['H', 'E', 'C']
        if n_c == 4:
            c = ['H', 'E', 'C', 'T']
        elif n_c == 8:
            c = ['H', 'G', 'I', 'E', 'C', 'T', 'B', 'S']
        idx = prob.argmax(axis=1)
        return ''.join([c[i] for i in idx])

    @staticmethod
    def _read_fasta_file(fn):
        with open(fn, 'r') as f:
            seq = ''
            for l in f.readlines():
                if l.startswith('>'):
                    continue
                else:
                    seq += l.strip()
        return seq.upper()

    @staticmethod
    def _is_ok(sequence):
        for i in sequence:
            if not i in AA.__members__:
                print('Error: {} is not a valid aminoacid code')
                return False
        return True

    @staticmethod
    def _seq2int(seq):
        code = np.zeros(len(seq), dtype=np.int)
        for i, aa in enumerate(seq):
            code[i] = AA[aa].value
        return code

    @staticmethod
    def _pad_input(x):
        idx = np.pad(x, (PAD, PAD), 'constant', constant_values=(AA[
            'before'].value, AA['after'].value))
        m = torch.zeros((1, INPUT_CODE, len(x) + 2 * PAD), requires_grad=False)
        m[0, idx, np.arange(len(x) + 2 * PAD)] = 1
        return m

    def asa(self, k='relative'):
        if 'buriedI_abs' in self.prediction:
            if k == 'absolute':
                acc = [max(0, MAX_SAS[self.seq[i]] - self.prediction['buriedI_abs'][i]) for i in range(len(self.seq))]
            elif k == 'relative':
                acc = [max(0, MAX_SAS[self.seq[i]] - self.prediction['buriedI_abs'][i])/MAX_SAS[self.seq[i]] for i in range(len(self.seq))]
        return acc

    def plot_probs(self):
        pass

    def plot_steps(self):
        pass

    def iplot_probabilities(self):
        import plotly.graph_objs as go
        from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
        init_notebook_mode(connected=True)
        x = [i for i in range(len(self.seq))]
        y0 = self.probabilities[:, (0)]
        y1 = self.probabilities[:, (1)]
        y2 = self.probabilities[:, (2)]
        n_c = self.probabilities.shape[1]
        data = []
        if n_c == 3 or n_c == 4:
            data.append(go.Bar(x=x, y=self.probabilities[:, (0)], opacity=
                0.4, name='Helix'))
            data.append(go.Bar(x=x, y=self.probabilities[:, (1)], opacity=
                0.4, name='Strand'))
            data.append(go.Bar(x=x, y=self.probabilities[:, (2)], opacity=
                0.4, name='Coil'))
            if n_c == 4:
                data.append(go.Bar(x=x, y=self.probabilities[:, (3)],
                    opacity=0.4, name='Turn'))
        elif n_c == 8:
            data.append(go.Scatter(x=x, y=self.probabilities[:, (0)], line=
                dict(shape='hvh'), name='alpha Helix'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (1)], line=
                dict(shape='hvh'), name='3 Helix'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (2)], line=
                dict(shape='hvh'), name='5 Helix'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (3)], line=
                dict(shape='hvh'), name='Strand'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (4)], line=
                dict(shape='hvh'), name='Coil'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (5)], line=
                dict(shape='hvh'), name='Turn'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (6)], line=
                dict(shape='hvh'), name='beta Bridge'))
            data.append(go.Scatter(x=x, y=self.probabilities[:, (7)], line=
                dict(shape='hvh'), name='Bend'))
        layout = go.Layout(barmode='overlay', yaxis=dict(range=[0, 1],
            hoverformat='%'), xaxis=go.layout.XAxis(ticktext=list(self.seq),
            tickvals=x, tickangle=0, tickfont=dict(family='Courier, mono',
            color='black'), ticks='outside', ticklen=2))
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)


def hello():
    print('Hello!')


if __name__ == '__main__':
    net = load('../models/teste_4.pth')
