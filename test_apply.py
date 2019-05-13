#%%
from seq2sec.apply import Protein
from seq2sec.model import load
import matplotlib.pyplot as plt
import numpy as np

#%%
p = Protein('./fasta_sequences/start2fold/P0A7Y4.fasta')

#%%
net = load('./models/teste-ss3_ss4_buried.pth')

#%%
a = p.predict_with(net.predict)

#%%
a

#%%
tmp = a['buriedI_abs'][-1][:,:,50:-50]

#%%
plt.plot(np.squeeze(tmp))

#%%


#%%
