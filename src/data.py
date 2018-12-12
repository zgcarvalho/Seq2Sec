from torch.utils.data import Dataset
import json

class SSDataset(Dataset):
    def __init__(self, json_config, use='training'):
        with open(json_config, 'r') as f:
            config = json.load(f)

            self.path = config['path']
            self.tasks = config['label']
            self.examples = config[use]
            
            f.close()


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        pass


# read len of examples and keep fn and len

# create sets of examples until a max size (these sets will be the new examples)

# len -> number of sets

# getitem -> read files and concatenate