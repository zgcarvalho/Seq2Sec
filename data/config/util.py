import json
import os

def check(d, t):
    for i in d[t]:
        if os.path.isfile(d['path']+i['id']+'.fth'):
            continue
        else:
            print('{}'.format(i['id']))

if __name__ == "__main__":
    f = open("data_cath95.json")
    d = json.load(f)
    check(d, 'training')
    check(d, 'testing')
    check(d, 'validation')
