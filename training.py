from seq2sec.train import train


if __name__ == '__main__':
    # RECEIVED_PARAMS = nni.get_next_parameter()
    # print(RECEIVED_PARAMS)
    # train('data/config/data_test.json', RECEIVED_PARAMS, 'models/teste_mt_3-4.pth')
    train('data/config/data_test.json','models/teste_mt_3-4.pth')
    train('data/config/data_test.json','models/teste_mt_3-4.pth', device='cuda')
