from seq2sec.train import train


if __name__ == '__main__':
    # RECEIVED_PARAMS = nni.get_next_parameter()
    # print(RECEIVED_PARAMS)
    # train('data/config/data_test.json', RECEIVED_PARAMS, 'models/teste_mt_3-4.pth')
    # train('data/config/data_cath95.json','models/resnet-uncloss-dropblock-simpleblock-ss3_ss4_buried.pth', epochs=1000, device='cuda')
    train('data/config/data_cath95.json','models/resnet-uncloss-dropblock-simpleblock-ss3_ss4_buried.pth', epochs=1000, device='cuda')

    # train('data/config/data_cath95.json','models/resnet_uncertloss_dropout2_cath95_ss3-ss4-buried.pth', device='cuda', epochs=200)
    # train('data/config/data_test.json','models/test_hyper_ss3-ss4.pth', device='cuda', epochs=20)
