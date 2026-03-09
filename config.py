# config.py

class Config:
    # ==========================
    # 训练基础参数
    # ==========================
    epoch = 200
    lr = 1e-4
    grad_norm = 0.5
    batchsize = 16
    weight_decay = 1e-5
    momentum = 0.9  # 原本的 mt 参数
    power = 0.5

    # ==========================
    # 数据与路径参数
    # ==========================
    trainsize = 512
    train_path = './data/TrainDataset'
    train_save_name = 'iESegNet_result2'

    # 自动生成的保存路径
    save_path = f'checkpoints/{train_save_name}/'
    log_dir = f'logs/{train_save_name}/'

    # ==========================
    # 多尺度训练参数
    # ==========================
    # 图像缩放比例，原代码中写死在 train 函数内部
    size_rates = [0.75, 1, 1.25]

    # ==========================
    # 硬件与环境参数
    # ==========================
    # 是否开启多GPU (使用布尔值替代原本的字符串)
    mgpu = False

    # ==========================
    # 记录与保存频率
    # ==========================
    print_freq = 30  # 每多少个step打印一次日志
    save_freq = 30  # 每多少个epoch保存一次权重
