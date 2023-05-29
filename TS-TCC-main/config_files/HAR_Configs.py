


# class Config(object):
#     def __init__(self):
#         # model configs
#         self.input_channels = 9
#         self.kernel_size = 8
#         self.stride = 1
#         self.final_out_channels = 128
#
#         self.num_classes = 6
#         self.dropout = 0.35
#         self.features_len = 18
#
#         # training configs
#         self.num_epoch = 40
#
#         # optimizer parameters
#         self.beta1 = 0.9
#         self.beta2 = 0.99
#         self.lr = 3e-4
#
#         # data parameters
#         self.drop_last = True
#         self.batch_size = 128
#
#         self.Context_Cont = Context_Cont_configs()
#         self.TC = TC()
#         self.augmentation = augmentations()

class Config(object):
    def __init__(self):
        # model configs
        # self.input_channels = 32
        # self.input_channels = 62 #SEED
        self.input_channels = 60  # P19
        # self.input_channels = 1  # Epilepsy
        # self.input_channels = 1  # HAR
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        # self.num_classes = 2
        # self.num_classes = 3#SEED
        self.num_classes = 2  # P19
        # self.num_classes = 2  # Epilepsy
        # self.num_classes = 6  # HAR
        self.dropout = 0.35
        # self.features_len = 18 #DEAP
        # self.features_len = 27 #SEED
        # self.features_len = 24  # HAR,Epilepsy
        self.features_len = 11  # P19

        # training configs
        self.num_epoch = 30

        self.warm_epochs = 5

        self.gamma = 0.98

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        # self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6



