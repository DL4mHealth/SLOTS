


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
        self.input_channels = 32 #DEAP
        # self.input_channels = 62 #SEED
        # self.input_channels = 60  # P19
        # self.input_channels = 1  # Epilepsy
        # self.input_channels = 1  # HAR
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        # self.num_classes = 2#DEAP
        self.num_classes = 3#SEED
        # self.num_classes = 2# P19
        # self.num_classes = 2  # Epilepsy
        # self.num_classes = 6  # HAR
        self.dropout = 0.35
        self.features_len = 18

        # training configs
        self.num_epoch = 30

        self.warm_epochs = 5

        self.gamma = 0.98

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-3
        # self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 100

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.TSlength_aligned = 128#DEAP
        # self.TSlength_aligned = 200#SEED
        # self.TSlength_aligned = 68  # P19
        # self.TSlength_aligned = 178  # Epilepsy
        # self.TSlength_aligned = 178  # HAR
        # self.num_classes_target = 2 #DEAP
        self.num_classes_target = 3 #SEED
        # self.num_classes_target = 2  # P19
        # self.num_classes_target = 2  # Epilepsy
        # self.num_classes_target = 6  # HAR

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



