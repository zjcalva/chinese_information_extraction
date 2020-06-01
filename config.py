from __future__ import division


class Config():
    def __init__(self):
        self.root_path = "./"

        # for data loader
        self.data_set = "ace05"
        self.batch_size = 8
        self.if_shuffle = True

        # override when loading data
        self.voc_size = None
        self.pos_size = None
        self.label_size = None
        self.relation_size=None
        self.actions = None
        self.gaz_alphabet_size=None

        self.gaz_alphabet=None
        self.id2relation=None
        self.id2word=None
        self.id2label=None
        self.id2action=None




        # embed size
        self.token_embed = 50    # word embedding dimension
        self.action_embed = 20   # action embedding dimension
        self.entity_embed = self.action_embed
        self.input_dropout = 0
        self.lstm_dropout = 0

        # for lstm
        self.if_treelstm = True
        self.rnn_layers = 1       #
        self.hidden_dim = 50
        self.bilstm_flag=True
        self.HP_fix_gaz_emb=False

        self.pretrain_gaz_embedding=True
        self.gaz_emb_dim=50
        self.gaz_dropout=0.33

        # reversed, for convenience of buffer
        self.reversed = False

        # for training
        self.embed_path = self.root_path + "/data/word_vec_{0}_{1}.pkl".format(self.data_set, self.token_embed)
        self.gaz_path = self.root_path + "/data/word_vec_{0}_gaz_{1}.pkl".format(self.data_set, self.token_embed)

        self.epoch = 500
        self.if_gpu = True
        self.opt = "Adam"
        self.lr = 0.005 # [0.3, 0.00006]
        self.l2 = 1e-4

        self.check_every = 1
        self.clip_norm = 3

        # for early stop
        self.lr_patience = 6
        self.decay_patience = 3

        self.pre_trained = True
        self.data_path = self.root_path + "/data/{0}".format(self.data_set)
        self.model_path = self.root_path + "/data/{0}_model.pt".format(self.data_set)



        #



    def __repr__(self):
        return str(vars(self))


config = Config()
