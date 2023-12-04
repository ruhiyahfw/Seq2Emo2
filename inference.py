import os 
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from models.seq2seq_lstm import LSTMSeq2Seq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, get_single_metrics
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
from data.data_loader import load_sem18_data, load_goemotions_data
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.others import find_majority
from utils.file_logger import get_file_logger

MODEL_NAME="model_sem18.pt"

# Argument parser
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--pad_len', default=50, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--folds', default=5, type=int, help="num of folds")
parser.add_argument('--en_lr', default=5e-4, type=float, help="encoder learning rate")
parser.add_argument('--de_lr', default=1e-4, type=float, help="decoder learning rate")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions'])
parser.add_argument('--en_dim', default=1200, type=int, help="dimension")
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--criterion', default='jaccard', type=str, choices=['jaccard', 'macro', 'micro', 'h_loss'])
parser.add_argument('--glove_path', default='data/glove.840B.300d.txt', type=str)
parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--encoder_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--patience', default=13, type=int, help='dropout rate')
parser.add_argument('--download_elmo', action='store_true')
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--warmup_epoch', default=0, type=float, help='')
parser.add_argument('--stop_epoch', default=10, type=float, help='')
parser.add_argument('--max_epoch', default=20, type=float, help='')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--fix_emo_emb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input_feeding', action='store_true', default=True)
parser.add_argument('--dev_split_seed', type=int, default=0)
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=bool, default=True)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--attention_heads', type=int, default=1)
parser.add_argument('--concat_signal', action='store_true')
parser.add_argument('--no_cross', action='store_true')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--attention_type', type=str, default="luong", choices=['transformer', 'luong'])
parser.add_argument('--load_emo_emb', action='store_true')
parser.add_argument('--shuffle_emo', type=str, default=None)
parser.add_argument('--single_direction', action='store_true')
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch
MAX_EPOCH = args.max_epoch
RANDOM_SEED = args.seed
# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



# Init Elmo model
if args.download_elmo:
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
else:
    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
elmo.eval()

GLOVE_EMB_PATH = args.glove_path
glove_tokenizer = GloveTokenizer(PAD_LEN)

data_path_postfix = '_split'
data_pkl_path = 'data/' + args.dataset + data_path_postfix + '_data.pkl'
if not os.path.isfile(data_pkl_path):
    if args.dataset == 'sem18':
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
            load_sem18_data()
    elif args.dataset == 'goemotions':
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
            load_goemotions_data()
    else:
        raise NotImplementedError

    with open(data_pkl_path, 'wb') as f:
        logger('Writing file')
        pkl.dump((X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name), f)

else:
    with open(data_pkl_path, 'rb') as f:
        logger('loading file')
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = pkl.load(f)

NUM_EMO = len(EMOS)


class TestDataReader(Dataset):
    def __init__(self, X, pad_len):
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len):
        super(TrainDataReader, self).__init__(X, pad_len)
        self.y = []
        self.read_target(y)

    def read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]]), \
               torch.LongTensor(self.y[idx])


def elmo_encode(ids):
    data_text = [glove_tokenizer.decode_ids(x) for x in ids]
    with torch.no_grad():
        character_ids = batch_to_ids(data_text).cuda()
        elmo_emb = elmo(character_ids)['elmo_representations']
        elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
    return elmo_emb


def show_classification_report(gold, pred):
    from sklearn.metrics import classification_report
    logger(classification_report(gold, pred, target_names=EMOS, digits=4))

if args.shuffle_emo is not None:
    new_order = np.asarray([int(tmp) for tmp in args.shuffle_emo.split()])
    y_train_dev = np.asarray(y_train_dev).T[new_order].T
    y_test = np.asarray(y_test).T[new_order].T

glove_tokenizer.build_tokenizer(X_train_dev + X_test, vocab_size=VOCAB_SIZE)
glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)

from sklearn.model_selection import ShuffleSplit, KFold

kf = KFold(n_splits=args.folds, random_state=args.dev_split_seed, shuffle=True)
# kf.get_n_splits(X_train_dev)

all_preds = []
gold_list = None

print('test', X_test[0])   
test_set = TestDataReader(X_test, MAX_LEN_DATA)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*3, shuffle=False)

# Model initialize
model = LSTMSeq2Seq(
    emb_dim=SRC_EMB_DIM,
    vocab_size=glove_tokenizer.get_vocab_size(),
    trg_vocab_size=NUM_EMO,
    src_hidden_dim=SRC_HIDDEN_DIM,
    trg_hidden_dim=TGT_HIDDEN_DIM,
    attention_mode=ATTENTION,
    batch_size=BATCH_SIZE,
    nlayers=2,
    nlayers_trg=2,
    dropout=args.dropout,
    encoder_dropout=args.encoder_dropout,
    decoder_dropout=args.decoder_dropout,
    attention_dropout=args.attention_dropout,
    args=args
)

model.load_state_dict(torch.load(args.model_path))
model.eval()

model.load_encoder_embedding(glove_tokenizer.get_embeddings(), fix_emb=args.fix_emb)
model.load_emotion_embedding(glove_tokenizer.get_emb_by_words(GLOVE_EMB_PATH, EMOS))
model.cuda()

preds = []
logger("Testing:")
for i, loader_input in tqdm(enumerate(test_loader), total=int(len(test_set) / BATCH_SIZE)):
    with torch.no_grad():
        src, src_len = loader_input
        elmo_src = elmo_encode(src)
        decoder_logit = model(src.cuda(), src_len.cuda(), elmo_src.cuda())
        preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
        del decoder_logit

preds = np.concatenate(preds, axis=0)
gold = np.asarray(y_test)
binary_gold = gold
binary_preds = preds
logger("NOTE, this is on the test set")
metric = get_metrics(binary_gold, binary_preds)
logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
metric = get_multi_metrics(binary_gold, binary_preds)
logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
# show_classification_report(binary_gold, binary_preds)
logger('Jaccard:', jaccard_score(gold, preds))

with open("preds.txt","w") as fh:
    for pred in preds:
        fh.write(str(pred)+"\n")

with open("golds.txt","w") as fh:
    for x in gold:
        fh.write(str(x)+"\n")
    

def read_sem18(file_name):
    emotion_list = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                    "optimism", "pessimism", "sadness", "surprise", "trust"]
    label_list = []
    text_list = []
    with open(file_name, 'r', encoding="utf-8") as f:
        num_tokens_per_line = None
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if idx == 0:
                column_names = line.split('\t')
                num_tokens_per_line = len(column_names)
            else:
                tokens = line.split('\t')
                assert len(tokens) == num_tokens_per_line

                one_label = [int(x) for x in tokens[2:]]
                assert len(one_label) == len(emotion_list)
                label_list.append(one_label)
                text_list.append(tokens[1])
    return text_list, label_list, emotion_list

def load_sem18_test():
    test_file = 'data/semeval2018t3ec/SemEval18_test.txt'
 
    all_data = []
    all_label = []
    path_prefix = 'data/semeval2018t3ec/'
    emo_list = None

    data, label, emo_list = read_sem18(test_file)
    all_data.extend(data)
    all_label.extend(label)

    return all_data, all_label, emo_list

text, label, emo_list = load_sem18_test()

#ubsh pred
print(EMOS_DIC)
