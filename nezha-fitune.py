'''
代码说明：
与baseline的PET任务不同，将bert的预训练模型学习任务修改为 预训练+微调

环境说明：
# !pip install toolkit4nlp==0.5.0 
# !pip install --upgrade tensorflow-gpu==1.15
'''

from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.utils import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
import numpy as np
import pandas as pd
import tensorflow as tf

import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

path = "/content/drive/MyDrive/textSim/"
'''
路径说明：
../code #保存代码
../data #保存数据
../train_subs #保存数据
../chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 #bert路径
'''

min_count = 3
maxlen = 32
batch_size = 128
bert_path = 'NEZHA-Base/'
config_path =  'NEZHA-Base/bert_config.json'
checkpoint_path =  'NEZHA-Base/model.ckpt-900000'
dict_path =  'NEZHA-Base/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            # truncate_sequences(maxlen, -1, a, b)
            D.append((a, b, c))
#             print (D)
    return D


# 统计词频

data = load_data(path+'data/gaiic_track3_round1_train_20210228.tsv')
test_data = load_data(path+'data/gaiic_track3_round1_testA_20210228.tsv')

tokens = {}
for d in data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])
tokens = {
    t[0]: i + 5
    for i, t in enumerate(tokens)
}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 

# BERT词频
counts = json.load(open(path+'counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = load_vocab(dict_path)
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])


#固定随机数
import random
random.seed(10) # 生成同一个随机数
import numpy as np
np.random.seed(10) # 生成同一个随机数；
import tensorflow as tf
#tf.random.set_seed(10)
tf.set_random_seed(10)
#一些调优参数
er_patience = 2 #early_stopping patience
lr_patience = 5 #ReduceLROnPlateau patience
max_epochs  = 5 #epochs
lr_rate   = 2e-5 #learning rate
batch_sz  = 64 #batch_size
maxlen    = 32 #设置序列长度为，base模型要保证序列长度不超过512
lr_factor = 0.85 #ReduceLROnPlateau factor
drop_rate = 0.1
n_folds   = 10 # 交叉验证折数
n_cls = 2  #class num

from sklearn.utils import shuffle
train_d = shuffle(train_d)

class test_generator:
    global batch_sz
    global maxlen
    def __init__(self, data, batch_size=batch_sz, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
 
    def __len__(self):
        return self.steps
 
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
 
            if self.shuffle:
                np.random.shuffle(idxs)
 
            X1, X2, Y = [], [], []
            for i in idxs:
              d = self.data[i]
              text1 = d[0]
              text2 = d[1]
              label = d[2]
              text1_ids = [tokens.get(t, 1) for t in text1]
              text2_ids = [tokens.get(t, 1) for t in text2]
              x1 = [2] + text1_ids + [3] + text2_ids + [3]
              first_segment_ids = [0] * len(text1_ids)
              second_segment_ids = [1] * len(text2_ids)
              x2 = [0] + first_segment_ids + [0] + second_segment_ids + [1]
              y = label
              X1.append(x1)
              X2.append(x2)
              Y.append([y])
              if len(X1) == self.batch_size or i == idxs[-1]:
                  X1 = pad_sequences(X1, maxlen=maxlen)
                  X2 = pad_sequences(X2, maxlen=maxlen)
                  Y = pad_sequences(Y, maxlen=1)
                  yield [X1, X2], Y[:, 0, :]
                  #yield [X1, X2], Y
                  [X1, X2, Y] = [], [], []

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, shuffle=False):
        X1, X2, Y = [], [], []
        for is_end, (text1,text2,label) in self.get_sample(shuffle):
            text1_ids = [tokens.get(t, 1) for t in text1]
            text2_ids = [tokens.get(t, 1) for t in text2]
            x1 = [2] + text1_ids + [3] + text2_ids + [3]
            first_segment_ids = [0] * len(text1_ids)
            second_segment_ids = [1] * len(text2_ids)
            x2 = [0] + first_segment_ids + [0] + second_segment_ids + [1]
            y = label
            X1.append(x1)
            X2.append(x2)
            Y.append([y])
            #Y.append(list(y))
            
            if is_end or len(X1) == self.batch_size:
              X1 = pad_sequences(X1, maxlen=maxlen)
              X2 = pad_sequences(X2, maxlen=maxlen)
              Y = pad_sequences(Y, maxlen=1)
              yield [X1, X2], Y[:, 0, :]
              #yield [X1, X2], Y
              [X1, X2, Y] = [], [], []
              
              

#交叉验证训练和测试模型
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from keras.layers import *
from keras.callbacks import *
import gc

def run_sk(trn_data, val_data, data_labels, data_test, train_model_pred, test_model_pred, train_index, test_index, fold):

    global er_patience
    global lr_patience
    global max_epochs
    global f1metrics
    global lr_factor
    global max_epochs
    global drop_rate

    # 构建模型
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        # checkpoint_path=checkpoint_path,
        model='nezha',
        # with_pool=True,
        return_keras_model=False,
        keep_tokens=[0, 100, 101, 102, 103] + keep_tokens[:len(tokens)]
    )
    bert.model.load_weights('/content/drive/MyDrive/textSim/bert_result/'+'nezha-pretrain-ngram-go-go-go-go.weights', by_name=True)

    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dense(
        units=n_cls,
        activation='softmax',
        #activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    # output = Dropout(rate=drop_rate)(bert.model.output)  #rate需要丢弃的输入比例。
    # output = Dense(units=n_cls,
    #         activation='softmax',
    #         kernel_initializer=bert.initializer)(output)
    
    model = keras.models.Model(bert.model.input, output)
    model.summary()
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr_rate),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    #model=load_model(weight_file) #载入已保存的模型文件
    early_stopping = EarlyStopping(monitor='val_auc', patience=er_patience)   #早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_auc", verbose=1, mode='max', factor=lr_factor, patience=lr_patience) #当评价指标不在提升时，减少学习率
    # _info = 'model.epoch{epoch:02d}_val_loss{val_loss:.4f}_val_accuracy{val_accuracy:.4f}_val_auc{val_auc:.4f}'
    checkpoint = ModelCheckpoint('/content/drive/MyDrive/textSim/bert_result/fitune-ngram.hdf5', monitor='val_auc',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型

    train_D = data_generator(trn_data, batch_size=batch_sz)
    valid_D = data_generator(val_data, batch_size=batch_sz)
    test_D = test_generator(data_test, batch_size=batch_sz)  # data_generator 会打乱样本的顺序，为了方便提交预测结果，自己写了一个test_generator按顺序预测test data
    #模型训练
    model.fit_generator(
        train_D.generator(),
        steps_per_epoch=len(train_D),
        epochs=max_epochs,
        validation_data=valid_D.generator(),
        validation_steps=len(valid_D),
        callbacks=[checkpoint, early_stopping, plateau],
    )

    # model.load_weights('./bert_dump/' + str(i) + '.hdf5')

    # return model
    train_model_pred[test_index, :] = model.predict_generator(valid_D.generator(), steps=len(valid_D), verbose=1)
    train_model_pred[train_index, :] = model.predict_generator(train_D.generator(), steps=len(train_D), verbose=1)
    #test_model_pred += model.predict_generator(test_D.generator(), steps=len(test_D), verbose=1)
    test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

    del model
    gc.collect()   #清理内存
    K.clear_session()   #clear_session就是清除一个session
    # break

    return train_model_pred, test_model_pred


#n折交叉验证
global n_folds

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=222)

X_trn = pd.DataFrame()
X_val = pd.DataFrame()

fold = 1
for train_index, test_index in skf.split(train_d, [d[-1] for d in train_d]):
  X_trn = [train_d[i] for i in train_index]
  X_val = [train_d[i] for i in test_index]

  DATA_LIST = []
  for data_row in X_trn:
      DATA_LIST.append((data_row[0], data_row[1], to_categorical(data_row[2], n_cls)))
  DATA_LIST = np.array(DATA_LIST)


  VAL_LIST = []
  for data_row in X_val:
    VAL_LIST.append((data_row[0], data_row[1], to_categorical(data_row[2], n_cls)))
  VAL_LIST = np.array(VAL_LIST)

  
  DATA_LIST_TEST = []
  for data_row in test_d:
      DATA_LIST_TEST.append((data_row[0], data_row[1], to_categorical(0, n_cls)))
  DATA_LIST_TEST = np.array(DATA_LIST_TEST)
  
  train_model_pred = np.zeros((len(train_d), n_cls))
  test_model_pred = np.zeros((len(test_d), n_cls))

  train_model_p, test_model_p = run_sk(DATA_LIST, VAL_LIST, None, DATA_LIST_TEST, train_model_pred, test_model_pred, train_index, test_index, fold)
 
  out_file='/content/drive/MyDrive/textSim/bert_result/'+ 'fitune-ngram' +str(fold)+ '-results.txt'
  F = open(out_file, 'w')
  i=1
  for q,p in test_model_p:
    if i<len(test_model_p):
      F.write('%f\n' % p)
    else:
      F.write('%f' % p)
    i += 1
  F.close()
  
  print('fold '+ str(fold)+' is finish!')
  
  fold += 1

