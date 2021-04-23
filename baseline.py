'''
代码说明：
苏剑林分享的开源baseline,参考链接：https://kexue.fm/archives/8213
利用预训练模型bert，通过PET任务进行语义相似性计算。
在baseline基础上添加了ngram mask、对抗训练、以及各种学习率策略。
环境说明：
# !pip install --upgrade tensorflow-gpu==1.15
# !pip install keras==2.3.1
# !pip install bert4keras==0.10.0
'''


import json
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from bert4keras.tokenizers import load_vocab
from bert4keras.backend import keras, set_gelu, K, search_layer
from keras.layers import Lambda, Dense
from bert4keras.models import build_transformer_model
from bert4keras.snippets import truncate_sequences
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam

from tqdm import tqdm
import os


path = "/content/drive/MyDrive/textSim/"
'''
路径说明：
../code #保存代码
../data #保存数据
../train_subs #保存数据
../chinese_L-12_H-768_A-12/#bert路径
'''

min_count = 3
maxlen = 32  # text1+text2的长度
batch_size = 128
bert_path = '/content/drive/MyDrive/' + 'NEZHA-Base/'
config_path = '/content/drive/MyDrive/' + 'NEZHA-Base/bert_config.json'
checkpoint_path = '/content/drive/MyDrive/' + 'NEZHA-Base/model.ckpt-900000'
dict_path = '/content/drive/MyDrive/' + 'NEZHA-Base/vocab.txt'

#### 加载数据与预处理  ####
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
            truncate_sequences(maxlen, -1, a, b)
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
    t[0]: i + 4
    for i, t in enumerate(tokens)
}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

# BERT词频
counts = json.load(open(path+'/chinese_L-12_H-768_A-12/'+'counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = load_vocab(dict_path)
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])


def can_mask(token_ids):
    if token_ids in (2,3,4):
        return False

    return True
def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 7)
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids

def random_ngram_mask(text_ids):
    '''
    #n-gram masking algorithm
    #30% use unigram , 30% use bigram, 40% use trigram
    '''
    import collections
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])

    import random
    #rng = random.Random(223)
    masked_lm_prob = 0.15
    max_predictions_per_seq = 5
    cand_indexes = []

    for (i, token) in enumerate(text_ids):
        cand_indexes.append(i)
    #rng.shuffle(cand_indexes) 
    random.shuffle(cand_indexes)

    #rands = np.random.random(len(text_ids))
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(text_ids) * masked_lm_prob))))
    masked_lms = []

    input_ids,output_ids=[],[]
    covered_indexes = set()

    for index in cand_indexes:
      if len(masked_lms) >= num_to_mask:
          break
      if index in covered_indexes:
          continue
      
      if np.random.random() >= 0.6*0.85 and index<len(text_ids)-1:
          for ind in [index,index+1]:
              if ind in covered_indexes:
                  continue
              covered_indexes.add(ind)
              # 2 gram 80% of the time, replace with [MASK]
              if np.random.random() < 0.8:
                  input_ids.append(4)
                  output_ids.append(text_ids[ind])
              else:
                  # 10% of the time, keep original
                  if np.random.random() < 0.5:
                      input_ids.append(text_ids[ind])
                      output_ids.append(text_ids[ind])
                  # 10% of the time, replace with random word
                  else:
                      input_ids.append(np.random.choice(len(tokens)) + 7)
                      output_ids.append(text_ids[ind]) 
              masked_lms.append(MaskedLmInstance(index=index, label=text_ids[ind]))
            
      elif np.random.random() >= 0.3*0.85 and index<len(text_ids)-2:
          for ind in [index,index+1,index+2]:
              if ind in covered_indexes:
                  continue
              covered_indexes.add(ind)
              # 2 gram 80% of the time, replace with [MASK]
              if np.random.random() < 0.8:
                  input_ids.append(4)
                  output_ids.append(text_ids[ind])
              else:
                  # 10% of the time, keep original
                  if np.random.random() < 0.5:
                      input_ids.append(text_ids[ind])
                      output_ids.append(text_ids[ind])
                  # 10% of the time, replace with random word
                  else:
                      input_ids.append(np.random.choice(len(tokens)) + 7)
                      output_ids.append(text_ids[ind]) 
              masked_lms.append(MaskedLmInstance(index=index, label=text_ids[ind])) 
      else:
          if index in covered_indexes:
              continue
          covered_indexes.add(index)
          if np.random.random() < 0.8:
              input_ids.append(4)
              output_ids.append(text_ids[index])
          else:
              # 10% of the time, keep original
              if np.random.random() < 0.5:
                  input_ids.append(text_ids[index])
                  output_ids.append(text_ids[index])
              # 10% of the time, replace with random word
              else:
                  input_ids.append(np.random.choice(len(tokens)) + 7)
                  output_ids.append(text_ids[index]) 
              
          masked_lms.append(MaskedLmInstance(index=index, label=text_ids[index]))    

    input=text_ids
    output=[0 for i in range(len(text_ids))]
    if len(covered_indexes)>0:
      for i in range(len(covered_indexes)):
          input[list(covered_indexes)[i]] = input_ids[i] 
          output[list(covered_indexes)[i]] = output_ids[i] 

    return input,output

def sample_convert(text1, text2, label, random=False):
    """转换为MLM格式
    """
    text1_ids = [tokens.get(t, 1) for t in text1]
    text2_ids = [tokens.get(t, 1) for t in text2]
    if random:
        if np.random.random() < 0.5:
            text1_ids, text2_ids = text2_ids, text1_ids
        text1_ids, out1_ids = random_mask(text1_ids)
        text2_ids, out2_ids = random_mask(text2_ids)
    else:
        out1_ids = [0] * len(text1_ids)
        out2_ids = [0] * len(text2_ids)
    token_ids = [2] + text1_ids + [3] + text2_ids + [3]
    segment_ids = [0] * len(token_ids)
    output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
    return token_ids, segment_ids, output_ids
    
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2, label, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
                
                
#### 定义一些模型训练的相关函数  ######
def masked_crossentropy(y_true, y_pred):
    """mask掉非预测部分
    """
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(y_true, 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]

def adversarial_training(model, embedding_name, epsilon=1):
        """给模型添加对抗训练
        其中model是需要添加对抗训练的keras模型，embedding_name
        则是model里边Embedding层的名字。要在模型compile之后使用。
        """
        if model.train_function is None:  # 如果还没有训练函数
            model._make_train_function()  # 手动make
        old_train_function = model.train_function  # 备份旧的训练函数

        # 查找Embedding层
        for output in model.outputs:
            embedding_layer = search_layer(output, embedding_name)
            if embedding_layer is not None:
                break
        if embedding_layer is None:
            raise Exception('Embedding layer not found')

        # 求Embedding梯度
        embeddings = embedding_layer.embeddings  # Embedding矩阵
        gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
        gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

        # 封装为函数
        inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
        )  # 所有输入层
        embedding_gradients = K.function(
            inputs=inputs,
            outputs=[gradients],
            name='embedding_gradients',
        )  # 封装为函数

        def train_function(inputs):  # 重新定义训练函数
            grads = embedding_gradients(inputs)[0]  # Embedding梯度
            delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
            K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
            outputs = old_train_function(inputs)  # 梯度下降
            K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
            return outputs

        model.train_function = train_function  # 覆盖原训练函数

class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

####  模型训练与测试  ######
#固定随机数
import random
random.seed(233) # 生成同一个随机数
import numpy as np
np.random.seed(233) # 生成同一个随机数；
import gc
import keras.backend as KFold
import math

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from bert4keras.optimizers import *
from tensorflow.keras.callbacks import LearningRateScheduler


kf = StratifiedKFold(n_splits=9,shuffle=True, random_state=233) # 9折交叉验证，分层采样。

fold = 0
result = []
i=1
for train_index, test_index in kf.split(data, [d[-1] for d in data]):
    if i==1:
      train_data= [data[i] for i in train_index]
      valid_data= [data[i] for i in test_index]
      
      # 模拟未标注
      for d in valid_data + test_data:
          train_data.append((d[0], d[1], -5))
      
      # 转换数据集
      train_generator = data_generator(train_data, batch_size)
      valid_generator = data_generator(valid_data, batch_size)
      test_generator = data_generator(test_data, batch_size)
      
      
      # 加载预训练模型
      model = build_transformer_model(
          config_path=config_path,
          checkpoint_path=checkpoint_path,
          with_mlm=True,
          model = 'nezha',
          keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)])
      
      # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      # monitor='val_score', factor=0.1, patience=4, verbose=0, mode='max')

      # learning rate schedule  当EpochDrop=10时表示每经过10epochs，学习率变为原来的一半。
      # def step_decay(epoch):
      #   initial_lrate = 6e-7
      #   drop = 0.5
      #   epochs_drop = 10.0
      #   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
      #   return lrate
      # lrate = LearningRateScheduler(step_decay)


      # lookahead优化器、带权重衰减和warmup的优化器
      # ahead = extend_with_lookahead(Adam, 'ahead')
      AdamW = extend_with_weight_decay(Adam, 'AdamW')
      # optimizer = AdamW(learning_rate=5e-5, weight_decay_rate=0.01)

      # AdamWLR = extend_with_piecewise_linear_lr(AdamW, 'AdamWLR')
      
      optimizer = AdamW(learning_rate=5e-5,
                weight_decay_rate=0.01)
      
      # model.load_weights('/content/drive/MyDrive/textSim/bert_result/'+'nezha-2.weights')  # 断点继续训练
      model.compile(loss=masked_crossentropy, optimizer=optimizer)


      model.summary()

      # 启用对抗训练只需要一行代码
      adversarial_training(model, 'Embedding-Token', 0.5)

      def evaluate(data):
          """
          线下评测函数
          """
          Y_true, Y_pred = [], []
          for x_true, y_true in data:
              y_pred = model.predict(x_true)[:, 0, 5:7]
              y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
              y_true = y_true[:, 0] - 5
              Y_pred.extend(y_pred)
              Y_true.extend(y_true)
          return roc_auc_score(Y_true, Y_pred)

      class Evaluator(keras.callbacks.Callback):
          """评估与保存
          """
          def __init__(self):
              self.best_val_score = 0.

          def on_epoch_end(self, epoch, logs=None):
              val_score = evaluate(valid_generator)
              if val_score > self.best_val_score:
                  self.best_val_score = val_score
                  model.save_weights('/content/drive/MyDrive/textSim/bert_result/'+'nezha-zheng.weights')
              print(
                  u'val_score: %.5f, best_val_score: %.5f\n' %
                  (val_score, self.best_val_score)
              )

      def predict_to_file(out_file):
          """预测结果到文件
          """
          F = open(out_file, 'w')
          for x_true, _ in tqdm(test_generator):
              y_pred = model.predict(x_true)[:, 0, 5:7]
              y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
              for p in y_pred:
                  F.write('%f\n' % p)
          F.close()
          
      
      evaluator = Evaluator()
      warmup = WarmUpLearningRateScheduler(warmup_batches=1500, init_lr=5e-5)

      model.fit(
          train_generator.forfit(),
          steps_per_epoch=len(train_generator),
          epochs=150,
          callbacks=[warmup, evaluator],
          #initial_epoch=56   #如果断点继续训练，加该参数

      )
      
      out_file='/content/drive/MyDrive/textSim/bert_result/'+ str(fold+1)+'.txt'
      F = open(out_file, 'w')
      for x_true, _ in tqdm(test_generator):
          y_pred = model.predict(x_true)[:, 0, 5:7]
          y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
          result.append(y_pred)
          for p in y_pred:
              F.write('%f\n' % p)
      F.close()
                  
      fold +=1
      print(str(fold)+" fold is finished!")
      
      del model
      gc.collect()
      K.clear_session()

    i += 1

    

