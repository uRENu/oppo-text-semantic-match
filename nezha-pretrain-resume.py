'''
代码说明：
预训练断点继续训练
'''

from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.utils import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
import numpy as np
import tensorflow as tf

import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras


path = "/content/drive/MyDrive/textSim/"
'''
路径说明：
../code #保存代码
../data #保存数据
../train_subs #保存数据
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
    #15% use unigram , 20% use bigram, 30% use trigram, 20% use four gram, 15% use five gram
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

def sample_convert(text1, text2):
    """转换为MLM格式
    """
    text1_ids = [tokens.get(t, 1) for t in text1]
    text2_ids = [tokens.get(t, 1) for t in text2]
    if np.random.random() < 0.5:
        text1_ids, text2_ids = text2_ids, text1_ids
    text1_ids, out1_ids = random_ngram_mask(text1_ids)
    text2_ids, out2_ids = random_ngram_mask(text2_ids)

    token_ids = [2] + text1_ids + [3] + text2_ids + [3]
    segment_ids = [0]+ [0]*len(text1_ids) + [0] + [1]*len(text2_ids) + [1]
    output_ids = [0] + out1_ids + [0] + out2_ids + [0]
    return token_ids, segment_ids, output_ids

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_output_ids,batch_is_masked = [], [], [],[]
        y=[]
        for is_end, (text1,text2,_) in self.get_sample(shuffle):

            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2)
            is_masked = [0 if i == 0 else 1 for i in output_ids]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            batch_is_masked.append(is_masked)
            y.append([0.])
            
            if is_end or  len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids,maxlen=maxlen)
                batch_segment_ids = pad_sequences(batch_segment_ids,maxlen=maxlen)
                batch_output_ids = pad_sequences(batch_output_ids,maxlen=maxlen)
                batch_is_masked = pad_sequences(batch_is_masked,maxlen=maxlen)

                yield [batch_token_ids, batch_segment_ids, batch_output_ids, batch_is_masked], [np.array(y),np.array(y)]
                batch_token_ids, batch_segment_ids, batch_output_ids, batch_is_masked = [], [], [], []
                y=[]
                
# 打乱数据
# 加载数据集
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]

# 模拟未标注
for d in valid_data + test_data:
    train_data.append((d[0], d[1], -5))
np.random.shuffle(train_data)
train_generator = data_generator(train_data, batch_size=batch_size)
for each in train_generator:
  # print(each)
  # print(len(e[0]))
  print(each[0][1])
  # print(each[0][2][0])
  # print(each[0][3][0])
  # print(each[1])
  break
  
####  定义训练相关函数  #####  
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
        else:
          lr = self.init_lr*1
          K.set_value(self.model.optimizer.lr, lr)

                

# 继续训练

from tensorflow.keras.callbacks import LearningRateScheduler
import math
epochs = 300
learning_rate = 5e-5

# 修改了bert_config.json的 vocab_size的长度  Input to reshape is a tensor with 9448 values, but the requested shape has 21128
config_path =  'NEZHA-Base/bert_config.json'

bert = build_transformer_model(
      config_path,
      with_mlm='linear',
      return_keras_model=False,
      model = 'nezha',
      keep_tokens=[0, 100, 101, 102, 103] + keep_tokens[:len(tokens)])
proba = bert.model.output

# 传入之前训练的权重
bert.model.load_weights('/content/drive/MyDrive/textSim/bert_result/'+'nezha-pretrain-ngram.weights')


# 辅助输入
token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记
# print(token_ids)
# print(bert.model.inputs+ [token_ids, is_masked])

def mlm_loss(inputs):
    """计算loss的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs
    y_true =  K.cast(y_true, K.floatx())
    mask =  K.cast(mask, K.floatx())
    loss = K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
    return loss


def mlm_acc(inputs):
    """计算准确率的函数，需要封装为一个层
    """
    y_true, y_pred, mask = inputs
    #         _, y_pred = y_pred
    #y_true = K.cast(y_true, K.floatx())
    y_true = K.cast(y_true,K.floatx())
    acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
    return acc

def step_decay(epoch):
  initial_lrate = 5e-5
  drop = 0.5
  epochs_drop = 10.0
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return lrate
lrate = LearningRateScheduler(step_decay)

mlm_loss = Lambda(mlm_loss, output_shape=(None, ),name='mlm_loss')([token_ids, proba, is_masked])
mlm_acc = Lambda(mlm_acc, output_shape=(None, ),name='mlm_acc')([token_ids, proba, is_masked])

train_model = Model(
    bert.model.inputs + [token_ids, is_masked], [mlm_loss, mlm_acc])

loss = {
    'mlm_loss': lambda y_true, y_pred: y_pred,
    'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
}


Opt1 = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt1)
opt = Opt(learning_rate=learning_rate,
          exclude_from_weight_decay=['Norm', 'bias'],
          weight_decay_rate=0.01,
          grad_accum_steps=2,
          )

#train_model.compile(loss=loss, optimizer=opt, metrics=[lr_metric])
train_model.compile(loss=loss, optimizer=opt)

train_model.summary()

adversarial_training(train_model, 'Embedding-Token', 0.5)

model_saved_path = '/content/drive/MyDrive/textSim/bert_result/nezha-pretrain-ngram-go.weights'
class ModelCheckpoint(keras.callbacks.Callback):
    """
        每50个epoch保存一次模型
    """

    def __init__(self):
        self.loss = 1e6

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(train_model.optimizer.lr)
        print('learning rate: ',lr)
        if logs['loss'] < self.loss:
            self.loss = logs['loss']
            self.model.save_weights(model_saved_path)

            print('current epoch:', epoch+1)

        #         print('epoch: {}, loss is : {}, lowest loss is:'.format(epoch, logs['loss'], self.loss))

        # if (epoch + 1) % 50 == 0:
        #     bert.save_weights_as_checkpoint(model_saved_path + '-{}'.format(epoch + 1))

# 保存模型
# _info = 'model.epoch{epoch:02d}_loss{loss:.4f}'
# checkpoint = ModelCheckpoint(model_saved_path+_info+'.ckpt',monitor='loss',verbose=2,save_best_only=True, mode='min')
checkpoint = ModelCheckpoint()
# 记录日志
csv_logger = keras.callbacks.CSVLogger('/content/drive/MyDrive/textSim/bert_result/training-ngram.log')

train_model.fit(
    train_generator.generator(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    #callbacks=[lrate,checkpoint, csv_logger],
    callbacks=[checkpoint, csv_logger],
    initial_epoch = 108,
)