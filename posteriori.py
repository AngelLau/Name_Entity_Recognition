
# coding: utf-8

# In[1]:


import gensim
import numpy as np
import gensim.models.keyedvectors as Word2Vec
import pandas as pd
import nltk
import torch
from torch import nn
import json
import matplotlib.pyplot as plt
from numpy.random import shuffle
import numpy as np
from time import time
import os, sys
import csv


# In[2]:


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  


# In[3]:


with open('eng.train') as fl:
    sents = fl.read().split('\n\n')[1:-1]
len(sents)


# In[4]:


sentences = []
# [[(word, label), (...), ...] , [...], [...]]
ner_tags = {}
for meta in sents:
    pairs = []
    for line in meta.split('\n'):
        entries = line.split()
        # word, NER tag
        pairs.append([entries[0], entries[-1]])
        ner_tags[entries[-1]] = True
    sentences.append(pairs)
    
ner_tags = list(ner_tags.keys())

print(len(sentences))
print(len(ner_tags))
print(sorted(ner_tags))


# In[5]:


lookup = ['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'I-PER', 'O']
for ent in lookup:
    assert ent in ner_tags # sanity check
ldict = { tag: ind for ind, tag in enumerate(lookup) }


# In[6]:


with open('lookup.json', 'w') as fl:
    json.dump(ldict, fl, indent=4)


# In[7]:


missing = {}
uniques = {}


# In[8]:


for sent in sentences:
    for word, tag in sent:
        if word not in model:
            if word not in missing:
                missing[word] = True
#                 print('Not in embedding:', word)
                uniques[word] = model['unk']
        else:
            uniques[word] = model[word]

uniques['unk'] = model['unk'] # also save unk for later use

print('Unique words (found in embedding):', len(uniques))
print('Missing in embedding:', len(missing))


# In[9]:


subset_words = sorted(list(uniques.keys()))


# In[10]:


with open('word_list.txt', 'w') as fl:
    fl.write('\n'.join(subset_words))


# In[11]:


emat = np.zeros((len(subset_words), 300))


# In[12]:


for si, word in enumerate(subset_words):
    emat[si, :] = uniques[word]


# In[13]:


np.save('word_embeds.npy', emat)


# In[14]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[15]:


with open('lookup.json') as fl:
    tag_lookup = json.load(fl)
    
with open('word_list.txt') as fl:
    vocab = fl.read().split('\n')

embeds = np.load('word_embeds.npy')
wlookup = { word: index for index, word in enumerate(vocab) }

len(vocab), embeds.shape, len(tag_lookup)


# In[16]:


with open('eng.train') as fl:
    test_sents = fl.read().split('\n\n')[1:-1]
with open('eng.testa') as fl:
    test_sents+=fl.read().split('\n\n')[1:-1]
with open('eng.testb') as fl:
    test_sents+=fl.read().split('\n\n')[1:-1]


# In[17]:


with open('eng.testa') as fl:
    eval_sents = fl.read().split('\n\n')[1:-1]


# In[18]:


def load_sent(sent):
    words = sent.split('\n')
    inps = []
    outs = []
    for wordinfo in words:
        word, _, _, tag = wordinfo.split()
        try:
            assert word in wlookup
        except:
            # word not in our known dictionary, so use the unk token
            word = 'unk'
        inps.append(embeds[wlookup[word]])
        hot = np.zeros(len(tag_lookup))
        hot[tag_lookup[tag]] = 1
        outs.append(hot)
    return [np.vstack(inps), np.vstack(outs)]

ins, outs = load_sent(sents[0])
ins.shape, outs


# In[19]:


BSIZE = 1
train_inds = list(range(len(sents)))
shuffle(train_inds)
train_inds[:10]


# In[20]:


import torch.optim as optim


# In[21]:


class RNN(nn.Module):
    def __init__(self, insize=300, outsize=8, hsize=128):
        super().__init__()
        
        # TODO: Dropout
        # TODO: nonlinearities
        # TODO: Bidirectional

        self.hsize = hsize
        self.inp = nn.Sequential(
            nn.Linear(insize, hsize),
        )
        self.out = nn.Sequential(
            nn.Linear(hsize*2, outsize),
            nn.Softmax(dim=-1),
        )

        # FIXME: this is a uni-directional LSTM
        self.rnn = nn.LSTM(hsize, hsize, 1, batch_first=True, bidirectional = True, dropout = 1)

    def forward(self, inputs, hidden=None):
        hin = self.inp(inputs)
        
        hout, hidden = self.rnn(hin)
        
        yout = self.out(hout)
        
        return yout, hidden
    
model = RNN().to(device)
criterion = nn.MSELoss().cuda()
opt = optim.Adam(model.parameters(), lr=0.0005)
# sch = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)


# In[22]:


EPS = 3
train_loss = []
eval_loss = []
n2t = lambda narr: torch.from_numpy(narr).to(device).float()

def eval_model(evaldata, results=False):
    model.eval()
    losses = []
    ypreds = []
    for sent in evaldata:
        Xs, Ys = zip(*[load_sent(sent)])
        Xs, Ys = np.array(Xs), np.array(Ys)
        Xs, Ys = n2t(Xs), n2t(Ys)
        
        with torch.no_grad():
            yhat, _ = model(Xs)
            ypreds.append(yhat)
            loss = criterion(yhat, Ys)
            losses.append(loss.item())
    print('Eval: %.4f' % np.mean(losses))
    
    if results: 
        return ypreds
    else:
        return np.mean(losses)
    
eval_loss.append(eval_model(eval_sents))

for epoch in range(EPS):
    model.train()
    t0 = time()
    batch_loss = []
    for bi in range(0, len(train_inds)-BSIZE, BSIZE):
        inds = train_inds[bi:bi+BSIZE]

        # TODO: correct formatting for batchsize >1
        Xs, Ys = zip(*[load_sent(sents[ind]) for ind in inds])
        Xs, Ys = np.array(Xs), np.array(Ys)
        Xs, Ys = n2t(Xs), n2t(Ys)
        # shape: (batch x seqlen x dim)

        yhat, _ = model(Xs)

        opt.zero_grad()
        loss = criterion(yhat, Ys)
        loss.backward()
        opt.step()

        sys.stdout.write('\r[E%d/%d - B%d/%d] Train: %.4f ' % (
            epoch+1, EPS,
            bi+1, len(train_inds),
            loss.item(),
        ))
        batch_loss.append(loss.item())
    train_loss.append(np.mean(batch_loss))
    sys.stdout.write('(elapsed: %.2fs)\n' % (time() - t0))
    sys.stdout.flush()
    
    loss = eval_model(eval_sents)
    eval_loss.append(loss)
        
    shuffle(train_inds)
    # TODO: shuffle train inds


# In[23]:


test_results = eval_model(test_sents, results=True)


# In[24]:


index2tag = { index: tag for tag, index in tag_lookup.items() }
with open('nn.output' , 'w') as fl:
    assert len(test_results) == len(test_sents)
    for ti, (yhat, schunk) in enumerate(zip(test_results, test_sents)):
        yhat = yhat.detach().cpu().numpy()
        for wi, wordinfo in enumerate(schunk.split('\n')):
            word, _, _, tag = wordinfo.split()
            taghat = index2tag[np.argmax(yhat[0, wi, :])]

            fl.write('%s %s %s\n' % (word, taghat, tag))
        fl.write('\n')


# In[380]:


data = []
with open("nn.output") as fd:
    sents = csv.reader(fd,delimiter=" ")
    for row in sents:
       data.append(row)


# In[738]:


df_data = pd.DataFrame(data)


# In[739]:


df_data.rename(columns = {0: 'entity',2:'NER'},inplace = True)


# In[740]:


df_data = df_data[['entity','NER']]


# In[741]:


df = df_data.dropna().reset_index()


# In[742]:


df = df[df['entity']!='-DOCSTART-']


# In[384]:


aida = []
with open("AIDA-YAGO2-dataset.tsv") as fl:
    sents = csv.reader(fl, delimiter="\t")
    for row in sents:
        if (len(row)>0):
            if ('DOCSTART' not in row[0]):
                aida.append(row)


# In[722]:


Aida = pd.DataFrame(aida)


# In[723]:


Aida.rename(columns = {0:'Entity',3:'sequence'},inplace = True)


# In[724]:


Aida = Aida[['Entity','sequence']]


# In[725]:


Aiya = Aida[~Aida.Entity.str.contains("\n")]


# In[726]:


Aiyaya = Aiya[~Aiya.Entity.str.contains("\t")]


# In[743]:


dfa = df[~df.entity.str.contains("\n")]


# In[744]:


dfay = dfa[~dfa.entity.str.contains("\t")]


# In[745]:


finaldf = dfay[:123595]


# In[746]:


finalAida = Aiyaya[:123595]


# In[771]:


df_final = finaldf.reset_index()


# In[772]:


Aida_final = finalAida.reset_index()


# In[776]:


final = df_final.merge(Aida_final,on = df_final.index)


# In[779]:


final = final[['entity','NER','sequence']]


# In[781]:


final = final.dropna()


# In[782]:


final = final[final['sequence']!='--NME--']


# In[802]:


final_agg = final.groupby(['entity','NER','sequence']).size().reset_index(name='countsAll')


# In[803]:


final_agg1 = final_agg.groupby(['entity','NER']).size().reset_index(name='countsEntNER')


# In[804]:


agg= final_agg.merge(final_agg1,on = ['entity','NER'])


# In[805]:


agg1 = agg.groupby(['sequence', 'NER']).size().reset_index(name='countsSeqNER')


# In[806]:


agg1 = agg1.merge(agg, on=['sequence', 'NER'])
agg1.head(20)


# In[814]:


agg1['probSeqNER'] = agg1['countsSeqNER'] / sum(agg1['countsSeqNER'])
agg1['probEntNER'] = agg1['countsEntNER'] / sum(agg1['countsEntNER'])
agg1['probAll'] = agg1['countsAll'] / sum(agg1['countsAll'])
agg1['prob'] = agg1['probAll'] / agg1['probSeqNER']


# In[830]:


agg1.head()
idx = agg1.sort_values('prob').groupby(['sequence', 'NER'], as_index=False).first()
print(agg1.shape)
print(idx.shape)
idx


# In[833]:


idx.to_csv('posteriori.csv', sep='\t',index=False)

