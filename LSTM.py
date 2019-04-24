
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
    sents = fl.read().split('\n\n')[1:-1]
    
with open('eng.testa') as fl:
    eval_sents = fl.read().split('\n\n')[1:-1]
    
with open('eng.testb') as fl:
    test_sents = fl.read().split('\n\n')[1:-1]
    
len(sents), len(eval_sents), len(test_sents)


# In[17]:


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


# In[18]:


BSIZE = 1
train_inds = list(range(len(sents)))
shuffle(train_inds)
train_inds[:10]


# In[19]:


import torch.optim as optim


# In[20]:


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


# In[21]:


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


# In[22]:


plt.figure(figsize=(14, 3))
plt.plot(train_loss)
plt.plot(eval_loss[1:])
plt.show(); plt.close()


# In[22]:


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


# In[25]:


get_ipython().system('python conlleval1.py nn.output')

