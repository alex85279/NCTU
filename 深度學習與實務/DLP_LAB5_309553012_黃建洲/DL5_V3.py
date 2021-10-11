from __future__ import unicode_literals, print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from io import open
import unicodedata
import string
import re
import random
import time
import os
import math
import json
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)
def __save_model(model_name, model, root):
    if not os.path.isdir(root):
        os.mkdir(root)
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    torch.save(model.state_dict(), p)
    return p

def save_model(models, root='./model'):
    p = {}
    for k, m in models.items():
        p[k] = __save_model(k, m, root)
    return p

def __load_model(model_name, model, root):
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    if not os.path.isfile(p):
        msg = "No model parameters file for {}!".format(model_name)
        return print(msg)
        raise AttributeError(msg)
    paras = torch.load(p)
    model.load_state_dict(paras)

def load_model(models, root='./model'):
    for k, m in models.items():
        __load_model(k, m, root)
        
def save_model_by_score(models, bleu_score, root):
    p = os.path.join(root, 'score.json')
    previous = None
    
    if np.isnan(bleu_score):
        raise AttributeError("BLEU score become {}".format(bleu_score))
        return
    
    if os.path.isfile(p):
        with open(p, 'r') as f:
            previous = json.load(f)
            
    if previous is not None and previous['score'] > bleu_score:
        return;
    
    save_model(models, root)
    previous = {'score' : bleu_score}
    with open(p, 'w') as f:
        json.dump(previous, f)
class CharDict:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0
        
        for i in range(26):
            self.addWord(chr(ord('a') + i))
        
        tokens = ["SOS", "EOS"]
        for t in tokens:
            self.addWord(t)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def longtensorFromString(self, s):
        s = ["SOS"] + list(s) + ["EOS"]
        return torch.LongTensor([self.word2index[ch] for ch in s])
    
    def stringFromLongtensor(self, l, show_token=False, check_end=True):
        s = ""
        for i in l:
            ch = self.index2word[i.item()]
            if len(ch) > 1:
                if show_token:
                    __ch = "<{}>".format(ch)
                else:
                    __ch = ""
            else:
                __ch = ch
            s += __ch
            if check_end and ch == "EOS":
                break
        return s

class wordsDataset(Dataset):
    def __init__(self, train=True):
        if train:
            f = './train.txt'
        else:
            f = './test.txt'
        self.datas = np.loadtxt(f, dtype=np.str)
        
        if train:
            self.datas = self.datas.reshape(-1)
        else:
            self.targets = np.array([
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1],
            ])
        
        #self.tenses = ['sp', 'tp', 'pg', 'p']
        self.tenses = [
            'simple-present', 
            'third-person', 
            'present-progressive', 
            'simple-past'
        ]
        self.chardict = CharDict()
        
        self.train = train
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        if self.train:
            c = index % len(self.tenses)
            return self.chardict.longtensorFromString(self.datas[index]), c
        else:
            i = self.chardict.longtensorFromString(self.datas[index, 0])
            ci = self.targets[index, 0]
            o = self.chardict.longtensorFromString(self.datas[index, 1])
            co = self.targets[index, 1]
            
            return i, ci, o, co
train_dataset = wordsDataset()
test_dataset = wordsDataset(False)            
class Encoder(nn.Module):
    def __init__(self, input_size = 28, RNN_hidden_size = 256, latent_size = 32, num_condition = 4, condition_size = 8):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.RNN_hidden_size = RNN_hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        self.condition_embedding = nn.Embedding(num_condition, condition_size)
        self.input_embedding = nn.Embedding(input_size, RNN_hidden_size)
        self.lstm = nn.LSTM(RNN_hidden_size, RNN_hidden_size)
        self.mean_hidden = nn.Linear(RNN_hidden_size, latent_size)
        self.logvar_hidden = nn.Linear(RNN_hidden_size, latent_size)
        self.mean_cell = nn.Linear(RNN_hidden_size, latent_size)
        self.logvar_cell = nn.Linear(RNN_hidden_size, latent_size)
    def condition(self,c):
        c = torch.LongTensor([c]).to(device)
        return self.condition_embedding(c).view(1,1,-1)
    def sampling(self):
        return torch.normal(
            torch.FloatTensor([0]*self.latent_size), 
            torch.FloatTensor([1]*self.latent_size)
        ).to(device)
    def initHiddenCell(self):
        return torch.zeros(
            1, 1, self.RNN_hidden_size - self.condition_size, 
            device=device
        )
    def forward(self, inputs, condition, init_hidden):
        c = self.condition(condition)
        x = self.input_embedding(inputs)
        x = x.view(-1, 1, self.RNN_hidden_size)
        hidden = torch.cat((init_hidden, c), dim=2)
        cell = torch.cat((init_hidden, c), dim=2)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        hidden_mean = self.mean_hidden(hidden)
        hidden_logvar = self.logvar_hidden(hidden)
        hidden_latent = self.sampling() * torch.exp(hidden_logvar/2) + hidden_mean
        
        cell_mean = self.mean_cell(cell)
        cell_logvar = self.logvar_cell(cell)
        cell_latent = self.sampling() * torch.exp(cell_logvar/2) + cell_mean
        
        return hidden_latent, hidden_mean, hidden_logvar, cell_latent, cell_mean, cell_logvar
class Decoder(nn.Module):
    def __init__(self, input_size = 28, RNN_hidden_size = 256, latent_size = 32, condition_size = 8):
        super(Decoder, self).__init__()
        self.RNN_hidden_size = RNN_hidden_size
        self.input_size = input_size
        
        self.latent_to_hidden = nn.Linear(latent_size+condition_size, RNN_hidden_size)
        self.input_embedding = nn.Embedding(input_size, RNN_hidden_size)
        self.condition_embedding = nn.Embedding(4, condition_size)
        self.lstm = nn.LSTM(RNN_hidden_size, RNN_hidden_size)
        self.fc = nn.Linear(RNN_hidden_size, input_size)
        
    def initHiddenCell(self, hidden_latent, cell_latent, c):
        hidden = torch.cat((hidden_latent, c), dim=2)
        cell = torch.cat((cell_latent, c), dim=2)
        return self.latent_to_hidden(hidden), self.latent_to_hidden(cell)
    def condition(self,c):
        c = torch.LongTensor([c]).to(device)
        return self.condition_embedding(c).view(1,1,-1)
    def forward(self, x, hidden):
        x = self.input_embedding(x)
        x = x.view(1,1,self.RNN_hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        output = self.fc(output)
        output = output.view(-1, self.input_size)
        return output, hidden, cell
        
def Decode(decoder, hidden_latent, cell_latent, c, maxlen, use_teacher_forcing = False, inputs = None):
    sos_token = train_dataset.chardict.word2index['SOS']
    eos_token = train_dataset.chardict.word2index['EOS']
    hidden_latent = hidden_latent.view(1,1,-1)
    cell_latent = cell_latent.view(1,1,-1)
    outputs = []
    x = torch.LongTensor([sos_token]).to(device)
    hidden, cell = decoder.initHiddenCell(hidden_latent, cell_latent, c)
    
    for i in range(maxlen):
        x = x.detach()
        output, hidden, cell = decoder(x, (hidden, cell))
        outputs.append(output)
        output_onehot = torch.max(torch.softmax(output, dim=1), 1)[1]
        
        # meet EOS
        if output_onehot.item() == eos_token and not use_teacher_forcing:
            break
        
        if use_teacher_forcing:
            x = inputs[i+1:i+2]
        else:
            x = output_onehot
    
    # get (seq, word_size)
    if len(outputs) != 0:
        outputs = torch.cat(outputs, dim=0)
    else:
        outputs = torch.FloatTensor([]).view(0, 28).to(device)
    
    return outputs

def TestModel(encoder, decoder, testDataset):
    encoder.eval()
    decoder.eval()
    
    bleu_score = []
    
    for i in range(len(testDataset)):
        data = testDataset[i]

        inputs, input_c, targets, target_c = data
        hidden_latent, _, _, cell_latent, _, _ = encoder(inputs[1:].to(device),input_c, encoder.initHiddenCell())
        outputs = Decode(decoder, hidden_latent, cell_latent, decoder.condition(target_c), maxlen=len(targets))  
        targets_str = train_dataset.chardict.stringFromLongtensor(targets, check_end=True)
        outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
        outputs_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot, check_end=True)
        bleu_score.append( compute_bleu(outputs_str, targets_str) )
        print(f"target_str: {targets_str}, outputs_str: {outputs_str}")  
    print(f"BLEU-4 score : {sum(bleu_score) / len(bleu_score)}")
    
    return bleu_score


def Gaussian_score(encoder, decoder):
    encoder.eval()
    decoder.eval()
    g_score = 0
    for i in range(100):
        gaussian_words = []
        sample_hidden = encoder.sampling()
        sample_cell = encoder.sampling()
        
        for k in range(len(train_dataset.tenses)):
            outputs = Decode(decoder, sample_hidden, sample_cell, decoder.condition(k), maxlen=20)
            word = torch.max(torch.softmax(outputs, dim=1), 1)[1]
            word = train_dataset.chardict.stringFromLongtensor(word)
            gaussian_words.append(word)

        match = 0
        for w in gaussian_words:
            for t in train_dataset.datas:
                if w == t:
                    match += 1

        if match == 4:
            g_score += 0.01
        
    return g_score        
#############################        
def KLD_Loss(hidden_m, hidden_logvar, cell_m, cell_logvar):
    return torch.sum(0.5 * (hidden_m ** 2 + torch.exp(hidden_logvar) - hidden_logvar - 1 + cell_m ** 2 + torch.exp(cell_logvar) - cell_logvar - 1))        
        
        
def TrainModel(encoder, decoder, epoch_size = 300, learning_rate = 0.007, KLD_weight = 0, teacher_forcing_ratio = 1):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='sum')    
    ###########
    show_loss_total = 0
    plot_loss_total = 0
    plot_kl_loss_total = 0
    best_score = 0.0
    best_g_score = 0
    kld_w = 0.0
    tfr = 0.0
    metrics = []
    ###########
    for epoch in range(epoch_size):
        print("\nEpoch ", epoch+1)
        encoder.train()
        decoder.train()
        ######### tfr and klw annealing #############
        if epoch % 60 < 15:
            KLD_weight = 0
        else:
            tmp_e = epoch % 60 - 15
            KLD_weight = (tmp_e % 100) * 0.02
            if KLD_weight > 1:
                KLD_weight = 1
        #print(f"KL weight = {KLweight}")
        if epoch > 150:
            teacher_forcing_ratio = 1 - 0.005 * ((epoch - 150))
        if teacher_forcing_ratio < 0:
            teacher_forcing_ratio = 0
        samples = random.sample(list(train_dataset), 5)
        
        print(f"tfr : {teacher_forcing_ratio}, klw: {KLD_weight}")
        #len(train_dataset)
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            #data = samples[i]
            inputs, c = data
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            hidden_latent, hidden_mean, hidden_logvar, cell_latent, cell_mean, cell_logvar = encoder(inputs[1:].to(device), c, encoder.initHiddenCell())
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            outputs = Decode(decoder, hidden_latent, cell_latent, encoder.condition(c), maxlen=inputs[1:].size(0), use_teacher_forcing=use_teacher_forcing, inputs=inputs.to(device))
            output_length = outputs.size(0)
            loss = criterion(outputs, inputs[1:1+output_length].to(device))
            #kld_loss = KL_loss(hidden_m, hidden_logvar) 
            klLoss = KLD_Loss(hidden_mean, hidden_logvar, cell_mean, cell_logvar)     
            total_loss = loss + KLD_weight * klLoss
            total_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            show_loss_total += loss.item() + ( KLD_weight*klLoss.item() )
            plot_loss_total += loss.item()
            plot_kl_loss_total += klLoss.item()
            
            # show output by string
            outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
            inputs_str = train_dataset.chardict.stringFromLongtensor(inputs, show_token=True)
            outputs_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot, show_token=True)
            #print(f"input_str: {inputs_str}, outputs_str: {outputs_str}")    
        
    ################    
        
        score = TestModel(encoder, decoder, test_dataset)
        bleu_score = sum(score) / len(score)
        gaussian_score = Gaussian_score(encoder, decoder)
        '''if gaussian_score >= best_g_score:
            if bleu_score >= 0.5 and bleu_score >= best_score or (best_g_score < 0.05 and gaussian_score >= 0.05 and blue_score >= 0.5):
                best_g_score = gaussian_score
                best_score = bleu_score
                torch.save(encoder.state_dict(), './best_encoder.pkl')
                torch.save(decoder.state_dict(), './best_decoder.pkl')
        print(f"best bleu: {best_score}, best gaussian: {best_g_score}")'''
        if bleu_score >= 0.6 and gaussian_score >= 0.3:
            best_score = bleu_score
            torch.save(encoder.state_dict(), './best_encoder.pkl')
            torch.save(decoder.state_dict(), './best_decoder.pkl')
        
        metrics.append((
            plot_loss_total, plot_kl_loss_total, bleu_score, 
            KLD_weight, teacher_forcing_ratio, learning_rate, gaussian_score
        ))
        
        print("Loss: ", plot_loss_total)
        print("KL Loss: ", plot_kl_loss_total)
        print("Score: ", bleu_score)
        print("Gaussian Score: %.2f" % gaussian_score)
        
        plot_loss_total = 0
        plot_kl_loss_total = 0
        
    return metrics    
def show_curve(df):
    plt.figure(figsize=(10,6))
    plt.title('Training loss/ratio curve')
    
    plt.plot(df.index, df.kl, label='KLD', linewidth=3)
    plt.plot(df.index, df.crossentropy, label='CrossEntropy', linewidth=3)
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    h1, l1 = plt.gca().get_legend_handles_labels()
    
    ax = plt.gca().twinx()
    ax.plot(metrics_df.index, metrics_df.score, '.', label='BLEU4-score',c="C2")
    ax.plot(metrics_df.index, metrics_df.klw, '--', label='KLD_weight',c="C3")
    ax.plot(metrics_df.index, metrics_df.tfr, '--', label='Teacher ratio',c="C4")
    ax.plot(metrics_df.index, metrics_df.gaussian_score, '.', label='Gaussian Score',c="C5")
    ax.set_ylabel('score / weight')
    
    h2, l2 = ax.get_legend_handles_labels()
    
    ax.legend(h1+h2, l1+l2)
    #plt.show()
    plt.savefig('result.png', dpi=300, bbox_inches='tight')
if __name__ == "__main__":
    
    encoder = Encoder().to(device)
    decoder = Decoder().to(device) 
    load_model({'encoder':encoder, 'decoder':decoder})
    #metric_result = TrainModel(encoder, decoder)
    #torch.save(metric_result, os.path.join('.', 'metrics.pkl'))
    #metrics_df = pd.DataFrame(metric_result, columns=["crossentropy", "kl", "score", "klw", "tfr", "lr", "gaussian_score"])
    #show_curve(metrics_df)  
    
    ################
    score = TestModel(encoder, decoder, test_dataset)
    bleu_score = sum(score) / len(score)
    gaussian_score = Gaussian_score(encoder, decoder)
    print(f"Bleu_score: {bleu_score}, Gaussian_score: {gaussian_score}")
    ################    