
import os, sys, json

def gen_vocab(path):
    vocab = {'#pad#':0, '<s>':1, '</s>':2, '<unk>':3, }
    tags = {}
    for line in open(path, 'rU'):
        for i, x in enumerate(line.strip().split()):
            if i == 0:
                continue
            elif i%2 == 1:
                x = x.lower()
                if x not in vocab: vocab[x] = len(vocab)
            else:
                if x not in tags: tags[x] = len(tags)
    return vocab, tags

def make_indices(path, vocab, tags):
    data = []
    for line in open(path, 'rU'):
        cur_data = []
        line = line.strip().split()
        for i in range(1, len(line), 2):
            x = line[i].lower()
            x = vocab[x] if x in vocab else vocab['<unk>']
            y = line[i+1]
            if y not in tags:
                tags[y] = len(tags)
            y = tags[y]
            cur_data.append((x,y))
        data.append(cur_data)

    data_x = []
    data_y = []
    st, ed = vocab['<s>'], vocab['</s>']
    for sent in data:
        for i in range(len(sent)):
            x = []
            x.append(st if i-2 < 0 else sent[i-2][0])
            x.append(st if i-1 < 0 else sent[i-1][0])
            x.append(sent[i][0])
            x.append(ed if i+1 >= len(sent) else sent[i+1][0])
            x.append(ed if i+2 >= len(sent) else sent[i+2][0])
            data_x.append(x)
            data_y.append([sent[i][1],])
    return data_x, data_y


vocab, tags = gen_vocab('../data/pos/train')

train_x, train_y = make_indices('../data/pos/train', vocab, tags)
test_x, test_y = make_indices('../data/pos/test', vocab, tags)
dev_x, dev_y = make_indices('../data/pos/dev', vocab, tags)

data = {'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y, 'dev_x':dev_x, 'dev_y':dev_y, 'vocab':vocab, 'tags':tags,}
json.dump(data, open('data.json','wb'))

