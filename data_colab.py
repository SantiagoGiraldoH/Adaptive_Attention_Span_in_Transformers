"""
data_colab.py - Versión optimizada para Colab
Incluye descarga automática de datasets
"""

import os
import torch
import urllib.request
import zipfile


class Dictionary(object):
    def __init__(self, path, sort_dict=False):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []

        assert os.path.exists(path), f"File not found: {path}"
        
        # Build vocabulary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if sort_dict:
                        self.word2count[word] = self.word2count.get(word, 0) + 1
                    elif word not in self.word2idx:
                        self.word2idx[word] = len(self.idx2word)
                        self.idx2word.append(word)
        
        if sort_dict:
            sorted_dict = sorted(self.word2count.items(), 
                                key=lambda kv: kv[1], reverse=True)
            for i, (word, count) in enumerate(sorted_dict):
                self.word2idx[word] = i
                self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


def _tokenize(text_path, dictionary):
    """Tokenize text file"""
    print(f'Tokenizing {text_path}')
    assert os.path.exists(text_path), f"File not found: {text_path}"

    ids = []
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ['<eos>']
            for token in tokens:
                ids.append(dictionary[token])
    
    ids = torch.LongTensor(ids)
    return ids


class Corpus:
    def __init__(self, data_path, sort_dict):
        print('Building dictionary')
        self._dictionary = Dictionary(
            os.path.join(data_path, 'train.txt'), sort_dict)

        self.train = _tokenize(
            os.path.join(data_path, 'train.txt'),
            self._dictionary.word2idx)
        self.valid = _tokenize(
            os.path.join(data_path, 'valid.txt'),
            self._dictionary.word2idx)
        self.test = _tokenize(
            os.path.join(data_path, 'test.txt'),
            self._dictionary.word2idx)

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    """Batchify data"""
    nb_batches = data_tensor.size(0) // batch_size
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def download_text8(data_dir='data'):
    """Download and prepare text8 dataset"""
    text8_dir = os.path.join(data_dir, 'text8')
    
    if os.path.exists(text8_dir) and \
       os.path.exists(os.path.join(text8_dir, 'train.txt')):
        print(f'text8 already exists at {text8_dir}')
        return text8_dir
    
    print('Downloading text8...')
    os.makedirs(text8_dir, exist_ok=True)
    
    # Download
    url = 'http://mattmahoney.net/dc/text8.zip'
    zip_path = os.path.join(text8_dir, 'text8.zip')
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(text8_dir)
    
    # Prepare splits
    print('Preparing splits...')
    with open(os.path.join(text8_dir, 'text8'), 'r') as f:
        data = f.read()
    
    # Split: 90M train, 5M val, 5M test
    train_data = data[:90000000]
    val_data = data[90000000:95000000]
    test_data = data[95000000:100000000]
    
    # Add spaces between characters (character-level)
    train_data = ' '.join(list(train_data))
    val_data = ' '.join(list(val_data))
    test_data = ' '.join(list(test_data))
    
    # Save
    with open(os.path.join(text8_dir, 'train.txt'), 'w') as f:
        f.write(train_data)
    with open(os.path.join(text8_dir, 'valid.txt'), 'w') as f:
        f.write(val_data)
    with open(os.path.join(text8_dir, 'test.txt'), 'w') as f:
        f.write(test_data)
    
    print(f'text8 prepared at {text8_dir}')
    return text8_dir


def download_enwik8(data_dir='data'):
    """Download and prepare enwik8 dataset"""
    enwik8_dir = os.path.join(data_dir, 'enwik8')
    
    if os.path.exists(enwik8_dir) and \
       os.path.exists(os.path.join(enwik8_dir, 'train.txt')):
        print(f'enwik8 already exists at {enwik8_dir}')
        return enwik8_dir
    
    print('Downloading enwik8...')
    os.makedirs(enwik8_dir, exist_ok=True)
    
    # Download
    url = 'http://mattmahoney.net/dc/enwik8.zip'
    zip_path = os.path.join(enwik8_dir, 'enwik8.zip')
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(enwik8_dir)
    
    # Prepare splits
    print('Preparing splits...')
    with open(os.path.join(enwik8_dir, 'enwik8'), 'rb') as f:
        data = f.read()
    
    # Split: 90M train, 5M val, 5M test
    train_data = data[:90000000]
    val_data = data[90000000:95000000]
    test_data = data[95000000:100000000]
    
    # Add spaces between bytes (character-level)
    train_data = ' '.join([str(b) for b in train_data])
    val_data = ' '.join([str(b) for b in val_data])
    test_data = ' '.join([str(b) for b in test_data])
    
    # Save
    with open(os.path.join(enwik8_dir, 'train.txt'), 'w') as f:
        f.write(train_data)
    with open(os.path.join(enwik8_dir, 'valid.txt'), 'w') as f:
        f.write(val_data)
    with open(os.path.join(enwik8_dir, 'test.txt'), 'w') as f:
        f.write(test_data)
    
    print(f'enwik8 prepared at {enwik8_dir}')
    return enwik8_dir


def _build_corpus(data_path, sort_dict):
    """Build corpus with caching"""
    # Cache path
    corpus_path = os.path.join(
        data_path, 
        'corpus_sorted.pt' if sort_dict else 'corpus.pt'
    )
    
    if os.path.exists(corpus_path):
        print(f'Loading corpus from {corpus_path}')
        corpus = torch.load(corpus_path)
    else:
        print(f'Creating corpus at {corpus_path}')
        corpus = Corpus(data_path, sort_dict)
        torch.save(corpus, corpus_path)
    
    return corpus


def get_train_val_test_data(data_params, env_params, batch_size, 
                            device, sort_dict):
    """Main function to get data - Colab version"""
    
    data_path = data_params['data_path']
    
    # Auto-download if needed
    if 'text8' in data_path and not os.path.exists(data_path):
        data_path = download_text8()
        data_params['data_path'] = data_path
    elif 'enwik8' in data_path and not os.path.exists(data_path):
        data_path = download_enwik8()
        data_params['data_path'] = data_path
    
    # Build corpus
    corpus = _build_corpus(data_path, sort_dict)
    data_params['vocab_size'] = corpus.vocab_size
    
    # Batchify
    train_data = _batchify(corpus.train, batch_size)
    val_data = _batchify(corpus.valid, batch_size)
    test_data = _batchify(corpus.test, batch_size)
    
    # No distributed in Colab
    assert not env_params['distributed'], \
        "Distributed training not supported in Colab"
    
    # Move to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    print(f'Data loaded:')
    print(f'  Vocab size: {corpus.vocab_size}')
    print(f'  Train: {train_data.shape}')
    print(f'  Valid: {val_data.shape}')
    print(f'  Test: {test_data.shape}')
    
    return train_data, val_data, test_data
