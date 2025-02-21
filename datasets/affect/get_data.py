"""Implements dataloaders for AFFECT data."""
import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from numpy.core.numeric import zeros_like
from torch.nn.functional import pad
from torch.nn import functional as F

sys.path.append(os.getcwd())
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pdb

np.seterr(divide='ignore', invalid='ignore')

def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed

def get_norm_stats(dataset, norm_modal, max_seq_len=50):
    """Get stats of the training dataset"""
    stats = {}
    for modal in norm_modal:
        unprocessed = dataset[modal][:, :max_seq_len, :]
        if modal == "audio":
            unprocessed[unprocessed == -np.inf] = 0.0
        stats[modal] = (unprocessed.mean((0, 1)), unprocessed.std((0, 1)))
    return stats

def norm_with_stats(dataset, stats, max_seq_len=50):
    processed = {}
    for modal, unprocessed in dataset.items():
        if modal in ["vision", "audio", "text"]:
            unprocessed = dataset[modal][:, :max_seq_len, :]
            if modal == "audio":
                unprocessed[unprocessed == -np.inf] = 0.0
            if modal in stats:
                processed[modal] = (unprocessed - stats[modal][0]) / stats[modal][1]
            else:
                processed[modal] = unprocessed
    processed['labels'] = dataset['labels']
    return processed

def get_rawtext(path, data_kind, vids):
    """Get raw text, video data from hdf5 file."""
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


def _get_word2id(text_data, vids):
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['unk']
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()
        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def _return_unk():
        return UNK

    word2id.default_factory = _return_unk
    return data_processed, word2id


def _get_word_embeddings(word2id, save=False):
    from torchtext import text
    vec = text.vocab.GloVe(name='840B', dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)
    
    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def _glove_embeddings(text_data, vids, paddings=50):
    data_prod, w2id = _get_word2id(text_data, vids)
    word_embeddings_looks_up = _get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        # TODO: fix some segs have more than 50 words (FIXed)
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(np.zeros(300, ))
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:
        
        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


class Affectdataset(Dataset):
    """Implements Affect data as a torch dataset."""
    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None, max_pad=False, max_pad_num=50, 
                 data_type='mosi', z_norm=False) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            data_type (str, optional): What data to load. Defaults to 'mosi'.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])
        

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num((vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            if data_type in ['mosi', 'mosei', 'sarcasm']:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                flag = flag.item()
                return [[flag]]
        
        tmp_label = self.dataset['labels'][ind]
        if self.data_type == 'humor' or self.data_type == 'sarcasm':
            if (self.task == None) or (self.task == 'regression'):
                if self.dataset['labels'][ind] < 1:
                    tmp_label = [[-1]]
                else:
                    tmp_label = [[1]]
        else:
            tmp_label = self.dataset['labels'][ind]
            # tmp_label = [self.dataset['labels'][ind].item()]

        label = torch.tensor(_get_class(tmp_label)).long() if self.task == "classification" else torch.tensor(
            tmp_label).float()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind, \
                    label]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = (F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0])), tmp[i].shape[0])
            else:
                # pdb.set_trace()
                tmp = [vision, audio, text, ind, label]
            return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]


def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, robust_test=False, data_type='mosi', 
        raw_path='/home/van/backup/pack/mosi/mosi.hdf5', norm_modal=None, z_norm=False) -> DataLoader:
    """Get dataloaders for affect data.

    Args:
        filepath (str): Path to datafile
        batch_size (int, optional): Batch size. Defaults to 32.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
        max_pad (bool, optional): Whether to pad data to max length or not. Defaults to False.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 2.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        task (str, optional): Which task to load in. Defaults to None.
        robust_test (bool, optional): Whether to apply robustness to data or not. Defaults to False.
        data_type (str, optional): What data to load in. Defaults to 'mosi'.
        raw_path (str, optional): Full path to data. Defaults to '/home/van/backup/pack/mosi/mosi.hdf5'.
        z_norm (bool, optional): Whether to normalize data along the z dimension or not. Defaults to False.

    Returns:
        DataLoader: tuple of train dataloader, validation dataloader, test dataloader
    """
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    process = eval("_process_2") if max_pad else eval("_process_1")

    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]
    if norm_modal is not None:
        train_stats = get_norm_stats(processed_dataset['train'], norm_modal, max_seq_len=max_seq_len)
        for dataset in processed_dataset:
            processed_dataset[dataset] = norm_with_stats(processed_dataset[dataset], train_stats, max_seq_len=max_seq_len)
    train = DataLoader(Affectdataset(processed_dataset['train'], flatten_time_series, task=task, max_pad=max_pad,               max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=lambda b: process(b, task))
    valid = DataLoader(Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=lambda b: process(b, task))
    assert not robust_test
    test = DataLoader(Affectdataset(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                    shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                    collate_fn=lambda b: process(b, task))
    return train, valid, test

def _process_1(inputs: List, task):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:
        
        inds.append(sample[-2])
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])
    tensor_labels = torch.tensor(labels) if task == "classification" else torch.tensor(labels).view(len(inputs), 1)
    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), tensor_labels


def _process_2(inputs: List, task):
    processed_input = []
    labels = []
    # Iterate over modality
    for i in range(len(inputs[0]) - 1):
        feature = []
        input_masks = []
        for sample in inputs:
            feature.append(sample[i][0])
            # Get the original lengths
            padded_length = sample[i][0].shape[0]
            # sample[i][1] is the original input length
            mask = torch.arange(padded_length) >= sample[i][1]
            input_masks.append(mask)
        processed_input.append((torch.stack(feature), torch.stack(input_masks)))
    for sample in inputs:
        
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])
    tensor_labels = torch.tensor(labels) if task == "classification" else torch.tensor(labels).view(len(inputs), 1)
    return processed_input[0], processed_input[1], processed_input[2], tensor_labels


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    traindata, validdata, test_robust = get_dataloader('./mosi/mosi_data.pkl', robust_test=False, max_pad=True, num_workers=0)
    # traindata, validdata, test_robust = get_dataloader('./mosei/mosei_raw.pkl', robust_test=False, max_pad=True, data_type='mosei', num_workers=0)
    # batch1 = next(iter(traindata))
    # traindata, validdata, test_robust = get_dataloader('./mosei/mosei_senti_data.pkl', robust_test=False, max_pad=True, data_type='mosei', num_workers=0)
    # batch2 = next(iter(traindata))

    # 371, 81, 300 for sarcasm/humor
    # traindata, validdata, test_robust = get_dataloader('./sarcasm/sarcasm.pkl', robust_test=False, max_pad=True, task='classification', data_type='sarcasm', num_workers=0)
    # batch1 = next(iter(traindata))
    # traindata, validdata, test_robust = get_dataloader('./humor/humor.pkl', robust_test=False, max_pad=True, 
    #                                                    data_type='humor', task="classification", num_workers=0, norm_modal=["vision", "audio"])
    # batch2 = next(iter(traindata))
    # traindata, validdata, test_robust = \
    # get_dataloader('./sarcasm/sarcasm.pkl', robust_test=False, max_pad=False, task='classification', data_type='sarcasm', max_seq_len=40, num_workers=0)

    # keys = list(test_robust.keys())
    

    # for batch in traindata:
    
    
    total_pos = 0
    total_data = 0
    for batch in traindata:
        total_pos += batch[3].sum()
        total_data += len(batch[3])
    print(total_data)
    print(total_pos / total_data)
    
    #     break
    for batch in traindata:
        # pdb.set_trace()
        print(batch[0][0].shape)
        print(batch[1][0].shape)
        print(batch[2][0].shape)
        print(batch[3][0].shape)
        break

    # test_robust[keys[0]][1]
    for batch in test_robust:
        print(batch[-1])
        break
        # for b in batch:
            
            
        
        
        
        # break
