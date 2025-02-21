
from datasets.avmnist.get_data import get_dataloader as get_dataloader_avmnist, get_dataloader_mfcc as get_dataloader_avmnist_mfcc
from datasets.affect.get_data import get_dataloader as get_dataloader_affect

def get_dataloader_fn(dataset_name):
    if dataset_name == "avmnist_mfcc":
        return get_dataloader_avmnist_mfcc
    if dataset_name == "mosi" or dataset_name == "mosei" or \
       dataset_name == "sarcasm" or dataset_name == "humor":
        return get_dataloader_affect
    raise NotImplementedError
    