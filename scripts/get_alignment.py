from collections import OrderedDict
import torch
import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
import numpy as np
from unimodals.common_models import Transformer # noqa
import utils.align_metrics as metrics
from torchvision.models.feature_extraction import create_feature_extractor
from utils.align_metrics import AlignmentMetrics
all_metrics = AlignmentMetrics.SUPPORTED_METRICS
import pdb

def get_all_hidden(x1, encoder1, x2, encoder2):
    def get_return_nodes(encoder):
        encoder_return_nodes = {}
        layer_idx = 1
        for name, _ in encoder.named_modules():
            if "relu" in name:
                encoder_return_nodes[name] = layer_idx
                layer_idx += 1
        return encoder_return_nodes
    encoder1_return_nodes = get_return_nodes(encoder1)
    encoder2_return_nodes = get_return_nodes(encoder2)
    with torch.no_grad():
        extractor1 = create_feature_extractor(encoder1, return_nodes=encoder1_return_nodes)
        all_hidden1 = extractor1(x1.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        extractor2 = create_feature_extractor(encoder2, return_nodes=encoder2_return_nodes)
        all_hidden2 = extractor2(x2.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        return all_hidden1, all_hidden2

def get_all_hidden_transformer(x1, encoder1, x2, encoder2):
    activation = {"encoder1": OrderedDict(), "encoder2": OrderedDict()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def get_activation(model_name, layer_name):
        def hook(model, input, output):
            activation[model_name][layer_name] = output.detach()
        return hook
    for layer_idx in range(encoder1.n_layers):
        encoder1.transformer.layers[layer_idx].register_forward_hook(get_activation("encoder1", layer_idx))
    for layer_idx in range(encoder2.n_layers):
        encoder2.transformer.layers[layer_idx].register_forward_hook(get_activation("encoder2", layer_idx))
    with torch.no_grad():
        if isinstance(x1, tuple) or isinstance(x1, list):
            encoder1((x1[0].to(device).float(), x1[1].to(device)))
        else:
            encoder1(x1.to(device).float())
        if isinstance(x2, tuple) or isinstance(x2, list):
            encoder2((x2[0].to(device).float(), x2[1].to(device)))
        else:
            encoder2(x2.to(device).float())
        return activation["encoder1"], activation["encoder2"]

def get_alignment(hidden1, hidden2, metric, contract="feat", topk=10, q=0.95):
    with torch.no_grad():
        # Remove outliers and normalize
        h1 = metrics.remove_outliers(hidden1, q, exact=True, max_threshold=None)
        h2 = metrics.remove_outliers(hidden2, q, exact=True, max_threshold=None)
        norm_h1 = nn.functional.normalize(h1, dim=-1)
        norm_h2 = nn.functional.normalize(h2, dim=-1)
        if "knn" in metric:
            score = metrics.AlignmentMetrics.measure(metric, norm_h1, norm_h2, topk)
        elif metric == "cmd":
            score = metrics.AlignmentMetrics.measure(metric, norm_h1, norm_h2, contract=contract)
        elif metric == "cmd_unnorm":
            score = metrics.AlignmentMetrics.measure("cmd", h1, h2, contract=contract)
        elif metric == "centered_cmd":
            score = metrics.AlignmentMetrics.measure(metric, norm_h1, norm_h2, contract=contract)
        elif metric == "centered_cmd_unnorm":
            score = metrics.AlignmentMetrics.measure("centered_cmd", h1, h2, contract=contract)
        else:
            score = metrics.AlignmentMetrics.measure(metric, norm_h1, norm_h2)
        return score 

def get_alignment_all_layers(x1, encoder1, x2, encoder2, metric, **kwargs):
    encoder1.eval()
    encoder2.eval()
    get_hidden_fn = get_all_hidden_transformer if isinstance(encoder1, Transformer) else get_all_hidden
    all_hidden1, all_hidden2 = get_hidden_fn(x1, encoder1, x2, encoder2)
    scores = np.zeros((len(all_hidden1), len(all_hidden2)))
    for idx1, hidden1 in all_hidden1.items():
        for idx2, hidden2 in all_hidden2.items():
            # Index 0 corresponds to layer 1
            if isinstance(encoder1, Transformer):
                if encoder1.use_cls_token:
                    final_hidden1 = hidden1[0]
                else:
                    # Average along sequence dimension
                    final_hidden1 = torch.mean(hidden1, dim=0)
            else:
                final_hidden1 = hidden1
            if isinstance(encoder2, Transformer):
                if encoder2.use_cls_token:
                    final_hidden2 = hidden2[0]
                else:
                    # Average along sequence dimension
                    final_hidden2 = torch.mean(hidden2, dim=0)
            else:
                final_hidden2 = hidden2
            scores[int(idx1) - 1][int(idx2) - 1] = get_alignment(final_hidden1, final_hidden2, metric, **kwargs)
    return scores