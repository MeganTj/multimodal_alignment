import torch
import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
from unimodals.common_models import CustomMLP, Transformer, VisionTransformer
from training_structures.unimodal_supervised import train

def get_model_criterion(args, task, num_classes, device):
    # Specify unimodal model
    final_hidden_dim = args.final_hidden_dim if hasattr(args, "final_hidden_dim") else None
    if args.model_type == "mlp":
        encoder = CustomMLP(args.input_dim, args.hidden_dim, num_hidden=args.num_hidden,
                            nonlin=args.nonlin).to(device)
    elif args.model_type == "vit":
        patch_size = args.patch_size if hasattr(args, "patch_size") else None
        encoder = VisionTransformer(args.n_features, max_seq_length=args.max_seq_length, 
                                    embed_dim=args.hidden_dim, use_cls_token=args.use_cls_token, patch_size=patch_size, n_layers=args.num_hidden, 
                                    nhead=args.nhead, norm_first=args.norm_first, final_hidden_dim=final_hidden_dim).to(device)
    elif args.model_type == "transformer":
        encoder = Transformer(args.n_features, max_seq_length=args.max_seq_length, 
                              embed_dim=args.hidden_dim, use_cls_token=args.use_cls_token, n_layers=args.num_hidden, nhead=args.nhead, 
                              norm_first=args.norm_first, tokenize=args.tokenize, final_hidden_dim=final_hidden_dim).to(device)
    else:
        raise NotImplementedError
    
    if task == "classification":
        out_dim = num_classes
        criterion = nn.CrossEntropyLoss()
    elif task == "regression":
        out_dim = 1
        criterion = nn.L1Loss()
    elif task == "reconstruction": 
        out_dim = args.input_dim
        criterion = nn.MSELoss()
    head_indim = args.hidden_dim if final_hidden_dim is None else final_hidden_dim
    head = nn.Linear(head_indim, out_dim).to(device)
    return encoder, head, criterion

def train_unimodal(args, traindata, validdata, device):
    # train
    encoder, head, criterion = get_model_criterion(args, args.task, args.num_classes, device)
    saved_encoder = args.saved_model + '_encoder.pt'
    saved_head = args.saved_model + '_head.pt'
    train(encoder, head, traindata, validdata, args.epochs, lr=args.lr, weight_decay=args.wd,
                criterion=criterion, task=args.task, auprc=False, modalnum=args.modality, recon_modalnum=args.recon_modality,
                save_encoder=saved_encoder, save_head=saved_head)
    encoder = torch.load(saved_encoder).to(device)
    head = torch.load(saved_head).to(device)
    return encoder, head

def load_encoder_head(saved_model, device):
    encoder = torch.load(saved_model + "_encoder.pt").to(device)
    head = torch.load(saved_model + "_head.pt").to(device)
    return encoder, head


