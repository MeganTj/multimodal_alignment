import os
import sys
sys.path.append(os.getcwd())
import argparse
import copy
import torch
from utils.utils import seed_everything
from training_structures.unimodal_supervised import test as test_uni
from training_structures.unimodal import train_unimodal

def train_eval_supervised(args, traindata, validdata, testdata, save_format, model_config, syn_depth, depth, 
                         epochs, lr, wd, modalnum, recon_modalnum=None):
    # Update arguments with model config
    unimodal_args = copy.deepcopy(args)
    args_dict = vars(unimodal_args)
    args_dict.update(model_config)
    unimodal_args = argparse.Namespace(**args_dict)
    unimodal_args.modality = modalnum
    unimodal_args.recon_modality = recon_modalnum
    unimodal_args.epochs = epochs
    unimodal_args.lr = lr
    unimodal_args.wd = wd
    unimodal_args.saved_model = save_format.format(syn_depth, depth, modalnum, recon_modalnum)
    print(f"Saving model to {unimodal_args.saved_model}")
    print(f"Training with {unimodal_args.epochs} epochs, lr={unimodal_args.lr}, wd={unimodal_args.wd }")
    if args.mode == "width":
        # TODO; Maybe just increase width linearly? 
        unimodal_args.hidden_dim = unimodal_args.hidden_dim * depth ** 2
    else:
        unimodal_args.num_hidden = depth

    # Load saved model if exists
    final_encoder_path = unimodal_args.saved_model + "_final_encoder.pt"
    final_head_path = unimodal_args.saved_model + "_final_head.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.use_saved_model and os.path.exists(final_encoder_path):
        encoder = torch.load(final_encoder_path, map_location=torch.device('cpu')).to(device)
        head = torch.load(final_head_path, map_location=torch.device('cpu')).to(device)
    else:
        if args.same_seed:
            seed_everything(args.base_seed)
        encoder, head = train_unimodal(unimodal_args, traindata, validdata, device)
        os.remove( unimodal_args.saved_model + "_encoder.pt")
        os.remove( unimodal_args.saved_model + "_head.pt")
    # Save final model
    torch.save(encoder, final_encoder_path)
    torch.save(head, final_head_path)
    all_perf = evaluate_model(encoder, head, traindata, validdata, testdata, unimodal_args.modality, 
                        unimodal_args.eval_task, recon_modalnum=unimodal_args.recon_modality)
    return encoder, head, unimodal_args.saved_model, all_perf

def evaluate_model(encoder, head, traindata, validdata, testdata, modalnum, eval_task, recon_modalnum=None):
    print("Training:")
    train_perf = test_uni(encoder, head, traindata, no_robust=True, auprc=False, modalnum=modalnum, 
                        recon_modalnum=recon_modalnum, task=eval_task)
    if isinstance(train_perf, dict):
        train_perf = train_perf["MSE"]
    print("Validation:")
    val_perf = test_uni(encoder, head, validdata, no_robust=True, auprc=False, modalnum=modalnum, 
                        recon_modalnum=recon_modalnum, task=eval_task)
    if isinstance(val_perf, dict):
        val_perf = val_perf["MSE"]
    print("Testing:")
    test_perf = test_uni(encoder, head, testdata, no_robust=True, auprc=False, modalnum=modalnum, 
                        recon_modalnum=recon_modalnum, task=eval_task)
    if isinstance(test_perf, dict):
        test_perf = test_perf["MSE"]
    return train_perf, val_perf, test_perf