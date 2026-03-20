import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
from model import *
from act_model import KcatModel as _KcatModel
from act_model import KmModel as _KmModel
from act_model import ActivityModel
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import os, sys
from pathlib import Path

class EnzymeDatasets(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)

def get_datasets(sequences, smiles, ProtT5_model, MolT5_model):
    
    seq_ProtT5 = Seq_to_vec(sequences, ProtT5_model)
    smi_molT5 = get_molT5_embed(smiles, MolT5_model)
    smi_macc = GetMACCSKeys(smiles)
    
    feats = th.from_numpy(np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)).to(th.float32)
    datasets = EnzymeDatasets(feats)
    dataloader = DataLoader(datasets, batch_size=32, shuffle=False)
    
    return dataloader

def inference(model, dataloader, task_type, device="cuda:0"):
    model.eval()
    with th.no_grad():
        pred_list = []
        for step, data in enumerate(dataloader):
            data = data.to(device)
            ezy_feats = data[:, :1024]
            sbt_feats = data[:, 1024:]

            if task_type == "KCAT/KM":
                pred = model(ezy_feats, sbt_feats)[-1].cpu().numpy()
            else:
                pred = model(ezy_feats, sbt_feats).cpu().numpy()
            
            pred_list.append(pred.ravel())
    
        return np.concatenate(pred_list, axis=0)
    
def main(input_path, output_path, task_type, device="cuda:0"):
    
    if task_type == "KCAT":
        model_dpath = f"{MODEL_DPATH}/kcat_models"
    elif task_type == "KM":
        model_dpath = f"{MODEL_DPATH}/Km_models"
    elif task_type == "KCAT/KM":
        model_dpath = f"{MODEL_DPATH}/act_models"

    df = pd.read_csv(input_path)
    sequences = df["Protein Sequence"].tolist()
    smiles = df["Substrate SMILES"].tolist()

    # Load SMILES model once
    print("Loading SMILES model...")

    dataloader = get_datasets(sequences, smiles, ProtT5_model, MolT5_model)
    prediction_list = []
    for fold in range(10):
        if task_type == "KCAT":    
            model = KcatModel(device=device).to(device)
            model.load_state_dict(th.load(f"{model_dpath}/{fold}_bestmodel.pth", map_location=device))
        elif task_type == "KM":
            model = KmModel(device=device).to(device)
            model.load_state_dict(th.load(f"{model_dpath}/{fold}_bestmodel.pth", map_location=device))
        elif task_type == "KCAT/KM":
            model = ActivityModel(device=device).to(device)
            model.load_state_dict(th.load(f"{model_dpath}/{fold}_bestmodel.pth", map_location=device))
    
        fold_pred = inference(model, dataloader, task_type, device=device)
        prediction_list.append(fold_pred.reshape(-1, 1))
    
    predictions = np.mean(np.concatenate(prediction_list, axis=-1), axis=-1)

    # Output - same format as original
    df_out = pd.DataFrame({"Predicted Value": predictions})
    df_out.to_csv(output_path, index=False)

            
if __name__ == "__main__":

    parent_dir = Path(__file__).resolve().parent.parent
    MODEL_DPATH = f"{parent_dir}/models"
    
    ProtT5_model = f"{MODEL_DPATH}/prot_t5_xl_uniref50/"
    MolT5_model = f"{MODEL_DPATH}/molt5-base-smiles2caption"

    if len(sys.argv) != 4:
        print("Usage: python run_unikp.py <input_csv> <output_csv> <task_type>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    task = sys.argv[3].upper()  # KCAT, KM, or KCAT/KM
    main(input_csv, output_csv, task)