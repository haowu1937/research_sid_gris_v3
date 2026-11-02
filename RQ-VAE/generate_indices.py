import argparse
import collections
import json
import os
import random
import sys
import polars as pl
import yaml

import numpy as np
import torch
from datasets_module import (
    APPNPEmbDataset,
    EmbDataset,
    GATFusionEmbDataset,
)
from graph_aware_sid.graph_utils import get_graph_building_conf
from models.rqvae import GRQVAE, RQVAE
from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from utils import constrained_km, yaml_to_argv


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# NOTE: why this is not shared with main.py? 
def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--save_parquet', action="store_true")
    parser.add_argument("--dataset", type=str,default="Instruments", help='dataset')
    parser.add_argument("--root_path", type=str,default="../checkpoint/", help='root path')
    parser.add_argument('--checkpoint', type=str, default='epoch_9999_collision_0.0012_model.pth', help='checkpoint name')
    parser.add_argument("--interactions_path", type=str, default="../data", help="Input data path.")
    parser.add_argument("--output_dir", type=str, default="../data", help="output path.")
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument("--graph_signal", action="store_true")
    graph_group.add_argument("--graph_encoder", action="store_true")
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducible SID generation')

    config_args, remaining_args = parser.parse_known_args()
    yaml_argv = []
    if config_args.config:
        with open(config_args.config, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        config = config.pop("sid_gen_conf", {})
        yaml_argv = yaml_to_argv(config)

    args = parser.parse_args(yaml_argv + sys.argv[1:])
    
    return args

args_setting = parse_args()

random.seed(args_setting.seed)
np.random.seed(args_setting.seed)
torch.manual_seed(args_setting.seed)
torch.cuda.manual_seed_all(args_setting.seed)

dataset = args_setting.dataset
ckpt_path = args_setting.root_path + args_setting.checkpoint
ckpt_id = args_setting.root_path.replace("..", "").replace("/", "_")

output_dir = f"{args_setting.output_dir}/{dataset}/"
os.makedirs(output_dir, exist_ok=True)
if args_setting.save_parquet:
    output_file = f"{dataset}.index.{ckpt_id}.parquet"
else:
    output_file = f"{dataset}.index.{ckpt_id}.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("npu:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]
if args.graph_signal:
    graph_saving_path = os.path.join(args.ckpt_dir, "graph.npy")
    pyg_saving_path = os.path.join(args.ckpt_dir, "pyg_data.npy")
    graph_building_kwargs = get_graph_building_conf(args)
    
    data = APPNPEmbDataset(
        args.interactions_path,
        args.data_path,
        fused_cache_path=graph_saving_path,
        pyg_cache_path=pyg_saving_path,
        eval_mode=True,
        skip_n_last=args.skip_n_last
    )
else:
    data = EmbDataset(args.data_path, eval_mode=True)

if args_setting.graph_encoder:
    # NOTE: this dims should be taken from files
    model = GRQVAE(
        semantic_dim=4096, gat_hidden_dim=4096, gat_dropout=0.0,
        vae_in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
else:
    model = RQVAE(in_dim=data.dim,
                num_emb_list=args.num_emb_list,
                e_dim=args.e_dim,
                layers=args.layers,
                dropout_prob=args.dropout_prob,
                bn=args.bn,
                loss_type=args.loss_type,
                quant_loss_weight=args.quant_loss_weight,
                kmeans_init=args.kmeans_init,
                kmeans_iters=args.kmeans_iters,
                sk_epsilons=args.sk_epsilons,
                sk_iters=args.sk_iters,
                )

model.load_state_dict(state_dict,strict=False)
model = model.to(device)
model.eval()
print(model)


if args_setting.graph_encoder:
    graph_dataset = GATFusionEmbDataset(args.interactions_path, args.data_path, num_hops=3)
    data_loader = NeighborLoader(
        graph_dataset.graph,
        num_neighbors=[30] * 2,
        batch_size=64,
        shuffle=True
    )
else:
    data_loader = DataLoader(data,num_workers=args.num_workers,
                            batch_size=64, shuffle=False,
                            pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]

labels = {"0":[],"1":[],"2":[], "3":[]}
embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in model.rq.vq_layers]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label
for _data in tqdm(data_loader, disable=not sys.stderr.isatty()):
    if args_setting.graph_encoder:
        emb_idx = _data.n_id[:_data.batch_size]
        graph_data = _data.to(device)
        d = _data.x[:_data.batch_size]
    else:
        d, emb_idx = _data[0], _data[1]
        d = d.to(device)
    
    if args_setting.graph_encoder:
        indices = model.get_indices(data, graph_data, labels, use_sk=False)
    else:
        indices = model.get_indices(d, labels,use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(index)

        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
tt = 0


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

if args_setting.save_parquet:
    sid_df = pl.DataFrame({
        "normalized_id": list(all_indices_dict.keys()),
        "semantic_id": [list(values) for values in all_indices_dict.values()],
    }).with_columns(
        pl.col("normalized_id").cast(pl.Int64)
    )

    sid_df = sid_df.with_columns(
        pl.col("semantic_id").list.eval(pl.element() + 1)
    )
    emb_norm_id = pl.read_parquet(args.data_path)
    result = emb_norm_id.join(sid_df, on="normalized_id", how="inner").select('item_id', 'normalized_id', 'semantic_id', 'raw_id')
    result.write_parquet(output_file)
else:
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict,fp)
