from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import yaml
import logging
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
import os

# from torch_geometric.loader import NeighborLoader
from datasets_module import (
    APPNPEmbDataset,
    EdgeWeightCollator,
    EmbDataset,
    GATFusionEmbDataset,
    MixedFusionDepthDataset,
)
from graph_aware_sid.graph_utils import get_graph_building_conf
from models.rqvae import GRQVAE, RQVAE
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from trainer import Trainer
from dataclasses import dataclass
from torch.utils.data.distributed import DistributedSampler
from utils import yaml_to_argv



def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducible RQ-VAE training')
    parser.add_argument('--graph_builder_dump_path ', type=str, default=None, help='random seed for reproducible RQ-VAE training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, )
    parser.add_argument('--persistent_workers', action="store_true",)
    parser.add_argument('--compile', action="store_true",)
    parser.add_argument('--prefetch_factor', type=int, default=None, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="../data", help="Input data path.")
    parser.add_argument("--interactions_path", type=str, default=None, help="Input data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument("--disable_vq_init", action="store_true", help="run one-pass codebook initialization before training")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--disable_cf_loss", action="store_true")
    parser.add_argument("--disable_sk", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight (encoder-directed)')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb', type=str, default=None, help='cf emb; leave unset to disable CF loss')
    parser.add_argument('--cf_loss_use_codebook_rep', action='store_true', help='compute CF loss on reconstructed codebook representations instead of dense_out')
    parser.add_argument('--cf_loss_last_m_codebooks', type=int, default=0, help='if >0, compute CF loss using only the last M codebooks')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="otherwise time based")
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument("--graph_signal", action="store_true")
    graph_group.add_argument("--graph_encoder", action="store_true")
    graph_group.add_argument("--graph_augmentation", action="store_true")

    parser.add_argument("--convolution_type", type=str, default="APPNP", help="otherwise time based")

    parser.add_argument("--graph_type", type=str, default='adjacent_cooc_prod')
    # graph_type = parser.add_subparsers(dest="graph_type")

    # p_windowed = graph_type.add_parser("windowed")
    # p_windowed.add_argument("--window_size", type=int, default=3, help="Size of the sliding window")
    # p_windowed.add_argument("--bidirectional", action="store_true", help="Make graph bidirectional")
    # p_windowed.add_argument("--dedup_within_window", action="store_true", help="Deduplicate pairs within window")
    # p_windowed.add_argument("--decay", type=str, choices=["inverse_distance"], default=None, help='Edge weighting scheme or None')

    # p_rw = graph_type.add_parser("rw")
    # p_rw.add_argument("--walk_length", type=int, default=10, help="Length of each random walk")
    # p_rw.add_argument("--walks_per_item", type=int, default=10, help="Number of walks per item")
    # p_rw.add_argument("--window_size", type=int, default=2, help="Context window size for co-occurrence")
    # p_rw.add_argument("--self_loops", action="store_true", help="Allow self-loops in random walks")
    # p_rw.add_argument("--dedup_per_walk", action="store_true", help="Deduplicate pairs within each walk")
    # p_rw.add_argument("--seed", type=int, default=42, help="Random seed")

    # graph_type.add_parser("adjacent_cooc")
    # graph_type.add_parser("adjacent_cooc_prod")

    ### APPNP param block
    parser.add_argument("--num_prop", type=int, default=10, help="num propagations")
    parser.add_argument("--appnp_alpha", type=float, default=0.1, help="self-loop prob")
    parser.add_argument("--use_edge_weight", action="store_true", help="whether to use edge weights")
    parser.add_argument("--skip_n_last", type=int, default=None)

    # APPNP as augmentation block
    parser.add_argument('--hops', type=int, nargs='+', default=[0], help="hops")
    parser.add_argument('--probs', type=int, nargs='+', default=None, help="hops")

    # edge reconstruction block
    parser.add_argument("--plain_reconstruction", action="store_true", help="whether to use edge weights")
    parser.add_argument("--use_edge_reconstruction_loss", action="store_true", help="whether to use reconstruct edges data on quantizer level")
    parser.add_argument("--num_anchors", type=int, default=1024)
    parser.add_argument("--lognorm_weights", action="store_true", help="whether to use log of weights")



    # spectral clustering block
    parser.add_argument("--spectral_dim", type=int, default=0, help="num spectral clusters, currently either 0 or inner quantizer dim")
    parser.add_argument("--cluster_labels_path", type=str, default=None)
    parser.add_argument("--alpha_graph_cluster", type=float, default=0.0, help="graph_cluster loss coef")
    parser.add_argument("--max_cluster_level", type=int, default=None, help="get cluster up to this level")

    config_args, remaining_args = parser.parse_known_args()
    yaml_argv = []
    if config_args.config:
        with open(config_args.config, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        config = config.pop("sid_train_conf", {})
        yaml_argv = yaml_to_argv(config)

    # Later arguments win, so real CLI arguments override YAML.
    args = parser.parse_args(yaml_argv + sys.argv[1:])
    # setup defaults
    # args.graph_type = args.graph_type or "adjacent_cooc_prod"

    return args

@dataclass
class DistTrainingState:
    device: str = "npu:0"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

def init_ddp_info_params(args: dict) -> Tuple[str, int, int, int]:
    """Mirror main.py distributed setup for torchrun-launched benchmark runs."""
    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_type = str(args.device.split(":", 1)[0])
    backend = "nccl" if device_type == "cuda" else "hccl"
    logging.info("tcp://%s:%s rank=%s", addr, port, rank_id)

    dist.init_process_group(
        backend,
        init_method=f"tcp://{addr}:{port}",
        rank=rank_id,
        world_size=world_size,
    )

    return f"{device_type}:{local_rank}", rank_id, local_rank, world_size


def init_dist_state(args: dict) -> DistTrainingState:
    return DistTrainingState(*init_ddp_info_params(args))


def wrap_model_for_ddp(
    model: torch.nn.Module, dist_state: DistTrainingState, ddp_on: bool
) -> torch.nn.Module:
    if not ddp_on:
        return model
    if dist_state.device.startswith(("cuda", "npu")):
        return DDP(model, device_ids=[dist_state.local_rank], broadcast_buffers=False)
    return DDP(model, broadcast_buffers=False)


if __name__ == '__main__':
    """fix the random seed"""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    print(args)
    logging.basicConfig(level=logging.DEBUG)
    cf_emb = None
    if args.cf_emb:
        cf_emb = torch.load(args.cf_emb).squeeze().detach().cpu().numpy()

    """build dataset"""
    if args.graph_augmentation:
        graph_building_kwargs = get_graph_building_conf(args)
        graph_saving_path = os.path.join(args.ckpt_dir, "multihop/")

        dataset = MixedFusionDepthDataset(
            args.interactions_path, 
            args.data_path, 
            hops=args.hops,
            fused_cache_root=graph_saving_path,
            alpha=args.appnp_alpha,
            use_edge_weight=args.use_edge_weight,
            graph_type=args.graph_type,
            convolution_type=args.convolution_type,
            graph_building_kwargs=graph_building_kwargs,
            probs=args.probs,
        )
    elif args.graph_signal:
        graph_building_kwargs = get_graph_building_conf(args)
        graph_saving_path = os.path.join(args.ckpt_dir, "graph.npy")
        pyg_saving_path = os.path.join(args.ckpt_dir, "pyg_data.npy")
        logging.info(f"pyg_saving_path: {pyg_saving_path}")
        dataset = APPNPEmbDataset(
            args.interactions_path, 
            args.data_path, 
            fused_cache_path=graph_saving_path,
            pyg_cache_path=pyg_saving_path,
            num_prop=args.num_prop,
            alpha=args.appnp_alpha,
            use_edge_weight=args.use_edge_weight,
            graph_type=args.graph_type,
            convolution_type=args.convolution_type,
            graph_building_kwargs=graph_building_kwargs,
            spectral_dim=args.spectral_dim,
            eval_mode=False,
            cluster_labels_path=args.cluster_labels_path,
            skip_n_last=args.skip_n_last
        )
    else:
        # dataset = EmbDataset(args.data_path, interactions_path=args.interactions_path, eval_mode=False)
        dataset = EmbDataset(args.data_path, interactions_path=None, eval_mode=False)

    
    if args.graph_encoder:
        # NOTE: this dims should be taken from files
        model = GRQVAE(
            semantic_dim=4096, gat_hidden_dim=4096, gat_dropout=0.0,
            vae_in_dim=dataset.dim,
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
            beta = args.beta,
            alpha = args.alpha,
            n_clusters= args.n_clusters,
            sample_strategy =args.sample_strategy,
            cf_embedding = cf_emb,
            use_cf_loss= not args.disable_cf_loss
        )
    else:
        model = RQVAE(in_dim=dataset.dim,
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
                    beta = args.beta,
                    alpha = args.alpha,
                    n_clusters= args.n_clusters,
                    sample_strategy =args.sample_strategy,
                    cf_embedding = cf_emb,
                    cf_loss_use_codebook_rep=args.cf_loss_use_codebook_rep,
                    cf_loss_last_m_codebooks=(args.cf_loss_last_m_codebooks if args.cf_loss_last_m_codebooks > 0 else None),
                    use_cf_loss= not args.disable_cf_loss,
                    alpha_graph_cluster=args.alpha_graph_cluster,
                    num_anchors=args.num_anchors,
                    seed=args.seed,
                    lognorm_weights=args.lognorm_weights
                    )
    print(model)
    # dist_state = init_dist_state(args)
    # model.to(dist_state.device)
    model.to(args.device)
    # model = DDP(
    #     model, 
    #     device_ids=[dist_state.local_rank], 
    #     broadcast_buffers=True, 
    #     find_unused_parameters=True, 
    # )

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    if args.graph_encoder:
        graph_dataset = GATFusionEmbDataset(args.interactions_path, args.data_path, num_hops=3)
        data_loader = NeighborLoader(
            graph_dataset.graph,
            num_neighbors=[30] * 2,
            batch_size=args.batch_size,
            shuffle=True
        )
    else:
        if args.use_edge_reconstruction_loss:
            collate_fn = EdgeWeightCollator(dataset.pyg_data, num_anchors=args.num_anchors)
            # collate_fn = default_collate
        else:
            collate_fn = default_collate
        # sampler = DistributedSampler(dataset, num_replicas=dist_state.world_size, rank=dist_state.rank, shuffle=True)
        data_loader = DataLoader(dataset,num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                batch_size=args.batch_size, shuffle=True,
                                pin_memory=True, persistent_workers=args.persistent_workers,
                                collate_fn=collate_fn, generator=data_generator, drop_last=True)#, sampler=sampler)

    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(dataset, data_loader, sampler=None)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)
