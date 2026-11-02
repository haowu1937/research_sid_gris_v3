import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        if residual:
            self.skip = nn.Linear(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, batch):
        # batch.x: [N_sub, in_dim]
        # batch.edge_index: [2, E_sub]
        # batch.batch_size: number of seed nodes

        x0 = self.input_proj(batch.x)

        x = self.gat1(x0, batch.edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, batch.edge_index)
        x = self.norm2(x)

        if self.residual:
            x = x + self.skip(x0)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # only return embeddings for seed nodes
        return x[:batch.batch_size]


class GATFusionEncoder(nn.Module):
    """
    Simple GAT encoder that fuses:
      1. semantic node representations
      2. graph neighborhood information from a sampled subgraph

    Expected inputs:
        node_ids:   LongTensor [B]
            Global ids of center nodes we want embeddings for.

        n_id:       LongTensor [N_sub]
            Global ids of all nodes participating in the sampled subgraph.
            Usually returned by a PyG sampler.

        edge_index: LongTensor [2, E]
            Subgraph edges in LOCAL indexing over n_id.

        semantic_x: FloatTensor [N_sub, D_sem]
            Semantic embeddings aligned with n_id.
            semantic_x[i] corresponds to global node n_id[i].

    Returns:
        fused_center: FloatTensor [B, out_dim]
            Fused embeddings for the requested center nodes only.
    """

    def __init__(
        self,
        semantic_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        use_residual_fusion: bool = True,
    ):
        super().__init__()

        self.use_residual_fusion = use_residual_fusion
        self.dropout = dropout

        # Project semantic representations before message passing
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # Two-layer GAT
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        # Optional residual branch from pure semantics
        if use_residual_fusion:
            self.semantic_skip = nn.Linear(hidden_dim, out_dim)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, graph_data) -> torch.Tensor:
        semantic_x = graph_data.x              # [N_sub, D_sem], sampled node features
        edge_index = graph_data.edge_index     # [2, E], local indexing over sampled nodes

        x0 = self.semantic_proj(semantic_x)    # [N_sub, hidden_dim]

        x = self.gat1(x0, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = self.norm2(x)

        if self.use_residual_fusion:
            x = x + self.semantic_skip(x0)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # In NeighborLoader, the first `batch_size` nodes are the seed/center nodes
        return x[: graph_data.batch_size]