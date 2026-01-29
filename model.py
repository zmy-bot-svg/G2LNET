import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, LayerNorm, ReLU, Sigmoid
from torch_geometric.nn import global_mean_pool, DiffGroupNorm
from torch_geometric.data import Data
from typing import Optional
from baseModule import BaseModule, GGCN
from utils.features import torsion_emb, angle_emb

class TwoLayerLinear(torch.nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bias=False, act=False):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = F.silu(x)
        x = self.lin2(x)
        if self.act:
            x = F.silu(x)
        return x

class RBFExpansion(torch.nn.Module):
    """Radial Basis Function expansion for continuous features."""
    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.centers = torch.linspace(vmin, vmax, bins)
        self.type = type

        if lengthscale is None:
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        base = self.gamma * (distance - self.centers.to(distance.device))
        switcher = {
            'gaussian': (-base ** 2).exp(),
        }
        return switcher.get(self.type, (-base ** 2).exp())

class EmbeddingLayer(torch.nn.Module):
    """Initial feature embedding with GroupNorm."""
    def __init__(self, input_features, output_features):
        super().__init__()
        self.mlp = Sequential(
            Linear(input_features, output_features),
            DiffGroupNorm(output_features, 6, track_running_stats=True),
            torch.nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class G2LNetUpdate(torch.nn.Module):
    """
    G2LNet Update Layer with Dual-Stream Fusion (Local + Periodic).
    Includes Mixture-of-Experts (MoE) gating mechanism.
    """
    def __init__(self, hidden_features, dropout_rate, global_fusion_alpha=0.5):
        super().__init__()
        
        # Local stream updates using Graph Convolution (GGCN)
        self.bondAndAngleUpdate = GGCN(dim=hidden_features, dropout_rate=dropout_rate)
        self.bondAndAtomUpdate = GGCN(dim=hidden_features, dropout_rate=dropout_rate)
        
        # Adaptive Fusion Gate (Soft Router for MoE)
        self.fusion_gate = torch.nn.Sequential(
            Linear(hidden_features * 2, hidden_features),
            LayerNorm(hidden_features),
            ReLU(),
            Linear(hidden_features, hidden_features),
            Sigmoid()
        )

    def forward(
        self,
        g: Data,
        atom_feats: torch.Tensor,
        bond_attr: torch.Tensor,
        triplet_feats: torch.Tensor,
        h_periodic_complete: Optional[torch.Tensor] = None
    ):
        # 1. Local Stream: Edge & Angle Update
        bond_updated_local, triplet_feats_updated = self.bondAndAngleUpdate(
            bond_attr, g.angle_index, triplet_feats
        )

        # 2. Local Stream: Atom Update
        atom_feats_updated_local, bond_attr_updated_local = self.bondAndAtomUpdate(
            atom_feats, g.edge_index, bond_updated_local
        )

        # 3. Dual-Stream Fusion (if Periodic features exist)
        if h_periodic_complete is not None:
            if (not hasattr(g, 'tuple_edge_index')) or g.tuple_edge_index is None or g.tuple_edge_index.numel() == 0:
                raise RuntimeError("Missing 'tuple_edge_index' for Global Expert stream!")

            atom_feats_updated_periodic, _ = self.bondAndAtomUpdate(
                atom_feats, g.tuple_edge_index, h_periodic_complete
            )

            g2lnet_norm = F.layer_norm(atom_feats_updated_local, atom_feats_updated_local.shape[-1:])
            global_norm = F.layer_norm(atom_feats_updated_periodic, atom_feats_updated_periodic.shape[-1:])

            gate_input = torch.cat([g2lnet_norm, global_norm], dim=-1)
            z = self.fusion_gate(gate_input)
            final_atom_feats = z * g2lnet_norm + (1 - z) * global_norm
        else:
            final_atom_feats = atom_feats_updated_local

        return final_atom_feats, bond_attr_updated_local, triplet_feats_updated

class G2LNet(BaseModule):
    def __init__(self, 
                 data: Data, 
                 firstUpdateLayers: int=4,
                 secondUpdateLayers: int=4,
                 atom_input_features: int=92,
                 edge_input_features: int=50,
                 triplet_input_features: int=40,
                 embedding_features: int=64,
                 hidden_features: int=256,
                 num_tasks: int=1,
                 min_edge_distance: float=0.0,
                 max_edge_distance: float=8.0,
                 min_angle: float=0.0,
                 max_angle: float=6.28,
                 dropout_rate=0.0,
                 use_global_context=True,
                 global_fusion_alpha=0.5,
                 gradnorm_alpha=1.5
                ): 
        super().__init__()
        self.use_global_context = use_global_context
        self.num_tasks = num_tasks
        self.firstUpdateLayers = firstUpdateLayers
        self.secondUpdateLayers = secondUpdateLayers  
        self.hidden_features = hidden_features

        # Embeddings
        self.atom_embedding = EmbeddingLayer(atom_input_features, hidden_features)
        self.edge_embedding = Sequential(
            RBFExpansion(vmin=min_edge_distance, vmax=max_edge_distance, bins=edge_input_features),
            EmbeddingLayer(edge_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )
        self.angle_embedding = Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=triplet_input_features),
            EmbeddingLayer(triplet_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        if self.use_global_context:
            self.global_torsion_embedding = torsion_emb(num_radial=3, num_spherical=2, cutoff=4.0)
            self.global_angle_embedding = angle_emb(num_radial=3, num_spherical=2, cutoff=4.0)
            self.lin_global_torsion = TwoLayerLinear(12, hidden_features, hidden_features)
            self.lin_global_angle = TwoLayerLinear(6, hidden_features, hidden_features)

        # Update Layers
        total_layers = firstUpdateLayers + secondUpdateLayers
        self.blocks = torch.nn.ModuleList(
            [G2LNetUpdate(hidden_features, dropout_rate, global_fusion_alpha) for _ in range(total_layers)]
        )

        # Output Heads
        self.output_heads = torch.nn.ModuleList(
            [Linear(hidden_features, 1) for _ in range(num_tasks)]
        )

        # Dense Connection Projection (DCN)
        num_dense_layers = 1 + total_layers
        total_dense_dim = self.hidden_features * num_dense_layers
        self.dense_projection = Sequential(
            torch.nn.Linear(total_dense_dim, self.hidden_features),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate * 0.5)
        )

        self.gradnorm_alpha = gradnorm_alpha
        self.loss_weights = torch.nn.Parameter(torch.ones(num_tasks).float())
 
    @property
    def target_attr(self):
        return "y"

    def forward(self, g: Data, return_latent: bool = False) -> torch.Tensor:
        atom_feats = self.atom_embedding(g.x)
        bond_attr = self.edge_embedding(g.edge_attr)
        triplet_feats = self.angle_embedding(g.angle_attr)
        
        # NaN Guard
        if torch.isnan(atom_feats).any() or torch.isnan(bond_attr).any() or torch.isnan(triplet_feats).any():
            atom_feats = torch.nan_to_num(atom_feats, nan=0.0)
            bond_attr = torch.nan_to_num(bond_attr, nan=0.0)
            triplet_feats = torch.nan_to_num(triplet_feats, nan=0.0)

        atom_feats_history = [atom_feats]

        h_periodic_complete = None
        if self.use_global_context:
            if (not hasattr(g, 'dist') or not hasattr(g, 'theta') or 
                not hasattr(g, 'phi') or not hasattr(g, 'tau') or g.dist.numel() == 0):
                raise RuntimeError("Missing global context inputs (dist/theta/phi/tau) while use_global_context=True.")

            tbf_emb = self.global_torsion_embedding(g.dist, g.phi, g.tau)
            sbf_emb = self.global_angle_embedding(g.dist, g.theta)
            h_torsion = self.lin_global_torsion(tbf_emb)
            h_angle = self.lin_global_angle(sbf_emb)
            h_periodic_complete = torch.clamp(h_torsion + h_angle, min=-10.0, max=10.0)

        for block in self.blocks:
            atom_feats, bond_attr, triplet_feats = block(
                g, atom_feats, bond_attr, triplet_feats, h_periodic_complete
            )
            atom_feats_history.append(atom_feats)
        
        # Dense Connection Fusion
        dense_atom_feats = torch.cat(atom_feats_history, dim=-1)
        projected_feats = self.dense_projection(dense_atom_feats)
        
        # Readout
        crys_fea = global_mean_pool(projected_feats, g.batch)

        if return_latent:
            return crys_fea
        
        # Multi-Head Output
        task_outputs = [head(crys_fea) for head in self.output_heads]
        final_output = torch.cat(task_outputs, dim=-1)

        return final_output

    def get_last_shared_layer(self):
        return self.blocks[-1].bondAndAtomUpdate.lin_s

    def compute_uncertainty(self, g: Data) -> torch.Tensor:
        self.eval()

        with torch.enable_grad():
            h = self.forward(g, return_latent=True)
            h = h.detach().clone().requires_grad_(True)

            uncertainties = []
            num_heads = len(self.output_heads)
            for idx, head in enumerate(self.output_heads):
                y_pred = head(h)
                grads = torch.autograd.grad(
                    y_pred,
                    h,
                    torch.ones_like(y_pred),
                    create_graph=False,
                    retain_graph=idx < num_heads - 1,
                )[0]
                s_crystal = torch.sum(grads ** 2, dim=1)
                u_geo = torch.log10(s_crystal + 1e-10)
                uncertainties.append(u_geo)

        return torch.stack(uncertainties, dim=1)
    
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)