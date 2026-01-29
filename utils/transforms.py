#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch_sparse import coalesce
from utils.helpers import compute_bond_angles, compute_periodic_complete_features


class GetY(object):
    """Select target for prediction (multi-task aware)."""
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            if data.y.dim() == 0:
                data.y = data.y
            elif data.y.dim() == 1 and len(data.y) == 1:
                data.y = data.y[self.index] if self.index < len(data.y) else data.y[0]
            elif data.y.dim() == 1 and len(data.y) > 1:
                pass
            else:
                try:
                    data.y = data.y[0][self.index]
                except (IndexError, TypeError):
                    pass
        return data


class GetMultiTaskY(object):
    """Multi-task target transform that preserves all targets."""
    def __call__(self, data):
        return data


class GetAngle(object):
    """Compute bond angles."""
    def __call__(self, data):
        angles, idx_kj, idx_ji = compute_bond_angles(
            data.pos, data.cell_offsets, data.edge_index, data.num_nodes
        )
        data.angle_index  = torch.stack([idx_kj, idx_ji], dim=0)
        data.angle_attr   = angles.reshape(-1, 1)
        return data


class ToFloat(object):
    """Convert graph features to float32."""
    def __call__(self, data):
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        if hasattr(data, 'angle_attr'):
            data.angle_attr = data.angle_attr.float()

        if hasattr(data, 'dist'):
            data.dist = data.dist.float()
        if hasattr(data, 'theta'):
            data.theta = data.theta.float()
        if hasattr(data, 'phi'):
            data.phi = data.phi.float()
        if hasattr(data, 'tau'):
            data.tau = data.tau.float()
            
        return data


class AddPeriodicCompleteFeatures(object):
    """Compute and attach periodic complete features (d, θ, φ, τ)."""
    def __init__(self, cutoff=4.0):
        self.cutoff = cutoff

    def __call__(self, data):
        try:
            if not hasattr(data, 'pos') or not hasattr(data, 'cell') or not hasattr(data, 'edge_index'):
                self._add_empty_features(data)
                return data

            tuple_edge_index, dist, theta, phi, tau = compute_periodic_complete_features(
                data.pos, data.cell, data.edge_index, data.num_nodes, self.cutoff
            )
            
            if dist.numel() == 0:
                self._add_empty_features(data)
                return data
            
            data.tuple_edge_index = tuple_edge_index
            data.dist = dist
            data.theta = theta
            data.phi = phi
            data.tau = tau
            
        except Exception as e:
            self._add_empty_features(data)
            
        return data
    
    def _add_empty_features(self, data):
        """Attach empty Global features to keep schema consistent."""
        device = data.pos.device if hasattr(data, 'pos') else 'cpu'
        
        if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
            num_edges = data.edge_index.shape[1]
            data.tuple_edge_index = data.edge_index.clone()
            data.dist = torch.zeros(num_edges, dtype=torch.float, device=device)
            data.theta = torch.zeros(num_edges, dtype=torch.float, device=device)
            data.phi = torch.zeros(num_edges, dtype=torch.float, device=device)
            data.tau = torch.zeros(num_edges, dtype=torch.float, device=device)
        else:
            data.tuple_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            data.dist = torch.empty(0, dtype=torch.float, device=device)
            data.theta = torch.empty(0, dtype=torch.float, device=device)
            data.phi = torch.empty(0, dtype=torch.float, device=device)
            data.tau = torch.empty(0, dtype=torch.float, device=device)
