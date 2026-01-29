import contextlib
import itertools
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch_geometric.data.data import Data
from torch_geometric.utils import degree
from torch_sparse import SparseTensor



def load_node_representation(node_representation="onehot"):
    node_rep_path = Path(__file__).parent
    default_reps = {"onehot": str(node_rep_path / "./node_representations/atom_init.json")}

    rep_file_path = node_representation
    if node_representation in default_reps:
        rep_file_path = default_reps[node_representation]

    file_type = rep_file_path.split(".")[-1]
    loaded_rep = None

    if file_type == "csv":
        loaded_rep = np.genfromtxt(rep_file_path, delimiter=",")
        loaded_rep = loaded_rep.astype(int)

    elif file_type == "json":
        import json
        with open(rep_file_path) as f:
            atom_dictionary = json.load(f) 
        loaded_rep = np.array(list(atom_dictionary.values()))

    return loaded_rep

def one_hot_degree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)               
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else: 
        data.x = deg

    return data

def generate_node_features(input_data, n_neighbors, device):
    node_reps = load_node_representation()
    node_reps = torch.from_numpy(node_reps).to(device)
    n_elements, n_features = node_reps.shape

    if isinstance(input_data, Data):
        original_features = node_reps[input_data.z - 1].view(-1, n_features)
        if hasattr(input_data, 'agni_features'):
            agni_features = input_data.agni_features.to(device)
            input_data.x = torch.cat([original_features, agni_features], dim=-1)
            delattr(input_data, 'agni_features')
            logging.debug(f"✅ LOPE integration: Combined features shape: {input_data.x.shape}")
        else:
            input_data.x = original_features
            logging.debug("⚠️  No AGNI features found, using original features only")
            
        return one_hot_degree(input_data, n_neighbors)

    for i, data in enumerate(input_data):
        original_features = node_reps[data.z - 1].view(-1, n_features)

        if hasattr(data, 'agni_features'):
            agni_features = data.agni_features.to(device)
            data.x = torch.cat([original_features, agni_features], dim=-1)
            delattr(data, 'agni_features')
        else:
            data.x = original_features

    for i, data in enumerate(input_data):
        input_data[i] = one_hot_degree(data, n_neighbors)


def get_ranges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max

def normalize_edge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = get_ranges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)

def get_distances(
    positions: torch.Tensor,
    offsets: torch.Tensor,
    device: str = "cpu",
    mic: bool = True,
):
    """
    Get pairwise atomic distances

    Parameters
        positions:  torch.Tensor
                    positions of atoms in a unit cell

        offsets:    torch.Tensor
                    offsets for the unit cell

        device:     str
                    torch device type

        mic:        bool
                    minimum image convention
    """

    # convert numpy array to torch tensors
    n_atoms = len(positions)
    n_cells = len(offsets)

    pos1 = positions.view(-1, 1, 1, 3).expand(-1, n_atoms, n_cells, 3)
    pos2 = positions.view(1, -1, 1, 3).expand(n_atoms, -1, n_cells, 3)
    offsets = offsets.view(-1, n_cells, 3).expand(pos2.shape[0], n_cells, 3)
    pos2 = pos2 + offsets

    # calculate pairwise distances
    atomic_distances = torch.linalg.norm(pos1 - pos2, dim=-1)

    # get minimum
    min_atomic_distances, min_indices = torch.min(atomic_distances, dim=-1)
    expanded_min_indices = min_indices.clone().detach()

    atom_rij = pos1 - pos2
    expanded_min_indices = expanded_min_indices[..., None, None].expand(
        -1, -1, 1, atom_rij.size(3)
    )
    atom_rij = torch.gather(atom_rij, dim=2, index=expanded_min_indices).squeeze()

    return min_atomic_distances, min_indices

def get_pbc_cells(cell: torch.Tensor, offset_number: int, device: str = "cpu"):
    """
    Get the periodic boundary condition (PBC) offsets for a unit cell

    Parameters
        cell:       torch.Tensor
                    unit cell vectors of ase.cell.Cell

        offset_number:  int
                    the number of offsets for the unit cell
                    if == 0: no PBC
                    if == 1: 27-cell offsets (3x3x3)
    """

    _range = np.arange(-offset_number, offset_number + 1)
    offsets = [list(x) for x in itertools.product(_range, _range, _range)]
    offsets = torch.tensor(offsets, device=device, dtype=torch.float)
    return offsets @ cell, offsets

def threshold_sort(all_distances, r, n_neighbors):

    A = all_distances

    # keep n_neighbors only
    N = len(A) - n_neighbors - 1
    if N > 0:
        _, indices = torch.topk(A, N)
        A = torch.scatter(
            A,
            1,
            indices,
            torch.zeros(len(A), len(A), device=all_distances.device, dtype=torch.float),
        )

    A[A > r] = 0
    return A

def get_cutoff_distance_matrix(
    pos, cell, r, n_neighbors, device, image_selfloop, offset_number=1
):
    cells, cell_coors = get_pbc_cells(cell, offset_number, device=device)
    distance_matrix, min_indices = get_distances(pos, cells, device=device)

    cutoff_distance_matrix = threshold_sort(distance_matrix, r, n_neighbors)


    all_cell_offsets = cell_coors[torch.flatten(min_indices)]
    all_cell_offsets = all_cell_offsets.view(len(pos), -1, 3)

    n_edges = torch.count_nonzero(cutoff_distance_matrix).item()
    cell_offsets = torch.zeros(n_edges + len(pos), 3, dtype=torch.float)

    cell_offsets[:n_edges, :] = all_cell_offsets[cutoff_distance_matrix != 0]

    return cutoff_distance_matrix, cell_offsets

def normalize_edge_cutoff(dataset, descriptor_label, r):
    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] / r
        )

def generate_edge_features(input_data, edge_steps, r, device):

    if isinstance(input_data, Data):
        input_data = [input_data]

    normalize_edge_cutoff(input_data, "distance", r)
    for i, data in enumerate(input_data):
        input_data[i].edge_attr = input_data[i].edge_descriptor["distance"].reshape(-1,1)



def triplets(edge_index, cell_offsets, num_nodes):
    """
    Taken from the DimeNet implementation on OCP
    """

    row, col = edge_index  

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)


    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()


    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()

    cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
    mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1).to(
        device=idx_i.device
    )

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

    return idx_i, idx_j, idx_k, idx_kj, idx_ji

def compute_bond_angles(
    pos: torch.Tensor, offsets: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Compute angle between bonds Taken from the DimeNet implementation on OCP
    """
    # Calculate triplets
    idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
        edge_index, offsets.to(device=edge_index.device), num_nodes
    )

    # Calculate angles.
    pos_i = pos[idx_i]
    pos_j = pos[idx_j]

    offsets = offsets.to(pos.device)

    pos_ji, pos_kj = (
        pos[idx_j] - pos_i + offsets[idx_ji],
        pos[idx_k] - pos_j + offsets[idx_kj],
    )

    a = (pos_ji * pos_kj).sum(dim=-1)
    b = torch.linalg.cross(pos_ji, pos_kj).norm(dim=-1)


    angle = torch.atan2(b, a)

    return angle, idx_kj, idx_ji

def create_global_feat(atoms_index_arr):
    import numpy as np
    comp    = np.zeros(108)
    temp    = np.unique(atoms_index_arr,return_counts=True)
    for i in range(len(temp[0])):
            comp[temp[0][i]]=temp[1][i]/temp[1].sum() 
    return comp.reshape(1, -1)


def clean_up(data_list, attr_list):
    if not attr_list:
        return

    # check which attributes in the list are removable
    removable_attrs = [t for t in attr_list if t in data_list[0].to_dict()]
    for data in data_list:
        for attr in removable_attrs:
            delattr(data, attr)

def compute_periodic_complete_features(pos, cell, edge_index, num_nodes, cutoff=4.0):
    """Compute PerCNet-inspired periodic complete features."""
    import numpy as np
    import torch

    offset_number = 1
    cells, cell_coors = get_pbc_cells(cell, offset_number, device=pos.device)
    distance_matrix, min_indices = get_distances(pos, cells, device=pos.device)

    adj_matrix = threshold_sort(distance_matrix.clone(), r=cutoff, n_neighbors=12)

    all_index = [[] for _ in range(num_nodes)]
    all_images = [[] for _ in range(num_nodes)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                all_index[i].append(j)
                offset_idx = min_indices[i, j]
                image_offset_vec = cell_coors[offset_idx]
                all_images[i].append(image_offset_vec.cpu().numpy())

    coords_in_cell = pos.cpu().numpy()
    matrix = cell.cpu().numpy()

    n0_index = [item[0] if len(item) > 0 else 0 for item in all_index]
    n0_image = [item[0] if len(item) > 0 else [0, 0, 0] for item in all_images]
    n1_index = [item[1] if len(item) > 1 else 0 for item in all_index]
    n1_image = [item[1] if len(item) > 1 else [0, 0, 0] for item in all_images]

    all_pos_ij, all_pos_iref, all_pos_jref_j = [], [], []
    tuple_u, tuple_v = [], []

    for i in range(num_nodes):
        pos_i = coords_in_cell[i]

        shifted_n0_i = np.sum(matrix * np.array(n0_image[i])[:, np.newaxis], axis=0) + coords_in_cell[n0_index[i]]
        pos_in0 = shifted_n0_i - pos_i

        for neighbor_idx, j in enumerate(all_index[i]):
            tuple_u.append(i)
            tuple_v.append(j)

            image_offset_j = np.array(all_images[i][neighbor_idx])
            shifted_j = np.sum(matrix * image_offset_j[:, np.newaxis], axis=0) + coords_in_cell[j]
            pos_ij = shifted_j - pos_i
            all_pos_ij.append(pos_ij)

            if j == n0_index[i] and np.allclose(image_offset_j, n0_image[i]):
                shifted_n1_i = np.sum(matrix * np.array(n1_image[i])[:, np.newaxis], axis=0) + coords_in_cell[n1_index[i]]
                all_pos_iref.append(shifted_n1_i - pos_i)
            else:
                all_pos_iref.append(pos_in0)

            n0_idx_j_val = n0_index[j]
            n0_img_j_val = np.array(n0_image[j]) + image_offset_j
            n1_idx_j_val = n1_index[j]
            n1_img_j_val = np.array(n1_image[j]) + image_offset_j

            shifted_n0_j = np.sum(matrix * n0_img_j_val[:, np.newaxis], axis=0) + coords_in_cell[n0_idx_j_val]
            pos_jn0 = shifted_n0_j - shifted_j

            if i == n0_idx_j_val and np.allclose(n0_img_j_val, [0, 0, 0]):
                shifted_n1_j = np.sum(matrix * n1_img_j_val[:, np.newaxis], axis=0) + coords_in_cell[n1_idx_j_val]
                all_pos_jref_j.append(shifted_n1_j - shifted_j)
            else:
                all_pos_jref_j.append(pos_jn0)

    if not all_pos_ij:
        empty_tensor = torch.empty(0, dtype=torch.long, device=pos.device)
        empty_float = torch.empty(0, dtype=torch.float, device=pos.device)
        return empty_tensor.view(2, -1), empty_float, empty_float, empty_float, empty_float

    all_pos_ij = torch.tensor(np.array(all_pos_ij), dtype=torch.float, device=pos.device)
    all_pos_iref = torch.tensor(np.array(all_pos_iref), dtype=torch.float, device=pos.device)
    all_pos_jref_j = torch.tensor(np.array(all_pos_jref_j), dtype=torch.float, device=pos.device)

    dist = all_pos_ij.norm(dim=-1)
    a_theta = (all_pos_ij * all_pos_iref).sum(dim=-1)
    b_theta = torch.linalg.norm(torch.linalg.cross(all_pos_ij, all_pos_iref), dim=-1)
    theta = torch.atan2(b_theta, a_theta)

    plane1 = torch.linalg.cross(all_pos_ij, all_pos_iref)
    stable_ref_vec = all_pos_iref - all_pos_ij
    plane2 = torch.linalg.cross(all_pos_iref, stable_ref_vec)
    a_phi = (plane1 * plane2).sum(dim=-1)
    b_phi = torch.linalg.norm(all_pos_iref, dim=-1) * (plane1 * torch.linalg.cross(plane1, plane2)).sum(dim=-1) / (plane1.norm(dim=-1) * plane2.norm(dim=-1) + 1e-9)
    phi = torch.atan2(b_phi, a_phi)

    plane1_tau = torch.linalg.cross(all_pos_ij, all_pos_iref)
    plane2_tau = torch.linalg.cross(-all_pos_ij, all_pos_jref_j)
    a_tau = (plane1_tau * plane2_tau).sum(dim=-1)
    b_tau_num = dist * (torch.linalg.cross(plane1_tau, plane2_tau) * all_pos_ij).sum(dim=-1)
    b_tau_den = (all_pos_ij).norm(dim=-1) * (all_pos_ij).norm(dim=-1) + 1e-9
    b_tau = b_tau_num / b_tau_den
    tau = torch.atan2(b_tau, a_tau)

    tuple_edge_index = torch.tensor([tuple_u, tuple_v], dtype=torch.long, device=pos.device)

    return tuple_edge_index, dist, theta, phi, tau