#!/usr/bin/python
# -*- encoding: utf-8 -*-
debug = True

import warnings
warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")

import torch

import logging
import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset,Data
from utils.helpers import (
    clean_up,
    generate_edge_features,
    generate_node_features,
    get_cutoff_distance_matrix,
)
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import Compose
from utils.transforms import GetY
import torch_geometric.transforms as T

class MP18(InMemoryDataset):

    def __init__(self, root='data/', name='MP18', transform=None, pre_transform=[GetY()], r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100, target_name="formation_energy_per_atom", global_cutoff=10.0):
        self.name = name.lower()
        assert self.name in ['mp18', 'pt','2d','mof','surface','cubic', 'cif','jarvis_fe_15k', 'jarvis_bg_15k', 'jarvis_multitask', 'test_minimal']
        self.r = r
        self.n_neighbors = n_neighbors
        self.edge_steps = edge_steps
        self.image_selfloop = image_selfloop
        self.points = points
        self.target_name = target_name
        self.global_cutoff = global_cutoff
        self.device = torch.device('cpu')

        super(MP18, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property  
    def raw_dir(self):
        if self.name == 'cif':
            return ''
        else:
            return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        if self.name in ['jarvis_fe_15k', 'jarvis_bg_15k', 'jarvis_multitask']:
            file_names = [f'{self.name}.json']

        elif self.name == 'cif':
            from glob import glob
            file_names = glob(f"{self.root}/*.cif")
        else:
            exit(1)
        return file_names

    @property
    def processed_file_names(self):
        processed_name = 'data_{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)
        return [processed_name]

    def process(self):
        logging.info("Data found at %s", self.raw_dir)
        logging.info("Starting dataset processing: %s", self.name)
        logging.info("Step 1/3: Reading and parsing structure data...")
        dict_structures, y = self.json_wrap()
        logging.info("Successfully read %s structures", len(dict_structures))

        logging.info("Step 2/3: Converting to graph data format...")
        data_list = self.get_data_list(dict_structures, y)
        logging.info("Graph data conversion completed")

        logging.info("Step 3/3: Saving processed data...")
        data, slices = self.collate(data_list) 
        torch.save((data, slices), self.processed_paths[0])
        logging.info("Processed data saved successfully.")
        logging.info("Dataset processing completed! Saved to: %s", self.processed_paths[0])

    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}.pt'.format(self.name, self.r, self.n_neighbors, self.edge_steps, self.image_selfloop, self.points, self.target_name)

    def __repr__(self):
        return '{}()'.format(self.name)

    def pymatgen2ase(self,pymat_structure):
        from pymatgen.io.ase import AseAtomsAdaptor
        Adaptor = AseAtomsAdaptor()
        return Adaptor.get_atoms(pymat_structure)

    def json_wrap(self):
        import pandas as pd
        import os
        logging.info("Reading individual structures using Pymatgen.")

        from pymatgen.core import Structure
        if self.name.lower() in ['cif']:
            cifFiles = []
            for i in self.raw_paths:
                with open(i, 'r') as f:
                    strContent = f.read()
                cifFiles.append(strContent)
            ids = [os.path.basename(i).split('.')[0] for i in self.raw_paths]
            df = pd.DataFrame({'structure': cifFiles, 'material_id': ids, 'property': [.0]*len(ids)})
        else:
            if self.name in ['jarvis_fe_15k', 'jarvis_bg_15k', 'jarvis_multitask']:
                logging.info("Reading custom data '%s' with orient='split'.", self.name)
                df = pd.read_json(self.raw_paths[0], orient='split')
            else:
                logging.info("Reading original data '%s' with default orient.", self.name)
                df  = pd.read_json(self.raw_paths[0])
        logging.info("Converting data to standardized form(dict format) for downstream processing.")

        if isinstance(self.target_name, list):
            y = df[self.target_name].to_numpy()
        else:
            y = df[[self.target_name]].to_numpy()

        dict_structures = []
        for i, s in enumerate(df["structure"]):
            if i == self.points:
                break
            s = Structure.from_str(s, fmt="cif") 
            s = self.pymatgen2ase(s)
            d = {}
            pos = torch.tensor(s.get_positions(), dtype=torch.float)  
            cell = torch.tensor(
                np.array(s.get_cell()), dtype=torch.float
            )
            atomic_numbers = torch.LongTensor(s.get_atomic_numbers())

            if self.name == 'cubic':
                def getAB(element):
                    if df['A'][i] == element:
                        return 7
                    elif df['B'][i] == element:
                        return 8
                    else:
                        return 9
                d["AB"] = torch.LongTensor([getAB(i)  for i in s.get_chemical_symbols()])

            d["positions"] = pos
            d["cell"] = cell
            d["atomic_numbers"] = atomic_numbers
            d["structure_id"] = str(df['material_id'][i])

            _atoms_index     = s.get_atomic_numbers()
            from utils.helpers import create_global_feat
            gatgnn_glob_feat = create_global_feat(_atoms_index)
            gatgnn_glob_feat = np.repeat(gatgnn_glob_feat,len(_atoms_index),axis=0)
            d["gatgnn_glob_feat"] = torch.Tensor(gatgnn_glob_feat).float()

            dict_structures.append(d)

            if i == 0:
                length = [len(_atoms_index)]
                elements = [list(set(_atoms_index))]
            else:
                length.append(len(_atoms_index))
                elements.append(list(set(_atoms_index)))
            n_atoms_max = max(length)
        species = list(set(sum(elements, [])))
        species.sort()
        num_species = len(species)
        logging.info("Max structure size: %s; Max number of elements: %s", n_atoms_max, num_species)
        y = y[:len(dict_structures)]
        
        return dict_structures, y
    
    def get_data_list(self, dict_structures, y):
        n_structures = len(dict_structures)
        data_list = [Data() for _ in range(n_structures)]

        logging.info("Getting torch_geometric.data.Data() objects.")

        logging.info("Processing %s structures...", n_structures)
        for i, sdict in enumerate(dict_structures):
            target_val = y[i]
            data = data_list[i]

            pos = sdict["positions"]
            cell = sdict["cell"]
            atomic_numbers = sdict["atomic_numbers"]
            structure_id = sdict["structure_id"]

            cd_matrix, cell_offsets = get_cutoff_distance_matrix(
                pos,
                cell,
                self.r,
                self.n_neighbors,
                image_selfloop=self.image_selfloop,
                device=self.device,
            )

            edge_indices, edge_weights = dense_to_sparse(cd_matrix) 

            data.n_atoms = len(atomic_numbers)
            data.pos = pos
            data.cell = cell
            data.y = torch.Tensor(np.array([target_val]))
            data.z = atomic_numbers
            if self.name == 'cubic':
                data.AB = sdict["AB"]
            data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
            data.edge_index, data.edge_weight = edge_indices, edge_weights
            data.cell_offsets = cell_offsets

            num_nodes = len(atomic_numbers)
            if num_nodes < 128:
                row = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
                col = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
                mask = row != col
                tuple_edge_index = torch.stack([row[mask], col[mask]], dim=0)
            else:
                dists = torch.cdist(pos, pos)
                mask = (dists < self.global_cutoff) & (dists > 1e-6)
                tuple_edge_index = mask.nonzero(as_tuple=False).t()
            data.tuple_edge_index = tuple_edge_index

            data.edge_descriptor = {}

            data.edge_descriptor["distance"] = edge_weights
            data.distances = edge_weights
            data.structure_id = [[structure_id] * len(data.y)]

            data.glob_feat   = sdict["gatgnn_glob_feat"]

            logging.info("All structures processed! Total: %s", n_structures)
        
        logging.info("Generating node features...")
        generate_node_features(data_list, self.n_neighbors, device=self.device)
        logging.info("Node feature generation completed")

        logging.info("Generating edge features...")
        generate_edge_features(data_list, self.edge_steps, self.r, device=self.device)
        logging.info("Edge feature generation completed")

        logging.debug("Applying transforms.")

        transform_name = self.pre_transform[0].__class__.__name__
        assert transform_name in ["GetY", "GetMultiTaskY"], \
            f"The target transform GetY or GetMultiTaskY is required in pre_transform, got {transform_name}"

        composition = Compose(self.pre_transform)

        for data in data_list:
            composition(data)

        clean_up(data_list, ["edge_descriptor"])

        return data_list
    

from torch_geometric.loader import DataLoader

def dataset_split(
    dataset,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 666,
    debug=True,
):     
    import logging
    if train_size + valid_size + test_size != 1:
        import warnings
        warnings.warn("Invalid sizes detected. Using default split of 80/10/10.")
        train_size, valid_size, test_size = 0.8, 0.1, 0.1

    dataset_size = len(dataset)

    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = dataset_size - train_len - valid_len
    
    unused_len = dataset_size - train_len - valid_len - test_len
    from torch.utils.data import random_split
    (train_dataset, val_dataset, test_dataset, unused_dataset) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
        generator=torch.Generator().manual_seed(seed),
    )
    logging.info(
        "train length: %s, val length: %s, test length: %s, unused length: %s, seed: %s",
        train_len,
        valid_len,
        test_len,
        unused_len,
        seed,
    )
    return train_dataset, val_dataset, test_dataset

def get_dataloader(
    train_dataset,   val_dataset,  test_dataset , batch_size: int, num_workers: int = 0,pin_memory=False
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,

    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader 

def split_data_CV(dataset, num_folds=5, seed=666, save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    logging.info("fold length: %s, unused length: %s, seed: %s", fold_length, unused_length, seed)
    return cv_dataset[0:num_folds]

def loader_setup_CV(index, batch_size, dataset,  num_workers=0):
    train_dataset = [x for i, x in enumerate(dataset) if i != index]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = dataset[index]

    train_loader = val_loader = test_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,)

    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    dataset = MP18(root="data",name='pt',transform=None, r=8.0, n_neighbors=12, edge_steps=50, image_selfloop=True, points=100,target_name="property")
    if debug:
        train_dataset, val_dataset, test_dataset = dataset_split( dataset, train_size=0.8,valid_size=0.15,test_size=0.05,seed=666)   
        train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, 64,24)