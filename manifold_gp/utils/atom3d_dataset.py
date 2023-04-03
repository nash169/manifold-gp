# !/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np

import atom3d.datasets as da
import atom3d.util.transforms as tr

from torch.utils.data import DataLoader
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid
import atom3d.splits.splits as spl


class TransformSMP(object):
    def __init__(self, label_name, random_seed=None, **kwargs):
        self.label_name = label_name
        self.random_seed = random_seed
        self.grid_config = dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'H': 0,
                'C': 1,
                'O': 2,
                'N': 3,
                'F': 4,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 7.5,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        # Use center of molecule as subgrid center
        pos = atoms[['x', 'y', 'z']].astype(np.float32)
        center = get_center(pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(atoms, center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def _lookup_label(self, item, name):
        if 'label_mapping' not in self.__dict__:
            label_mapping = [
                'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                'u0', 'u298', 'h298', 'g298', 'cv',
                'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
            ]
            self.label_mapping = {k: v for v, k in enumerate(label_mapping)}
        return item['labels'][self.label_mapping[name]]

    def __call__(self, item):
        # Transform molecule into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature': self._voxelize(item['atoms']).sum(axis=0).ravel(),
            'label': self._lookup_label(item, self.label_name),
            'id': item['id'],
        }
        return transformed


class Atom3D():
    def __init__(self, path, format, transform):
        dataset = da.load_dataset(path, format, transform=transform)
        indices = torch.randperm(len(dataset))
        train_split = int(len(dataset)*0.6)
        test_split = int(len(dataset)*0.2)
        self.train_dataset, self.val_dataset, self.test_dataset = spl.split(dataset, indices[:train_split].tolist(
        ), indices[train_split:train_split+test_split].tolist(), indices[train_split+test_split:].tolist())

    def trainset(self, device, cut=None):
        dataloader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset) if cut is None else cut, shuffle=False)
        for item in dataloader:
            features = item['feature'].to(device).to(torch.float32)
            label = item['label'].to(device).to(torch.float32)
            break
        return features, label

    def valset(self, device, cut=None):
        dataloader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset) if cut is None else cut, shuffle=False)
        for item in dataloader:
            features = item['feature'].to(device).to(torch.float32)
            label = item['label'].to(device).to(torch.float32)
            break
        return features, label

    def testset(self, device, cut=None):
        dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset) if cut is None else cut, shuffle=False)
        for item in dataloader:
            features = item['feature'].to(device).to(torch.float32)
            label = item['label'].to(device).to(torch.float32)
            break
        return features, label

# # Atom3D dataset
# dataset = Atom3D('manifold_gp/data/smp', 'lmdb', TransformSMP(label_name='alpha', radius=10.0))
# train_x, train_y = dataset.trainset(device, 1000)
# test_x, test_y = dataset.testset(device, 1000)

# Load Dataset
# data = np.loadtxt('manifold_gp/data/casp.csv', delimiter=',')
# sampled_x = data[:, 1:]
# sampled_y = data[:, 0]
# sampled_x, id_unique = np.unique(sampled_x, axis=0, return_index=True)
# sampled_y = sampled_y[id_unique]
# sampled_x = torch.from_numpy(sampled_x).float().to(device)
# sampled_y = torch.from_numpy(sampled_y).float().to(device)
# del data

# data = files('manifold_gp.data').joinpath('dragon10k.stl')
# # data = files('manifold_gp.data').joinpath('dragon100k.msh')
# nodes, faces, truth = groundtruth_from_mesh(data)
# sampled_x = torch.from_numpy(nodes).float().to(device)
# sampled_y = torch.from_numpy(truth).float().to(device)

# # # Cut Dataset
# # cut = 10000
# # sampled_x = sampled_x[:cut, :]
# # sampled_y = sampled_y[:cut]

# # Input normalization
# # mu_n, std_n = sampled_x.mean(axis=0), sampled_x.std(axis=0)
# # sampled_x = (sampled_x - mu_n)/std_n
# # sampled_x.sub_(mu_n).div_(std_n)

# # # Split Dataset
# # split = int(np.round(sampled_x.shape[0]*0.6))
# # train_x = torch.from_numpy(sampled_x[:split, :]).float().to(device)
# # train_y = torch.from_numpy(sampled_y[:split]).float().to(device)
# # test_x = torch.from_numpy(sampled_x[split:, :]).float().to(device)
# # test_y = torch.from_numpy(sampled_y[split:]).float().to(device)
# # del sampled_x, sampled_y


# # # Output normalization
# # mu_n, std_n = train_y.mean(), train_y.std()
# # train_y.sub_(mu_n).div_(std_n)
