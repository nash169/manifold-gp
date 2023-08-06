# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import h5py
import atom3d.datasets as da
from torch.utils.data import DataLoader
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid

# Regression datasets: SMP*, LBA, PSR*, RSR


class Voxel3D_SMP(object):
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
            'feature': self._voxelize(item['atoms']).ravel(),
            'label': np.float32(self._lookup_label(item, self.label_name)),
            # 'id': item['id'],
        }
        return transformed


class Voxel3D_LBA(object):
    def __init__(self, random_seed=None, **kwargs):
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
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms_pocket, atoms_ligand):
        # Use center of ligand as subgrid center
        ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
        ligand_center = get_center(ligand_pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                        ligand_center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein/ligand into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature': self._voxelize(item['atoms_pocket'], item['atoms_ligand']),
            'label': item['scores']['neglog_aff'],
            'id': item['id']
        }
        return transformed


class Voxel3D_PSR(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config = dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'C': 0,
                'O': 1,
                'N': 2,
                'S': 3,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 40.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.3,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        # Use center of protein as subgrid center
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

    def __call__(self, item):
        # Transform protein into voxel grids.
        # Apply random rotation matrix.
        id = eval(item['id'])
        transformed = {
            'feature': self._voxelize(item['atoms']).ravel(),
            'label': item['scores']['gdt_ts'],
            'target': id[0],
            'decoy': id[1],
        }
        return transformed


class Voxel3D_RSR(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config = dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                'C': 0,
                'O': 1,
                'N': 2,
                'P': 3,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 40.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.3,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms):
        # Use center of RNA as subgrid center
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

    def __call__(self, item):
        # Transform RNA into voxel grids.
        # Apply random rotation matrix.
        id = eval(item['id'])
        transformed = {
            'feature': self._voxelize(item['atoms']),
            'label': item['scores']['rms'],
            'target': id[0],
            'decoy': id[1],
        }
        return transformed


if __name__ == "__main__":
    import sys

    name = 'smp'  # 'smp', 'lba', 'psr', 'rsr'

    if name == 'smp':
        dataset = da.load_dataset('outputs/smp/', 'lmdb', transform=Voxel3D_SMP(label_name='alpha', radius=10.0))
    elif name == 'lba':
        dataset = da.load_dataset('outputs/lba/', 'lmdb', transform=Voxel3D_LBA(radius=10.0))
    elif name == 'psr':
        dataset = da.load_dataset('outputs/psr/', 'lmdb', transform=Voxel3D_PSR(radius=10.0))
    elif name == 'rsr':
        dataset = da.load_dataset('outputs/rsr/', 'lmdb', transform=Voxel3D_RSR(radius=10.0))
    else:
        print("Dataset not supported.")
        sys.exit()

    dataloader = DataLoader(dataset, batch_size=int(len(dataset)/32), num_workers=32)

    data = []
    for item in dataloader:
        data.append(np.append(item['feature'], item['label'][:, np.newaxis], axis=1))
    data = np.vstack(data)

    with open('outputs/datasets/'+name+'.npy', 'wb') as f:
        np.save(f, data)

    # data = np.load('outputs/datasets/smp.npy')

    # h5f = h5py.File('outputs/datasets/smp.h5', 'w')
    # h5f.create_dataset('dataset', data=data)
    # h5f.close()

    # h5f = h5py.File('outputs/datasets/smp.h5','r')
    # data = h5f['dataset'][:]
    # h5f.close()
