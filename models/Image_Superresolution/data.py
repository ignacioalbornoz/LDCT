import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from dataset import DefaultDataset
from sketch_dataset import SketchPairDataset
from torch.utils.data import DataLoader

def create_dataset(dataset_opt, phase):
    name = dataset_opt.get('name', '').lower()
    if name == 'sketchy_edges2sketch':
        # Usa el nuevo dataset
        return SketchPairDataset(
            dataroot=dataset_opt['dataroot'],
            img_size=dataset_opt.get('r_resolution', 256),
            norm=True,
            img_datatype=None,
            transforms=None
        )
    else:
        # Usa el dataset original (ajusta los argumentos según tu config)
        return DefaultDataset(
            file_path=dataset_opt['dataroot'],
            img_size=dataset_opt.get('r_resolution', 512),
            train=(phase=='train'),
            transforms=None
        )

def create_dataloader(dataset, dataset_opt, phase):
    return DataLoader(
        dataset,
        batch_size=dataset_opt.get('batch_size', 4),
        shuffle=dataset_opt.get('use_shuffle', True) if phase=='train' else False,
        num_workers=dataset_opt.get('num_workers', 4),
        pin_memory=True
    ) 