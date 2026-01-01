import os
import sys

import numpy as np
import torch
from easy_to_hard_data import MazeDataset as EasyToHardMazeDataset
from maze_dataset import set_serialize_minimal_threshold
from maze_dataset.dataset.rasterized import (
    MazeDataset,
    MazeDatasetConfig,
    RasterizedMazeDataset,
)
from maze_dataset.generation import LatticeMazeGenerators
from torch.utils import data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    from deepthinking.utils.plot import plot_maze_and_target

set_serialize_minimal_threshold(int(10**7))  # prevent crashing on large datasets

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_maze_loader_new(
    dataset="easy-to-hard-data",
    maze_size_train=9,
    maze_size_test=33,
    num_mazes_train=50000,
    num_mazes_test=10000,
    gen="dfs_perc",
    percolation=0.0,
    deadend_start=True,
    train_batch_size=32,
    test_batch_size=32,
    shuffle=True,
    train_val_split=0.8,
):
    """Generate mazes of the given size and number,
    from the given dataset, and load to device as DataLoaders"""

    trainset, valset, testset = None, None, None
    assert dataset in ["maze-dataset", "easy-to-hard-data"]

    if dataset == "maze-dataset":
        """ https://github.com/understanding-search/maze-dataset """

        assert maze_size_train % 2 == 1
        grid_n_train = (maze_size_train + 1) // 2
        assert maze_size_test % 2 == 1
        grid_n_test = (maze_size_test + 1) // 2

        maze_ctor = None
        maze_ctor_kwargs = {}

        if gen == "dfs":
            maze_ctor = LatticeMazeGenerators.gen_dfs
            maze_ctor_kwargs = dict()
        elif gen == "dfs_perc":
            maze_ctor = LatticeMazeGenerators.gen_dfs_percolation
            maze_ctor_kwargs = dict(p=percolation)
        elif gen == "percolation":
            maze_ctor = LatticeMazeGenerators.gen_percolation
            maze_ctor_kwargs = dict(p=percolation)
        endpoint_kwargs = dict(deadend_start=deadend_start, endpoints_not_equal=True)

        base_dataset_train = MazeDataset.from_config(
            MazeDatasetConfig(
                name="train",
                grid_n=grid_n_train,
                n_mazes=num_mazes_train,
                maze_ctor=maze_ctor,
                maze_ctor_kwargs=maze_ctor_kwargs,
                endpoint_kwargs=endpoint_kwargs,
            ),
            local_base_path="data/maze-dataset/",
        )

        base_dataset_test = MazeDataset.from_config(
            MazeDatasetConfig(
                name="test",
                grid_n=grid_n_test,
                n_mazes=num_mazes_test,
                maze_ctor=maze_ctor,
                maze_ctor_kwargs=maze_ctor_kwargs,
                endpoint_kwargs=endpoint_kwargs,
            ),
            local_base_path="data/maze-dataset/",
        )

        maze_dataset_train = RasterizedMazeDataset.from_base_MazeDataset(
            base_dataset=base_dataset_train,
            added_params=dict(
                remove_isolated_cells=True,
                extend_pixels=True,  # maps from 1x1 to 2x2 pixels and adds 3 padding
            ),
        )

        maze_dataset_test = RasterizedMazeDataset.from_base_MazeDataset(
            base_dataset=base_dataset_test,
            added_params=dict(
                remove_isolated_cells=True,
                extend_pixels=True,
            ),
        )

        class NormalizedMazeDataset(torch.utils.data.Dataset):
            def __init__(self, maze_dataset):
                self.maze_dataset = maze_dataset

            def __len__(self):
                return len(self.maze_dataset)

            def __getitem__(self, idx):
                x, y = self.maze_dataset[idx]

                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()

                x = x / 255.0
                y = y / 255.0

                x = x.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                y = y[:, :, 0]  # rgb -> binary
                return x, y

        train_dataset = NormalizedMazeDataset(maze_dataset_train)
        test_dataset = NormalizedMazeDataset(maze_dataset_test)

        train_size = int(train_val_split * len(train_dataset))
        val_size = len(train_dataset) - train_size

        trainset, valset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        testset = test_dataset

    elif dataset == "easy-to-hard-data":
        train_dataset = EasyToHardMazeDataset(
            "../../../data", train=True, size=maze_size_train, download=True
        )
        test_dataset = EasyToHardMazeDataset(
            "../../../data", train=False, size=maze_size_test, download=True
        )

        if num_mazes_train and num_mazes_train < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:num_mazes_train]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        if num_mazes_test and num_mazes_test < len(test_dataset):
            indices = torch.randperm(len(test_dataset))[:num_mazes_test]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)

        train_size = int(train_val_split * len(train_dataset))
        val_size = len(train_dataset) - train_size

        trainset, valset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        testset = test_dataset

    trainloader = data.DataLoader(
        trainset,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    valloader = data.DataLoader(
        valset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    testloader = data.DataLoader(
        testset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loaders = {"train": trainloader, "val": valloader, "test": testloader}
    return loaders


def prepare_maze_loader(
    train_batch_size, test_batch_size, train_data, test_data, shuffle=True
):
    train_data = EasyToHardMazeDataset(
        "../../../data", train=True, size=train_data, download=True
    )
    testset = EasyToHardMazeDataset(
        "../../../data", train=False, size=test_data, download=True
    )

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(
        train_data,
        [train_split, int(len(train_data) - train_split)],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = data.DataLoader(
        trainset,
        num_workers=0,
        batch_size=train_batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    valloader = data.DataLoader(
        valset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    testloader = data.DataLoader(
        testset,
        num_workers=0,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


# if __name__ == "__main__":
#     dataset = "maze-dataset"
#     # dataset =  "maze-dataset"

#     loaders = get_dataloaders_new(
#         dataset=dataset,
#         maze_size_train=9,
#         maze_size_test=11,
#         num_mazes_train=1000,
#         num_mazes_test=100,
#         gen="dfs_perc",
#         percolation=0.7,
#         deadend_start=False,
#         train_batch_size=32,
#         test_batch_size=32,
#         shuffle=True,
#     )
#     print("Data loaders created successfully.")
#     for key, loader in loaders.items():
#         print(f"{key} loader has {len(loader.dataset)} samples.")
#         for x, y in loader:
#             print(f"Batch shape: {x.shape}, {y.shape}")
#             for maze, target in zip(x, y):
#                 print(f"Maze shape: {maze.shape}, Target shape: {target.shape}")
#                 plot_maze_and_target(
#                     maze, target, save_str=f"maze_sample_{dataset}_{key}.png"
#                 )
#                 break

#             break
