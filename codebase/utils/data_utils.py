import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(torch.utils.data.Dataset):
    """NpzDataset: loads a npz file as input."""

    def __init__(self, opt, partition):
        self.opt = opt
        self.root_dir = Path(opt.cwd, opt.input.load_path)
        file_name = Path(self.root_dir, f"{opt.input.file_name}_{partition}.npz")

        self.dataset = np.load(file_name)
        self.images = torch.Tensor(self.dataset["images"])
        self.labels = self.dataset["labels"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = (self.images[idx] + 1) / 2  # Normalize to [0, 1] range.
        return images, self.labels[idx]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(opt, dataset):
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )
    iterator = iter(data_loader)

    return data_loader, iterator


def get_data(opt, partition):
    dataset = NpzDataset(opt, partition)
    loader, iterator = get_dataloader(opt, dataset)
    return loader, iterator


def get_input(opt, iterator, train_loader):
    try:
        input = next(iterator)
    except StopIteration:
        # Create new generator if the previous generator is exhausted.
        iterator = iter(train_loader)
        input = next(iterator)

    input_images, labels = input

    if opt.use_cuda:
        input_images = input_images.cuda()

    return input_images, labels
