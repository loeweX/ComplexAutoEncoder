import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from datetime import timedelta

from codebase.model import model_utils
from codebase.utils import data_utils, utils


def train(opt, model, optimizer, train_iterator, train_loader):
    start_time = time.time()

    for step in range(opt.training.steps + 1):
        input_images, labels = data_utils.get_input(opt, train_iterator, train_loader)

        optimizer, lr = utils.update_learning_rate(optimizer, opt, step)

        optimizer.zero_grad()
        outputs = model(input_images, labels, step)
        outputs["loss"].backward()
        optimizer.step()

        if step % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, step, model, "val")

    total_train_time = time.time() - start_time
    print(f"Total training time: {timedelta(seconds=total_train_time)}")

    model_utils.save_model(opt, model, optimizer)
    return optimizer, step


def validate_or_test(opt, step, model, partition):
    data_loader, data_iterator = data_utils.get_data(opt, partition)

    test_results = defaultdict(float)

    model.eval()
    print(partition)

    test_time = time.time()

    with torch.no_grad():
        for _ in tqdm(range(len(data_loader))):
            input_images, labels = data_utils.get_input(opt, data_iterator, data_loader)

            outputs = model(
                input_images, labels, step, partition=partition, evaluate=True
            )

            test_results["loss"] += outputs["loss"] / len(data_loader)
            test_results["ARI+BG"] += outputs["ARI+BG"] / len(data_loader)
            test_results["ARI-BG"] += outputs["ARI-BG"] / len(data_loader)

    utils.print_results(partition, step, time.time() - test_time, test_results)

    model.train()


@hydra.main(config_path="config", config_name="config")
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)

    model, optimizer = model_utils.get_model_and_optimizer(opt)
    train_loader, train_iterator = data_utils.get_data(opt, "train")

    optimizer, step = train(opt, model, optimizer, train_iterator, train_loader)

    validate_or_test(opt, step, model, "test")


if __name__ == "__main__":
    my_main()
