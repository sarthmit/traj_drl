# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
# from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from collections import defaultdict
import matplotlib.pyplot as plt

from config import datasets as datasets_new
from config import cli
from config.augmentations import RandAugment
from torchvision.transforms import transforms
# import helpers
import torch.nn as nn
import torchvision

from functools import partial
import pandas as pd
import PIL
import glob

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, io
from torchvision.datasets.utils import verify_str_arg

FLAGS = flags.FLAGS

class CelebADataset(Dataset):
    """CelebA Dataset class"""

    def __init__(self,
                 root,
                 split="train",
                 target_type="attr",
                 transform=None,
                 target_transform=None,
                 download=False
                 ):
        """
        """

        self.root = root
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download_from_kaggle()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }

        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.csv"), delim_whitespace=False, header=0, index_col=0)
        # This file is not available in Kaggle
        # identity = pd.read_csv(fn("identity_CelebA.csv"), delim_whitespace=True, header=None, index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.csv"), delim_whitespace=False, header=0, index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.csv"), delim_whitespace=False, header=0,
                                      index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.csv"), delim_whitespace=False, header=0, index_col=0)

        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        self.filename = splits[mask].index.values
        # self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def download_from_kaggle(self):

        # Annotation files will be downloaded at the end
        label_files = ['list_attr_celeba.csv', 'list_bbox_celeba.csv', 'list_eval_partition.csv',
                       'list_landmarks_align_celeba.csv']

        # Check if files have been downloaded already
        files_exist = False
        for label_file in label_files:
            if os.path.isfile(os.path.join(self.root, label_file)):
                files_exist = True
            else:
                files_exist = False

        if files_exist:
            print("Files exist already")
        else:
            print("Downloading dataset. Please while while the download and extraction processes complete")
            # Download files from Kaggle using its API as per
            # https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python

            # Kaggle authentication
            # Remember to place the API token from Kaggle in $HOME/.kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            # Download all files of a dataset
            # Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
            api.dataset_download_files(dataset='jessicali9530/celeba-dataset',
                                       path=self.root,
                                       unzip=True,
                                       force=False,
                                       quiet=False)

            # Downoad the label files
            # Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
            for label_file in label_files:
                api.dataset_download_file(dataset='jessicali9530/celeba-dataset',
                                          file_name=label_file,
                                          path=self.root,
                                          force=False,
                                          quiet=False)

            # Clear any remaining *.csv.zip files
            files_to_delete = glob.glob(os.path.join(self.root, "*.csv.zip"))
            for f in files_to_delete:
                os.remove(f)

            print("Done!")

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root,
                                        "img_align_celeba",
                                        "img_align_celeba",
                                        self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(f"Target type {t} is not recognized")

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

channel_stats = {
    'cifar10': dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    'cifar100': dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
    'mini_imgnet': dict(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
}

sizes = {
    'cifar10': 32,
    'cifar100': 32,
    'mini_imgnet': 84
}

padding = {
    'cifar10': 4,
    'cifar100': 4,
    'mini_imgnet': 8
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_transforms(dataset, aug):
    if dataset == 'synthetic' or dataset == 'cmnist':
        return transforms.ToTensor()

    if dataset == 'celeba':
        return transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor()
        ])

    size = sizes[dataset]
    pad = padding[dataset]
    stats = channel_stats[dataset]
    if aug == 'none':
        transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(**stats)])
    elif aug == 'simclr':
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(**stats)])
    elif aug == 'laplace_strong':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=pad, padding_mode="reflect"),
            RandAugment(2),
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ])
    elif aug == 'laplace_weak':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=pad, padding_mode="reflect"),
            RandAugment(1),
            transforms.ToTensor(),
            transforms.Normalize(**stats)
        ])
    else:
        print('No Augmentation Found')
        exit()

    return transform

def get_dataloaders(dataset, bsz, aug='none'):
    transform = get_transforms(dataset, aug)
    if dataset == 'cifar10':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar10/by-image/train+val',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar10/by-image/test',
            transform = transform
        )
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/train+val',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/cifar/cifar100/by-image/test',
            transform = transform
        )
    elif dataset == 'mini_imgnet':
        train_data = torchvision.datasets.ImageFolder(
            root = './data/images/miniimagenet/train',
            transform = transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root = './data/images/miniimagenet/test',
            transform = transform
        )
    elif dataset == 'synthetic':
        train_x = torch.Tensor(np.load('./data/synthetic/train_imgs.npy', allow_pickle=True))
        train_y = torch.Tensor(np.load('./data/synthetic/train_labels.npy', allow_pickle=True))
        test_x = torch.Tensor(np.load('./data/synthetic/test_imgs.npy', allow_pickle=True))
        test_y = torch.Tensor(np.load('./data/synthetic/test_labels.npy', allow_pickle=True))

        train_data = torch.utils.data.TensorDataset(train_x, train_y)
        test_data = torch.utils.data.TensorDataset(test_x, test_y)
    elif dataset == 'cmnist':
        train_x = torch.Tensor(np.load('./data/cmnist/train_imgs.npy', allow_pickle=True))
        train_y = torch.Tensor(np.load('./data/cmnist/train_labels.npy', allow_pickle=True))
        test_x = torch.Tensor(np.load('./data/cmnist/test_imgs.npy', allow_pickle=True))
        test_y = torch.Tensor(np.load('./data/cmnist/test_labels.npy', allow_pickle=True))

        train_data = torch.utils.data.TensorDataset(train_x, train_y)
        test_data = torch.utils.data.TensorDataset(test_x, test_y)
    elif dataset == 'celeba':
        train_data = CelebADataset('./data/celeba/', split='train',
                                                 transform = transform,
                                                 download=True)
        test_data = CelebADataset('./data/celeba/', split='test',
                                                 transform=transform,
                                                download=True)
    else:
        print('Dataset Not Supported')
        exit()

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = bsz,
        shuffle=True,
        num_workers = 1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = bsz,
        shuffle=True,
        num_workers = 1
    )

    return train_loader, test_loader

def cycle(iterable):
    while True:
        for aug_images, target in iterable:
            yield aug_images, target

def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    logging.info('train')
    logging.info(tf.config.list_physical_devices('GPU'))
    set_seed(config.training.seed)
    print(f"Dataset: {config.data.dataset}")
    print(f"Augmentations: {config.data.aug}")
    print(f"Seed: {config.training.seed}")
    print(f"Widen Factor: {config.model.widen_factor}")
    print(f"Regularizer: {config.training.lambda_z}")
    print(f"Probabilistic: {config.training.probabilistic_encoder}")

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, f"samples_{config.training.experiment_name}")
    tf.io.gfile.makedirs(sample_dir)

    args = cli.parse_commandline_args()
    args.dataset = config.data.dataset
    if args.dataset == 'mini_imgnet':
        config.data.image_size = 84
        config.model.ch_mult = (1, 2, 2)
    elif args.dataset == 'celeba':
        config.data.image_size = 64
        config.model.ch_mult = (1, 2, 2)
    # args = helpers.load_args(args)
    dataset_config = datasets_new.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    args.num_classes = num_classes
    train_loader, eval_loader = get_dataloaders(args.dataset, args.batch_size, config.data.aug)

    train_iter = iter(cycle(train_loader))
    eval_iter = iter(cycle(eval_loader))

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, f"checkpoints_{config.training.experiment_name}")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, f"checkpoints-meta_{config.training.experiment_name}", "checkpoint.pth")
    checkpoint_enc_dir = os.path.join(workdir, f"checkpointsenc_{config.training.experiment_name}")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_enc_dir)
    tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting, config=config)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting, config=config)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        images, labels = next(train_iter)
        batch = images.to(config.device)

        if step == 0:
            logging.info(batch.shape)
        if step == 0:
            logging.info(batch.shape)

        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        if step % config.training.eval_freq == 0:
            images, labels = next(eval_iter)
            eval_batch = images.to(config.device)

            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

        if step % config.training.snapshot_freq == 0 or step == num_train_steps:
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            if getattr(config.training, 'include_encoder', False):
                encoder_state = state['model'].module.encoder.state_dict()
                torch.save(encoder_state, os.path.join(checkpoint_enc_dir, f'encoder_state_{save_step}.pth'))

            # Generate and save samples
            if config.training.snapshot_sampling and step > 0:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                tf.io.gfile.makedirs(this_sample_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, image_grid.cpu().numpy()) # sample)

                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, f'{eval_folder}_{config.training.experiment_name}')
    tf.io.gfile.makedirs(eval_dir)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, f"checkpoints_{config.training.experiment_name}")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
        ds_bpd = train_ds_bpd
        bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds_bpd = eval_ds_bpd
        bpd_num_repeats = 5
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        if config.eval.enable_loss:
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss.item())
                if (i + 1) % 1000 == 0:
                    logging.info("Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = np.asarray(all_losses)
            with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            bpds = []
            for repeat in range(bpd_num_repeats):
                bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (
                            ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    with tf.io.gfile.GFile(os.path.join(eval_dir,
                                                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                           "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}")
                tf.io.gfile.makedirs(this_sample_dir)
                samples, n = sampling_fn(score_model)
                samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
                # Write samples to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                # Force garbage collection before calling TensorFlow code for Inception network
                gc.collect()
                latents = evaluation.run_inception_distributed(samples, inception_model,
                                                               inceptionv3=inceptionv3)
                # Force garbage collection again before returning to JAX code
                gc.collect()
                # Save latent represents of the Inception network to disk or Google Cloud Storage
                with tf.io.gfile.GFile(
                        os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(
                        io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
                    fout.write(io_buffer.getvalue())

            # Compute inception scores, FIDs and KIDs.
            # Load all statistics that have been previously computed and saved for each host
            all_logits = []
            all_pools = []
            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
            for stat_file in stats:
                with tf.io.gfile.GFile(stat_file, "rb") as fin:
                    stat = np.load(fin)
                    if not inceptionv3:
                        all_logits.append(stat["logits"])
                    all_pools.append(stat["pool_3"])

            if not inceptionv3:
                all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
            all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

            # Load pre-computed dataset statistics.
            data_stats = evaluation.load_dataset_stats(config)
            data_pools = data_stats["pool_3"]

            # Compute FID/KID/IS on all samples together.
            if not inceptionv3:
                inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
            else:
                inception_score = -1

            fid = tfgan.eval.frechet_classifier_distance_from_activations(
                data_pools, all_pools)
            # Hack to get tfgan KID work for eager execution.
            tf_data_pools = tf.convert_to_tensor(data_pools)
            tf_all_pools = tf.convert_to_tensor(all_pools)
            kid = tfgan.eval.kernel_classifier_distance_from_activations(
                tf_data_pools, tf_all_pools).numpy()
            del tf_data_pools, tf_all_pools

            logging.info(
                "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
                    ckpt, inception_score, fid, kid))

            with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                                   "wb") as f:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
                f.write(io_buffer.getvalue())
