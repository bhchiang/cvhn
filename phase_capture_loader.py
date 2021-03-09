import os
import torch
import random
from jax import numpy as jnp
import numpy as np
from imageio import imread
import skimage.io
import utils


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]
        return images


def get_image_filenames_without_focus(dir, capture_dir):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    if isinstance(dir, str):
        files = os.listdir(capture_dir)
        exts = (os.path.splitext(f)[1] for f in files)
        images = [os.path.join(dir, f'{f.split("_")[0]}_{f.split("_")[1]}.png')
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and f'{f.split("_")[0]}_{f.split("_")[1]}.png' in os.listdir(dir)]
        print(len(images))
        images = list(set(list))
        print(len(images))
        return images


class PhaseCaptureLoader(torch.utils.data.IterableDataset):
    """Loads (phase, captured) pairs for forward model training
    Class initialization parameters
    -------------------------------
    phase_path: folder containing phases ([img_idx]_[iters].png)
    captured_path: folder containing intermediate captured_amps ([img_idx]_[iters]_5.png)
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    shuffle: True to randomize image order across batches, default True
    idx_subset: for the iterator, skip all but these images. Given as a list of
        indices corresponding to sorted filename order. Forces shuffle=False and
        batch_size=1. Defaults to None to not subset at all.

    Usage
    -----
    To be used as an iterator:

    >>> phase_capture_loader = PhaseCaptureLoader(...)
    >>> for slm_phase, captured_amp in phase_capture_loader:
    >>>     ...

    for batch size N
    slm_phase: (N, H, W, 1) tensor.
    captured_amp: (N, H, W, 1) tensor.

    >>> slm_phase, captured_amp = phase_capture_loader.load_pair(idx)

    idx: the index for the image to load, indices are alphabetical based on the
        file path.
    """
    def __init__(self, phase_path, captured_path, channel=None, batch_size=1,
                 image_res=(1080, 1920), shuffle=True, idx_subset=None, sled=False):
        # Set path for loading data
        if not os.path.isdir(phase_path):
            raise NotADirectoryError(f'Data folder: {phase_path}')
        if not os.path.isdir(captured_path):
            raise NotADirectoryError(f'Data folder: {captured_path}')
        self.phase_path = phase_path
        self.captured_path = captured_path
        # Set parameters for batch loading
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.subset = idx_subset
        if sled:
            self.im_names = get_image_filenames_without_focus(phase_path, captured_path)
        else:
            self.im_names = get_image_filenames(phase_path)
        self.im_names.sort()
        # if subsetting indices, force no randomization and batch size 1
        if self.subset is not None:
            self.shuffle = False
            self.batch_size = 1
        # create list of image IDs
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)


    def __iter__(self):
        """
        reset iteration and shuffle order
        """
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self


    def __next__(self):
        """
        :return next element
        """
        if self.subset is not None:
            while self.ind not in self.subset and self.ind < len(self.order):
                self.ind += 1

        if self.ind < len(self.order):
            phase_idx = self.order[self.ind]

            self.ind += 1
            return self.load_pair(phase_idx[0])
        else:
            raise StopIteration


    def __len__(self):
        """
        :return length of order
        """
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)


    def load_pair(self, filenum):
        """
        Load image but process it on gpu (should be much faster)

        :param filenum: the index for the image to load, indices are alphabetical
                    based on the file path.
        :return slm_phase, captured_amp
        """

        im = imread(self.im_names[filenum])
        im = (1 - im / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        phase_im = torch.tensor(im, dtype=torch.float32).reshape(*im.shape, 1)

        _, captured_filename = os.path.split(os.path.splitext(self.im_names[filenum])[0])
        idx = captured_filename.split('/')[-1]

        captured_filename = os.path.join(self.captured_path, f'{idx}_5.png') # Extract only the intermediate plane
        captured_intensity = utils.im2float(skimage.io.imread(captured_filename))
        captured_intensity = torch.tensor(captured_intensity, dtype=torch.float32).reshape(*im.shape, 1)
        captured_amp = torch.sqrt(captured_intensity)

        return (phase_im, captured_amp)
