import os
import torch
import random
from jax import numpy as jnp
from imageio import imread
from skimage.transform import resize
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


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(jnp.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(jnp.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res)
    else:
        image = utils.crop_image(image, resized_res)

    # switch to numpy channel dim convention, resize, switch back
    image = jnp.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return jnp.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=False, stacked_complex=False):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image, target_res), target_res)




class PhaseCaptureLoader(torch.utils.data.IterableDataset):
    """Loads (phase, captured, focal_idxs) pairs for forward model training

    Class initialization parameters
    -------------------------------
    phase_path: folder containing phases ([img_idx]_[iters].png)
    captured_path: folder containing captured_amps ([img_idx]_[iters]_[focus_idx].png)
    focus_idxs: indices of focal stack, default None - [0,1,3,4,5,6,7,9,10]
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    homography_res: 2d dims to scale the image to before final crop to image_res
        for consistent resolutions (crops to preserve input aspect ratio),
        default (1080, 1920)
    shuffle: True to randomize image order across batches, default True
    idx_subset: for the iterator, skip all but these images. Given as a list of
        indices corresponding to sorted filename order. Forces shuffle=False and
        batch_size=1. Defaults to None to not subset at all.
    crop_to_homography: if True, only crops the image instead of scaling to get
        to target homography resolution, default False

    Usage
    -----
    To be used as an iterator:

    >>> pairs_loader = PairsLoader(...)
    >>> for slm_phase, captured_amp, focus_idxs in pairs_loader:
    >>>     ...

    for batch size N and number of focal planes M,
    slm_phase: (N, 1, H, W) tensor.
    captured_amp: (N, M, H, W) tensor.
    focus_idxs: a python list length of M.

    >>> slm_phase, captured_amp, focus_idxs = pairs_loader.load_image(idx)

    idx: the index for the image to load, indices are alphabetical based on the
        file path.
    """

    def __init__(self, phase_path, captured_path, channel=None, batch_size=1,
                 image_res=(1080, 1920), shuffle=True, idx_subset=None, sled=False):

        if not os.path.isdir(phase_path):
            raise NotADirectoryError(f'Data folder: {phase_path}')
        if not os.path.isdir(captured_path):
            raise NotADirectoryError(f'Data folder: {captured_path}')

        self.phase_path = phase_path
        self.captured_path = captured_path

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

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __next__(self):
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
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)

    def load_pair(self, filenum):
        """
        Load image but process it on gpu (should be much faster)

        :param filenum:
        :param focus_idxs: suffices for focus states in filename
        :return:
        """

        im = imread(self.im_names[filenum])
        im = (1 - im / jnp.iinfo(jnp.uint8).max) * 2 * jnp.pi - jnp.pi
        phase_im = jnp.expand_dims(im, axis=0).astype(jnp.float32)
        phase_im = torch.tensor(phase_im, device=self.dev)

        _, captured_filename = os.path.split(os.path.splitext(self.im_names[filenum])[0])
        idx = captured_filename.split('/')[-1]

        captured_filename = os.path.join(self.captured_path, f'{idx}_5.png') # Extract only the intermediate plane
        captured_intensity = jnp.sqrt(utils.im2float(skimage.io.imread(captured_filename)))
        captured_intensity = jnp.expand_dims(captured_intensity, axis=0)
        captured_intensity = torch.tensor(captured_intensity, device=self.dev)

        return (phase_im, captured_amp)
