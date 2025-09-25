import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

COCO_DATASET_ROOT = "/home/sunyw/rofclip/datasets/coco2014"
GNM_DATASET_ROOT = "/home/sunyw/rofclip/datasets/gnm/portrayal_v2"

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

class NegCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, hard_captions_key="neg_caption", sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.hard_captions = df[hard_captions_key].tolist()
        self.hard_images = df["neg_image"].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open("/home/sunyw/rofclip/datasets/coco2014/"+str(self.images[idx]).replace('coco/','')))
        texts = self.tokenizer([str(self.captions[idx])])[0]

        chosen_caption = random.choice(self.hard_captions[idx])
        hard_captions = self.tokenizer([str(chosen_caption)])[0]

        chose_image_index = random.choice(self.hard_images[idx])

        new_images = self.transforms(Image.open("/home/sunyw/rofclip/datasets/coco2014/"+str(self.images[chose_image_index]).replace('coco/','')))
        new_texts = self.tokenizer([str(self.captions[chose_image_index])])[0]

        chosen_caption = random.choice(self.hard_captions[chose_image_index])
        new_hard = self.tokenizer([str(chosen_caption)])[0]
        
        images = torch.cat([images.unsqueeze(0), new_images.unsqueeze(0)],dim=0)
        texts = torch.stack([texts, new_texts, hard_captions, new_hard])
        return images, texts

class NpyDataset(Dataset):
    def __init__(self, samples_path, transforms=None, train_num_samples=25000, tokenizer=None, split=None, dist_token=False, pro=False):
        if split==None:
            if 'val' in samples_path:
                self.split='val'
            else:
                self.split='train'
        else:
            self.split=split

        if 'coco' in samples_path:
            self.data='coco'
        else:
            self.data='cc3m'
        if os.path.isdir(samples_path):
            if pro:
                data_file_splits = ['generated_data_pro/coco_train/processed_strict_dataset8.npy', 'generated_data_pro/coco_train/processed_strict_dataset7.npy', 'generated_data_pro/coco_train/processed_strict_dataset3.npy', 'generated_data_pro/coco_train/processed_strict_dataset1.npy', 'generated_data_pro/coco_train/processed_strict_dataset4.npy', 'generated_data_pro/coco_train/processed_strict_dataset5.npy', 'generated_data_pro/coco_train/processed_strict_dataset0.npy', 'generated_data_pro/coco_train/processed_strict_dataset6.npy', 'generated_data_pro/coco_train/processed_strict_dataset2.npy']
                # data_file_splits = ['generated_data_pro/coco_train/processed_strict_dataset8.npy', 'generated_data_pro/coco_train/processed_strict_dataset7.npy', 'generated_data_pro/coco_train/processed_strict_dataset6.npy', 'generated_data_pro/coco_train/processed_strict_dataset5.npy', 'generated_data_pro/coco_train/processed_strict_dataset4.npy', 'generated_data_pro/coco_train/processed_strict_dataset3.npy', 'generated_data_pro/coco_train/processed_strict_dataset2.npy', 'generated_data_pro/coco_train/processed_strict_dataset1.npy', 'generated_data_pro/coco_train/processed_strict_dataset0.npy']
                
            else:
                data_file_splits = ['processed_dataset8.npy', 'processed_dataset7.npy', 'processed_dataset3.npy', 'processed_dataset1.npy', 'processed_dataset4.npy', 'processed_dataset5.npy', 'processed_dataset0.npy', 'processed_dataset6.npy', 'processed_dataset2.npy']
                # data_file_splits = ['processed_dataset8.npy', 'processed_dataset7.npy', 'processed_dataset6.npy', 'processed_dataset5.npy', 'processed_dataset4.npy', 'processed_dataset3.npy', 'processed_dataset2.npy', 'processed_dataset1.npy', 'processed_dataset0.npy']
                
            
            print(data_file_splits)
            print(f'merging {len(data_file_splits)} splied files from {samples_path}')
            self.samples=[]
            for file_split in data_file_splits:
                self.samples.extend(self.loadList(os.path.join(samples_path,file_split)))
            
            self.embeddings = []
            if dist_token:
                embed_file_splits = ["generated_data/llama_embed_pca_8.npy", "generated_data/llama_embed_pca_7.npy", "generated_data/llama_embed_pca_3.npy", "generated_data/llama_embed_pca_1.npy", "generated_data/llama_embed_pca_4.npy", "generated_data/llama_embed_pca_5.npy", "generated_data/llama_embed_pca_0.npy", "generated_data/llama_embed_pca_6.npy", "generated_data/llama_embed_pca_2.npy"]
                # embed_file_splits = ["generated_data/llama_embed_pca_8.npy", "generated_data/llama_embed_pca_7.npy", "generated_data/llama_embed_pca_6.npy", "generated_data/llama_embed_pca_5.npy", "generated_data/llama_embed_pca_4.npy", "generated_data/llama_embed_pca_3.npy", "generated_data/llama_embed_pca_2.npy", "generated_data/llama_embed_pca_1.npy", "generated_data/llama_embed_pca_0.npy"]
                
                for file_split in embed_file_splits:
                    print(file_split)
                    item = np.load(file_split)
                    self.embeddings.extend([torch.tensor(item[i]) for i in range(item.shape[0])])
        else:
            # load single splited file fiven the path
            self.samples=self.loadList(samples_path)
        if train_num_samples:
            #######################
            self.samples=self.samples[:train_num_samples]
            ###
            # self.embeddings = self.embeddings[:train_num_samples]
            
        self.transforms = transforms
        self.tokenize = tokenizer
        self.pro = pro
    def loadList(self,file_path):
        # the filename should mention the extension '.npy'
        tempNumpyArray=np.load(file_path, allow_pickle=True)
        return tempNumpyArray.tolist()
    def __len__(self):
        return len(self.samples)
    
    def validation(self, caption, valid_caption):
        print(caption)
        if caption == "###":
            valid_caption.append(0)
        else:
            valid_caption.append(1)
            
    def __getitem__(self,index):
        ###
        if self.pro:
            captions=torch.stack([self.tokenize([str(self.samples[index]['caption'])])[0],  # 0
            self.tokenize([str(self.samples[index]['obj_relation_aug_caption'])])[0],     # 1
            self.tokenize([str(self.samples[index]['adj_relation_aug_caption'])])[0],     # 2
            self.tokenize([str(self.samples[index]['adj_aug_caption'])])[0],                # 3
            self.tokenize([str(self.samples[index]['noun_aug_caption'])])[0],               # 4
            self.tokenize([str(self.samples[index]['verb_aug_caption'])])[0],])               # 5          
            # self.tokenize([str(self.samples[index]['rel_aug_caption'])])[0]])             # 6
            
            if len(self.samples[index]['valid_caption']) == 7:
                # del self.samples[index]['valid_caption'][2]
                del self.samples[index]['valid_caption'][6] 
        else:            
            captions=torch.stack([self.tokenize([str(self.samples[index]['caption'])])[0],  
            self.tokenize([str(self.samples[index]['relation_aug_caption'])])[0],
            self.tokenize([str(self.samples[index]['adj_aug_caption'])])[0],                
            self.tokenize([str(self.samples[index]['noun_aug_caption'])])[0],               
            self.tokenize([str(self.samples[index]['verb_aug_caption'])])[0],])               

        if self.data=='coco':
            image_id=self.samples[index]['image_id']
            data_split='train2014' if self.split=='train' else "val2014"
            image_path=os.path.join(COCO_DATASET_ROOT, data_split, f"COCO_train2014_{'0'*(12-len(str(image_id)))}{image_id}.jpg")
            image = self.transforms(Image.open(image_path).convert("RGB"))
        else:
            image = self.transforms(Image.open(str(self.samples[index]['image_path'])).convert("RGB"))
        valid_caption_mask=torch.tensor(self.samples[index]['valid_caption'])
        
        return image, captions, valid_caption_mask[1:]


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None):
        logging.debug(f'Loading json data from {input_filename}.')
        with open(input_filename, 'r') as f:
            self.samples = json.load(f)
        self.transforms = transforms
        self.tokenize = tokenizer
        
        # self.all_features = {}
        # dist_dir = "/home/sunyw/rofclip/datasets/coco2014/siglip_llava_ov_384_14_1152"
        # files = os.listdir(dist_dir)
        # for file in files:
        #     features = torch.load(os.path.join(dist_dir,file))
        #     self.all_features.update(features)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        # get image
        image_id = self.samples[index]["image_id"]
        data_split='train2014'
        image_path=os.path.join(COCO_DATASET_ROOT, data_split, f"COCO_train2014_{'0'*(12-len(str(image_id)))}{image_id}.jpg")
        image = self.transforms(Image.open(image_path).convert("RGB"))
        # feature = torch.ones((1,1)) #self.all_features[image_id]
        # get captions
        captions = torch.stack([
            self.tokenize([str(self.samples[index]['caption'])])[0], 
            self.tokenize([str(self.samples[index]['expand'])])[0], 
            self.tokenize([str(self.samples[index]['obj_rep'])])[0],
            self.tokenize([str(self.samples[index]['att_rep'])])[0],
            self.tokenize([str(self.samples[index]['rel_rep'])])[0],
            self.tokenize([str(self.samples[index]['obj_add'])])[0],
            self.tokenize([str(self.samples[index]['att_add'])])[0],
            self.tokenize([str(self.samples[index]['rel_add'])])[0],
            self.tokenize([str(self.samples[index]['obj_swap'])])[0],
            self.tokenize([str(self.samples[index]['att_swap'])])[0]  
        ]
        )
        # idxs = random.sample([1,2,3,4,5,6,7,8],1)
        # for idx in idxs:
        #     captions[idx] = self.tokenize([str(self.samples[index]['caption'])])[0]
        return image, captions
            
class Collate4JsonD:
    def __call__(self, batch):
        images = torch.stack([example[0] for example in batch])
        true_captions = torch.stack([example[1][0] for example in batch])
        dense_captions = torch.stack([example[1][1] for example in batch])
        hard_negatives = torch.cat([example[1][2:] for example in batch])
        texts = torch.cat([true_captions, dense_captions, hard_negatives])
        # features = torch.stack([example[2] for example in batch])
        return images, texts

class HardNegative_Collate:
    def __call__(self,batch):
        img=torch.stack([example[0] for example in batch])
        ture_caption=torch.stack([example[1][0] for example in batch])
        hard_negative=torch.cat([example[1][1:] for example in batch])
        text=torch.cat([ture_caption,hard_negative])
        valid_caption_mask=torch.stack([example[2] for example in batch])
        # print(text.shape, valid_caption_mask.shape)
        return img,text,valid_caption_mask

class Collate4NegCsv:
    def __call__(self, batch):
        images = torch.cat([example[0] for example in batch])
        # print(batch[0][1].shape, batch[0][0].shape)
        true_captions = torch.cat([example[1][:2] for example in batch])
        hard_negatives = torch.cat([example[1][2:] for example in batch])
        texts = torch.cat([true_captions, hard_negatives])
        # features = torch.stack([example[2] for example in batch])
        return images, texts
        
class GnmDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None):
        logging.debug(f'Loading json data from {input_filename}.')
        with open(input_filename, 'r') as f:
            self.samples = json.load(f)
        self.samples = self.samples.values()
        self.transforms = transforms
        self.tokenize = tokenizer
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        # get image
        image_id = self.samples[index]["image_id"]
        image_id = image_id.split('_')[-1]
        data_split='train2014'
        image_path=os.path.join(GNM_DATASET_ROOT, data_split, f"COCO_train2014_{'0'*(12-len(str(image_id)))}/img.jpg")
        image = self.transforms(Image.open(image_path).convert("RGB"))
        neg_images = self.samples[index]["neg_images"]
        neg_captions = self.samples[index]["neg_captions"]
        idxs = random.sample([i for i in range(len(neg_images))], 3)
        neg_images = neg_images[idxs]
        neg_captions = neg_captions[idxs]
        
        
        # get captions
        captions = torch.stack([
            self.tokenize([str(self.samples[index]['caption'])])[0],
            self.tokenize([str(self.samples[index]['expand'])])[0], 
            self.tokenize([str(self.samples[index]['obj_rep'])])[0],
            self.tokenize([str(self.samples[index]['att_rep'])])[0],
            self.tokenize([str(self.samples[index]['rel_rep'])])[0],
            self.tokenize([str(self.samples[index]['obj_add'])])[0],
            self.tokenize([str(self.samples[index]['att_add'])])[0],
            self.tokenize([str(self.samples[index]['rel_add'])])[0]  
        ]
        )
        
        return image, captions
    
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_negcsv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = NegCsvDataset(
        input_filename,
        preprocess_fn,
        img_key="filepath",
        caption_key="title",
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    # dataset = dataset[:25000]
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=Collate4NegCsv()
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_npy_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = NpyDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=HardNegative_Collate()
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_json_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset(
        input_filename,
        preprocess_fn,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=Collate4JsonD()
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    elif dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "negcsv":
        return get_negcsv_dataset
    elif dataset_type == "npy":
        return get_npy_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    
def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
