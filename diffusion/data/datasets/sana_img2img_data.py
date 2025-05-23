# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import random

import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset

from diffusion.data.builder import DATASETS, get_data_path
from diffusion.utils.logger import get_root_logger


@DATASETS.register_module()
class SanaImg2ImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="",
        transform=None,
        resolution=256,
        load_vae_feat=False,
        config=None,
        img_extension=".png",
        **kwargs,
    ):
        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.resolution = resolution
        self.img_extension = img_extension

        self.data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        self.dataset = []
        
        # Load image pairs from metadata
        for data_dir in self.data_dirs:
            metadata_path = osp.join(data_dir, "metadata.jsonl")
            
            if osp.exists(metadata_path):
                # Load from metadata.jsonl format
                with open(metadata_path, 'r') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        source_path = osp.join(data_dir, item["source"])
                        target_path = osp.join(data_dir, item["target"])
                        self.dataset.append({
                            "source": source_path,
                            "target": target_path
                        })
            else:
                # Load from directory structure: source/ and target/
                source_dir = osp.join(data_dir, "source")
                target_dir = osp.join(data_dir, "target")
                
                if osp.exists(source_dir) and osp.exists(target_dir):
                    # Get matching files from source and target directories
                    source_files = sorted([f for f in os.listdir(source_dir) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
                    target_files = sorted([f for f in os.listdir(target_dir) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
                    
                    # Match source and target files by filename
                    for source_file in source_files:
                        if source_file in target_files:
                            source_path = osp.join(source_dir, source_file)
                            target_path = osp.join(target_dir, source_file)
                            self.dataset.append({
                                "source": source_path,
                                "target": target_path
                            })
                else:
                    self.logger.warning(f"No valid image pair data found in {data_dir}")

        if len(self.dataset) == 0:
            self.logger.error("No image pairs found! Please check your data directory structure.")
            raise ValueError("Empty dataset")

        # For toy dataset testing, repeat the dataset
        if len(self.dataset) < 100:
            original_len = len(self.dataset)
            repeat_times = max(1, 100 // original_len)
            self.dataset = self.dataset * repeat_times
            self.logger.info(colored(f"Dataset is repeated {repeat_times} times for toy dataset (original: {original_len})", "red", attrs=["bold"]))
        
        self.ori_imgs_nums = len(self)
        self.logger.info(f"Image2Image Dataset samples: {len(self.dataset)}")
        self.logger.info("Dataset format: Source image -> Target image (no text conditioning)")

    def getdata(self, idx):
        data_item = self.dataset[idx]
        source_path = data_item["source"]
        target_path = data_item["target"]
        
        # Load source and target images
        try:
            source_img = Image.open(source_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Error loading images: {source_path}, {target_path}. Error: {e}")
            raise e

        # Apply transforms
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        # Create data info (similar to original dataset)
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        return (
            source_img,    # Source image (condition)
            target_img,    # Target image (ground truth)
            data_info,     # Data info
            idx,           # Index
            os.path.basename(source_path),  # Source filename
            os.path.basename(target_path),  # Target filename
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                self.logger.warning(f"Error loading data at index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.dataset)
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)


@DATASETS.register_module()
class SanaImg2ImgWebDataset(torch.utils.data.Dataset):
    """
    WebDataset version for Image-to-Image training with tar files
    Expected tar structure:
    - source images: {key}.source.jpg/png
    - target images: {key}.target.jpg/png
    - metadata: {key}.json (optional)
    """
    def __init__(
        self,
        data_dir="",
        meta_path=None,
        cache_dir="/cache/data/sana-img2img-webds-meta",
        max_shards_to_load=None,
        transform=None,
        resolution=256,
        load_vae_feat=False,
        config=None,
        sort_dataset=False,
        num_replicas=None,
        **kwargs,
    ):
        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.resolution = resolution

        # Initialize similar to SanaWebDataset but for image pairs
        data_dirs = data_dir if isinstance(data_dir, list) else [data_dir]
        meta_paths = meta_path if isinstance(meta_path, list) else [meta_path] * len(data_dirs)
        self.meta_paths = []
        
        for data_path, meta_path in zip(data_dirs, meta_paths):
            self.data_path = osp.expanduser(data_path)
            self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None

            _local_meta_path = osp.join(self.data_path, "wids-meta.json")
            if meta_path is None and osp.exists(_local_meta_path):
                self.logger.info(f"loading from {_local_meta_path}")
                self.meta_path = meta_path = _local_meta_path

            if meta_path is None:
                self.meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".img2img.wdsmeta.json",
                )

            assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
            self.logger.info(f"[Img2Img] Loading meta information {self.meta_path}")
            self.meta_paths.append(self.meta_path)

        self._initialize_dataset(num_replicas, sort_dataset)
        self.logger.info("Image2Image WebDataset initialized")
        self.logger.warning(f"Sort the dataset: {sort_dataset}")

    def _initialize_dataset(self, num_replicas, sort_dataset):
        from diffusion.data.wids import ShardListDatasetMulti, ShardListDataset
        import hashlib
        import getpass
        import torch.distributed as dist

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        if len(self.meta_paths) > 0:
            self.dataset = ShardListDatasetMulti(
                self.meta_paths,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
                sort_data_inseq=sort_dataset,
                num_replicas=num_replicas or dist.get_world_size(),
            )
        else:
            self.dataset = ShardListDataset(
                self.meta_path,
                cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
            )
        self.ori_imgs_nums = len(self)
        self.logger.info(f"{self.dataset.data_info}")

    def getdata(self, idx):
        data = self.dataset[idx]
        self.key = data["__key__"]
        
        # Load source and target images
        source_img = None
        target_img = None
        
        # Try different extensions for source and target
        for ext in [".source.jpg", ".source.png", ".source.jpeg"]:
            if f"{ext}" in data:
                source_img = data[ext]
                break
        
        for ext in [".target.jpg", ".target.png", ".target.jpeg"]:
            if f"{ext}" in data:
                target_img = data[ext]
                break
        
        if source_img is None or target_img is None:
            raise ValueError(f"Missing source or target image for key: {self.key}")

        # Apply transforms
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)

        # Create data info
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        }

        dataindex_info = {
            "index": data["__index__"],
            "shard": "/".join(data["__shard__"].rsplit("/", 2)[-2:]),
            "shardindex": data["__shardindex__"],
        }

        return (
            source_img,      # Source image (condition)
            target_img,      # Target image (ground truth)
            data_info,       # Data info
            idx,             # Index
            self.key,        # Key
            dataindex_info,  # Shard info
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                self.logger.warning(f"Error loading data at index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.dataset)
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.dataset)

    def get_data_info(self, idx):
        try:
            data = self.dataset[idx]
            info = data.get(".json", {})
            key = data["__key__"]
            version = info.get("version", "img2img")
            return {"key": key, "version": version}
        except Exception as e:
            self.logger.warning(f"Error getting data info: {str(e)}")
            return None


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from diffusion.data.transforms import get_transform

    image_size = 512
    transform = get_transform("default_train", image_size)
    
    # Test local dataset
    train_dataset = SanaImg2ImgDataset(
        data_dir="asset/example_img2img_data",
        resolution=image_size,
        transform=transform,
        load_vae_feat=False,
    )
    
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)

    for data in dataloader:
        source_img, target_img, data_info, idx, source_name, target_name = data
        print(f"Source shape: {source_img.shape}, Target shape: {target_img.shape}")
        print(f"Data info: {data_info}")
        print(f"Source files: {source_name}, Target files: {target_name}")
        break 