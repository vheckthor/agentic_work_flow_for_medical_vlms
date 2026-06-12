"""Implemented by me"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_answer_vocab(train_samples: List[dict]) -> Tuple[Dict[str, int], List[str]]:
    answers = sorted({s["answer"] for s in train_samples})
    a2i = {a: i for i, a in enumerate(answers)}
    return a2i, answers

#This class is used to load the SLAKE dataset, it is a custom dataset class that inherits from the Dataset class andd prepares the data for the model

class SlakeVQADataset(Dataset):
    """Loads SLAKE samples; labels are indices into the train answer vocabulary."""

    def __init__(
        self,
        json_path: str,
        imgs_root: str,
        answer_to_idx: Dict[str, int],
        tokenizer,
        image_transform,
        split: str = "train",
        skip_unknown_answers: bool = False,
        max_question_length: int = 64,
        clip_processor: Optional[Any] = None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.samples: List[dict] = []
        self.imgs_root = imgs_root
        self.answer_to_idx = answer_to_idx
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_question_length = max_question_length
        self.clip_processor = clip_processor

        for row in raw:
            ans = row["answer"]
            path = os.path.join(imgs_root, row["img_name"])
            if not os.path.isfile(path):
                continue
            if ans not in answer_to_idx:
                if skip_unknown_answers:
                    continue
                label = -1
            else:
                label = answer_to_idx[ans]
            self.samples.append(
                {
                    "path": path,
                    "question": row["question"],
                    "label": label,
                    "answer": ans,
                }
            )
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {json_path} (check paths and vocab).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.clip_processor is not None:
            proc = self.clip_processor(
                text=[s["question"]],
                images=img,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            return {
                "pixel_values": proc["pixel_values"].squeeze(0),
                "input_ids": proc["input_ids"].squeeze(0),
                "attention_mask": proc["attention_mask"].squeeze(0),
                "labels": torch.tensor(s["label"], dtype=torch.long),
            }
        img = self.image_transform(img)
        enc = self.tokenizer(
            s["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": img,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }


def default_train_transform(image_size: int = 224, augment: bool = False):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05)),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def default_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
