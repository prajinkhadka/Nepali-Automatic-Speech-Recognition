import os
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric, Dataset,concatenate_datasets, set_caching_enabled, ClassLabel
import pandas as pd

import random
from IPython.display import display, HTML

import json
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2ForCTC,Wav2Vec2Processor,Trainer,TrainingArguments,Wav2Vec2FeatureExtractor

import re
set_caching_enabled(False)

import soundfile as sf
import torchaudio


import IPython.display as ipd

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm
import torch

# Set environment variables
os.environ['WANDB_DISABLED '] = 'True'

import warnings
warnings.filterwarnings('ignore')

import logging
import transformers
transformers.logging.get_verbosity = lambda: logging.NOTSET
transformers.logging.get_verbosity()

import datasets
datasets.logging.get_verbosity = lambda: logging.NOTSET

import glob 
data_dir  = "/workspace/data_processed/*"
data_dir_list = glob.glob(data_dir)
data_dir_list.remove('/workspace/data_processed/5')

data_set_list_total = []
for i in range(len(data_dir_list)):
    common_voice_val = Dataset.load_from_disk(data_dir_list[i])
    data_set_list_total.append(common_voice_val)


common_voice_train = concatenate_datasets(data_set_list_total[:-1])

common_voice_test = concatenate_datasets(data_set_list_total[-1:])

from huggingface_hub import notebook_login
notebook_login()
repo_name = "wav2vec2-large-xlsr-300m-nepali"

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali")

model = Wav2Vec2ForCTC.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali")

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print(len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali")
model.freeze_feature_extractor()
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=2,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=5e-5,
  warmup_steps=1000,
  save_total_limit=2,
  push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

print("Statr training")
trainer.train()
trainer.push_to_hub()
