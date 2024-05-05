import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    sub_data: str = "",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        elif sub_data == "":
            dataset = load_dataset(data, split=split)
        else:
            dataset = load_dataset(data, sub_data, split=split)
        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        print(sample.size())
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    print(cat_samples.size())
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def custom_multimodal_dataset(
    data_dict: dict,


):
    '''
    第一列文本prompt,第二列图像地址暂时是本地的,自己shuffle
    
    from PIL import Image
    samples = []
    n_run = 0
    for data in data_list:

        line = data[0].strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        line_encoded = torch.tensor([line_encoded])
        if line_encoded.numel() == 0:
            continue
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    r=[
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
    print('data pre fin')
    '''
    return data_dict
    