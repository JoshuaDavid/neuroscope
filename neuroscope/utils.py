# %%
import einops
import re
import os
from pathlib import Path
import huggingface_hub
import torch
import json
import numpy as np
import plotly.express as px
import logging
import shutil
import pprint
import argparse
import datasets
from typing import Tuple, Union
from IPython import get_ipython
from transformer_lens.utils import get_corner
from transformer_lens import HookedTransformer
from functools import lru_cache
import transformer_lens
from dataclasses import dataclass
from typing import *
from .config import (
    CACHE_DIR,
    REPO_ROOT,
    OLD_CHECKPOINT_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
)

def download_file_from_hf(repo_name, file_name, subfolder=".", cache_dir=CACHE_DIR):
    file_path = huggingface_hub.hf_hub_download(
        repo_id=f"NeelNanda/{repo_name}",
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
    )
    print(f"Saved at file_path: {file_path}")
    if file_path.endswith(".pth"):
        return torch.load(file_path)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


# %%
def model_name_to_data_name(model_name):
    if "old" in model_name or "pile" in model_name:
        data_name = "pile"
    elif "pythia" in model_name:
        data_name = "pile-big"
    elif "gpt" in model_name:
        data_name = "openwebtext"
    elif model_name.startswith("solu") or model_name.startswith("gelu"):
        data_name = "c4-code"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return data_name

# %%

def array_to_trunc_floats(array: np.ndarray, decimal_places: int = 6):
    return array.round(decimal_places).tolist()

# %%
def push_to_hub(repo_dir):
    """Pushes a directory/repo to HuggingFace Hub

    Args:
        repo_dir (str or Repository): The directory of the relevant repo
    """
    if isinstance(repo_dir, huggingface_hub.Repository):
        repo_dir = repo_dir.repo_dir
    # -C means "run command as though you were in that directory"
    # Uses explicit git calls on CLI which is way faster than HuggingFace's Python API for some reason
    os.system(f"git -C {repo_dir} add .")
    os.system(f"git -C {repo_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {repo_dir} push")

def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(str(repo_folder), repo_url)

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)



# %%
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


# %%
class TokenDatasetWrapper:
    """
    A wrapper around a HuggingFace Dataset which allows the slicing syntax (ie dataset[4], dataset[4:8], dataset[[5, 1, 7, 8]] etc.)
    Used to allow for uint16 datasets which are Torch incompatible (used for the Pile), but consume half the space!.
    Explicitly used for datasets of tokens
    """

    def __init__(self, dataset):
        if isinstance(dataset, datasets.Dataset):
            self.dataset = dataset
        elif isinstance(dataset, TokenDatasetWrapper):
            self.dataset = dataset.dataset
        elif isinstance(dataset, datasets.DatasetDict):
            self.dataset = dataset["train"]
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        self.is_unint16 = self.dataset.features["tokens"].feature.dtype == "uint16"
        if self.is_unint16:
            self.dataset = self.dataset.with_format("numpy")
        else:
            self.dataset = self.dataset.with_format("torch")
        
    def __getitem__(self, idx) -> torch.Tensor:
        tokens = self.dataset[idx]['tokens']
        if self.is_unint16:
            tokens = tokens.astype(np.int32)
            return torch.tensor(tokens)
        else:
            return tokens
    
    def __len__(self):
        return len(self.dataset)

LOCAL_DATASET_NAMES = {
    "c4-code": "c4_code_valid_tokens.hf",
    "c4": "c4_valid_tokens.hf",
    "code": "code_valid_tokens.hf",
    "pile": "pile_big_int32.hf",
    "pile-big": "pile_big_int32.hf",
    "pile-big-uint16": "pile_big_int16.hf",
    "openwebtext": "openwebtext_tokens.hf",
}
REMOTE_DATASET_NAMES = {
    "c4": "NeelNanda/c4-tokenized-2b",
    "code": "NeelNanda/code-tokenized",
    "pile": "NeelNanda/pile-small-tokenized-2b",
    "pile-small": "NeelNanda/pile-small-tokenized-2b",
    "pile-big": "NeelNanda/pile-tokenized-10b",
    "pile-big-uint16": "NeelNanda/pile-tokenized-10b",
    "openwebtext": "NeelNanda/openwebtext-tokenized-9b",
}

@lru_cache(maxsize=None)
def get_dataset(dataset_name: str, local=False) -> TokenDatasetWrapper:
    """Loads in one of the model datasets over which we take the max act examples. If local, loads from local folder, otherwise loads from HuggingFace Hub

    Args:
        dataset_name (str): Name of the dataset, must be one of the entries in the dictionary
        local (bool, optional): Whether to load from a local folder or remotely. Defaults to False.

    Returns:
        datasets.Dataset: _description_
    
    Test:
        for name in ["c4", "code", "pile", "openwebtext", "pile-big", "c4-code"]:
            code_remote = nutils.get_dataset(name, local=False)
            code_local = nutils.get_dataset(name, local=True)
            a = torch.randint(0, len(code_remote), (100,))
            rtokens = code_remote[a]
            ltokens = code_local[a]
            try:
                assert len(code_remote) == len(code_local)
                assert (rtokens==ltokens).all()
                print("Success", name)
            except:
                print("Failed", name)
    """
    if local:
        tokens = datasets.load_from_disk(os.path.join(
            DATA_DIR, LOCAL_DATASET_NAMES[dataset_name]))
        tokens = tokens.with_format("torch")
        return TokenDatasetWrapper(tokens)
    else:
        if dataset_name=="c4-code":
            c4_data = datasets.load_dataset(REMOTE_DATASET_NAMES["c4"], split="train")
            code_data = datasets.load_dataset(REMOTE_DATASET_NAMES["code"], split="train")
            tokens = datasets.concatenate_datasets([c4_data, code_data])
        else:
            tokens = datasets.load_dataset(REMOTE_DATASET_NAMES[dataset_name], split="train")
        tokens = tokens.with_format("torch")
        return TokenDatasetWrapper(tokens)

def get_dataset_with_local_cache(dataset_name: str):
    try:
        return get_dataset(dataset_name, local=True)
    except:
        print(f"Failed to load dataset {dataset_name} from local disk. Fetching from remote.")
        dataset_wrapper = get_dataset(dataset_name, local=False)
        print("Got dataset {dataset_name}")
        dataset_wrapper.dataset.save_to_disk(
            os.path.join(DATA_DIR, LOCAL_DATASET_NAMES[dataset_name]))
        return dataset_wrapper

class MaxStore:
    """Used to calculate max activating dataset examples - takes in batches of activations repeatedly, and tracks the top_k examples activations + indexes"""

    def __init__(self, top_k, length, device="cuda"):
        self.top_k = top_k
        self.length = length
        self.device = device

        self.max = -torch.inf * torch.ones(
            (top_k, length), dtype=torch.float32, device=device
        )
        self.index = -torch.ones((top_k, length), dtype=torch.long, device=device)

        self.counter = 0
        self.total_updates = 0
        self.batches_seen = 0

    def update(self, new_act, new_index):
        min_max_act, min_indices = self.max.min(dim=0)
        mask = min_max_act < new_act
        num_updates = mask.sum().item()
        self.max[min_indices[mask], mask] = new_act[mask]
        self.index[min_indices[mask], mask] = new_index[mask]
        self.total_updates += num_updates
        return num_updates

    def batch_update(self, activations, text_indices=None):
        """
        activations: Shape [batch, length]
        text_indices: Shape [batch,]

        activations is the largest MLP activation, text_indices is the index of the text strings.

        Sorts the activations into descending order, then updates with each column until we stop needing to update
        """
        batch_size = activations.size(0)
        new_acts, sorted_indices = activations.sort(0, descending=True)
        if text_indices is None:
            text_indices = torch.arange(
                self.counter,
                self.counter + batch_size,
                device=self.device,
                dtype=torch.int64,
            )
        new_indices = text_indices[sorted_indices]
        for i in range(batch_size):
            num_updates = self.update(new_acts[i], new_indices[i])
            if num_updates == 0:
                break
        self.counter += batch_size
        self.batches_seen += 1

    def save(self, dir, folder_name=None):
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir
        path.mkdir(exist_ok=True)
        torch.save(self.max, path / "max.pth")
        torch.save(self.index, path / "index.pth")
        with open(path / "config.json", "w") as f:
            filt_dict = {
                k: v for k, v in self.__dict__.items() if k not in ["max", "index"]
            }
            json.dump(filt_dict, f)
        print("Saved Max Store to:", path)

    def switch_to_inference(self):
        """Switch from updating mode to inference - move to the CPU and sort by max act."""
        self.max = self.max.cpu()
        self.index = self.index.cpu()
        self.max, indices = self.max.sort(dim=0, descending=True)
        self.index = self.index.gather(0, indices)

    @classmethod
    def load(cls, dir, folder_name=None, continue_updating=False, transpose=False):
        dir = Path(dir)
        if folder_name is not None:
            path = dir / folder_name
        else:
            path = dir

        max = torch.load(path / "max.pth")
        index = torch.load(path / "index.pth")
        if transpose:
            max = max.T
            index = index.T
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        mas = cls(config["top_k"], config["length"])
        for k, v in config.items():
            mas.__dict__[k] = v
        mas.max = max
        mas.index = index
        if not continue_updating:
            mas.switch_to_inference()
        return mas

    def __repr__(self):
        return f"MaxStore(top_k={self.top_k}, length={self.length}, counter={self.counter}, total_updates={self.total_updates}, device={self.device})\n Max Values: {get_corner(self.max)}\n Indices: {get_corner(self.index)}"

# %%

def model_name_to_data_name(model_name):
    if "old" in model_name or "pile" in model_name:
        data_name = "pile"
    elif "pythia" in model_name:
        data_name = "pile-big"
    elif "gpt2" in model_name:
        data_name = "openwebtext"
    elif model_name.startswith("solu") or model_name.startswith("gelu") or model_name.startswith("attn-only"):
        # Note that solu-{}l-pile will go into the first set!
        data_name = "c4-code"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return data_name

def model_name_to_fancy_data_name(model_name):
    fancy_data_names = {
        "c4-code": "80% C4 (Web Text) and 20% Python Code",
        "c4": "C4 (Web Text)",
        "code": "Python Code",
        "pile": "The Pile",
        "pile-big": "The Pile",
        "pile-small": "The Pile",
        "openwebtext": "Open Web Text",
    }
    data_name = model_name_to_data_name(model_name)
    return fancy_data_names[data_name]

def get_fancy_model_name(model_name):
    cfg = transformer_lens.loading.get_pretrained_model_config(model_name)
    if cfg.act_fn in ["solu", "solu_ln"]:
        return f"SoLU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    elif "gelu" in cfg.act_fn:
        if cfg.original_architecture == "neel":
            return f"GELU Model: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
        elif cfg.original_architecture == "GPT2LMHeadModel":
            return f"GPT-2 {model_name.split('-')[-1].capitalize()}: {cfg.n_layers} Layers, {cfg.d_mlp} Neurons per Layer"
    else:
        raise ValueError(f"{model_name} Invalid Model Name for fancy model name")
    
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
@dataclass
class Config:
    model_name: str = "solu-1l"
    data_name: str = "c4"
    max_tokens: int = -1
    debug: bool = False
    batch_size: int = 8
    version: int = 3
    overwrite: bool = False
    use_pred_log_probs: bool = False
    use_max_neuron_act: bool = False
    use_neuron_logit_attr: bool = False
    use_head_logit_attr: bool = False
    use_activation_stats: bool = False
    neuron_top_k: int = 20
    head_top_k: int = 200

    def __post_init__(self):
        if "attn-only" in self.model_name:
            self.use_max_neuron_act = False
            self.use_neuron_logit_attr = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `Config` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def __get_item__(self, string):
        return self.__dict__[string]

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "Config:\n" + pprint.pformat(self.to_dict())

class PredLogProbs:
    def __init__(self, cfg: Config, model: HookedTransformer):
        self.cfg = cfg
        self.debug = self.cfg.debug
        if self.debug:
            self.base_dir = Path("/workspace/solu_outputs/debug/full_pred_log_probs") / cfg.data_name / cfg.model_name  # type: ignore
        else:
            self.base_dir = Path("/workspace/solu_outputs/full_pred_log_probs") / cfg.data_name / cfg.model_name  # type: ignore
        self.base_dir.mkdir(exist_ok=True, parents=True)

        if self.debug:
            self.save_dir = self.base_dir
        else:
            self.save_dir = self.base_dir / f"v{self.cfg.version}"
            assert (not self.cfg.overwrite) or (
                not self.save_dir.exists()
            ), f"Trying to overwrite existing dir: {self.save_dir}"

        self.cpu_plps = []
        self.gpu_plps = []
        self.max_gpu_len = 100

        self.model = model

    def step(self, logits, tokens):
        pred_log_probs = self.model.loss_fn(logits, tokens, per_token=True)
        self.gpu_plps.append(pred_log_probs.detach())
        if len(self.gpu_plps) > self.max_gpu_len:
            self.cpu_plps.append(torch.cat(self.gpu_plps, dim=0).detach().cpu())
            del self.gpu_plps
            self.gpu_plps = []

    def save(self):
        if self.gpu_plps:
            self.cpu_plps.append(torch.cat(self.gpu_plps, dim=0).detach().cpu())
        self.save_dir.mkdir(exist_ok=True)
        final_out = torch.cat(self.cpu_plps, dim=0)
        out_path = self.save_dir / "pred_log_probs.pth"
        torch.save(final_out, out_path)
        print("Saved Pred Log Probs to:", out_path)

    def log(self) -> dict:
        return {}

    def finish(self):
        self.save()


class BaseMaxTracker:
    def __init__(self, cfg: Config, model: HookedTransformer, name: str):
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.model = model
        self.name = name

        if self.debug:
            self.base_dir = Path(f"/workspace/solu_outputs/debug/{name}") / cfg.data_name / cfg.model_name  # type: ignore
        else:
            self.base_dir = Path(f"/workspace/solu_outputs/{name}") / cfg.data_name / cfg.model_name  # type: ignore
        self.base_dir.mkdir(exist_ok=True, parents=True)

        if self.debug:
            self.save_dir = self.base_dir
        else:
            self.save_dir = self.base_dir / f"v{self.cfg.version}"
            assert (not self.cfg.overwrite) or (
                not self.save_dir.exists()
            ), f"Trying to overwrite existing dir: {self.save_dir}"

    def step(self, logits, tokens):
        pass

    def save(self):
        raise NotImplementedError

    def log(self) -> dict:
        return {}

    def finish(self):
        self.save()

class NeuronMaxAct(BaseMaxTracker):
    def __init__(self, cfg: Config, model: HookedTransformer):
        super().__init__(cfg, model, name="neuron_max_act")

        self.stores = []
        for layer in range(self.model.cfg.n_layers):
            store = sutils.MaxStore(self.cfg.neuron_top_k, self.model.cfg.d_mlp)
            self.stores.append(store)

            def update_max_act_hook(neuron_acts, hook, store):
                store.batch_update(
                    einops.reduce(neuron_acts, "batch pos d_mlp -> batch d_mlp", "max")
                )

            if self.model.cfg.act_fn == "solu_ln":
                hook_fn = partial(update_max_act_hook, store=store)
                self.model.blocks[layer].mlp.hook_mid.add_hook(hook_fn)
            elif self.model.cfg.act_fn in ["gelu", "relu", "gelu_new"]:
                hook_fn = partial(update_max_act_hook, store=store)
                self.model.blocks[layer].mlp.hook_post.add_hook(hook_fn)
            else:
                raise ValueError(f"Invalid Act Fn: {self.model.cfg.act_fn}")

    def save(self):
        self.save_dir.mkdir(exist_ok=True)
        for layer, store in enumerate(self.stores):
            store.save(folder_name=str(layer), dir=self.save_dir)
        print(f"Saved {self.name} stores to:", self.save_dir)


class HeadLogitAttr(BaseMaxTracker):
    """Stores the max positive and max negative contribution from each head to the correct logit"""

    def __init__(self, cfg: Config, model: HookedTransformer):
        super().__init__(cfg, model, name="head_logit_attr")

        self.W_OU = einsum(
            "layer head_index d_head d_model, d_model d_vocab -> layer head_index d_head d_vocab",
            self.model.W_O,
            self.model.W_U,
        )

        self.head_zs = [None] * self.model.cfg.n_layers

        self.pos_store = sutils.MaxStore(
            self.cfg.head_top_k, self.model.cfg.n_heads * self.model.cfg.n_layers
        )
        self.neg_store = sutils.MaxStore(
            self.cfg.head_top_k, self.model.cfg.n_heads * self.model.cfg.n_layers
        )

        def cache_z_hook(z, hook, layer, head_zs):
            head_zs[layer] = z.detach()

        for layer in range(self.model.cfg.n_layers):
            self.model.blocks[layer].attn.hook_z.add_hook(
                partial(cache_z_hook, layer=layer, head_zs=self.head_zs)
            )

        self.ln_scale_cache = {}

        def cache_ln_scale_hook(ln_scale, hook, cache):
            cache["ln_scale"] = ln_scale.detach()

        self.model.ln_final.hook_scale.add_hook(
            partial(cache_ln_scale_hook, cache=self.ln_scale_cache)
        )

    def step(self, logits, tokens):
        weights_to_true_logit = self.W_OU[..., tokens]
        weights_to_true_logit = einops.rearrange(
            weights_to_true_logit,
            "layer head_index d_head batch pos -> batch pos (layer head_index) d_head",
        )

        # Same shape as weights_to_true_logit
        cached_z = torch.cat(self.head_zs, dim=-2)
        cached_ln_scale = self.ln_scale_cache["ln_scale"]

        head_to_true_logit = einops.reduce(
            cached_z * weights_to_true_logit,
            "batch pos component d_head -> batch pos component",
            "sum",
        )
        head_to_true_logit = head_to_true_logit / cached_ln_scale

        max_head_to_true_logit = einops.reduce(
            head_to_true_logit, "batch pos component -> batch component", "max"
        )
        self.pos_store.batch_update(max_head_to_true_logit)

        min_head_to_true_logit = einops.reduce(
            head_to_true_logit, "batch pos component -> batch component", "min"
        )
        self.neg_store.batch_update(-min_head_to_true_logit)

    def save(self):
        self.save_dir.mkdir(exist_ok=True)
        self.pos_store.save(self.save_dir, "pos")
        self.neg_store.save(self.save_dir, "neg")


class NeuronLogitAttr(BaseMaxTracker):
    """Stores the max direct contribution from each neuron to the correct logit."""

    def __init__(self, cfg: Config, model: HookedTransformer):
        super().__init__(cfg, model, name="neuron_logit_attr")

        self.W_out_U = einsum(
            "layer d_mlp d_model, d_model d_vocab -> layer d_mlp d_vocab",
            self.model.W_out,
            self.model.W_U,
        )

        self.cache = {}

        def cache_neuron_post_hook(act_pos, hook, layer, cache):
            cache[f"post_{layer}"] = act_pos.detach()

        self.stores = []
        for layer in range(self.model.cfg.n_layers):
            self.stores.append(
                sutils.MaxStore(self.cfg.neuron_top_k, self.model.cfg.d_mlp)
            )
            # hook_post means the post MLP hook in both gelu & solu
            self.model.blocks[layer].mlp.hook_post.add_hook(
                partial(cache_neuron_post_hook, layer=layer, cache=self.cache)
            )

        def cache_ln_scale_hook(ln_scale, hook, cache):
            cache["ln_scale"] = ln_scale.detach()

        self.model.ln_final.hook_scale.add_hook(
            partial(cache_ln_scale_hook, cache=self.cache)
        )

    def step(self, logits, tokens):
        weights_to_true_logit = self.W_out_U[..., tokens]
        weights_to_true_logit = einops.rearrange(
            weights_to_true_logit, "layer d_mlp batch pos -> layer batch pos d_mlp"
        )

        cached_ln_scale = self.cache["ln_scale"]
        for layer in range(self.model.cfg.n_layers):
            neuron_post = self.cache[f"post_{layer}"]
            weights = weights_to_true_logit[layer]
            neuron_logit_attr = weights * neuron_post
            scaled_neuron_logit_attr = neuron_logit_attr / cached_ln_scale
            max_logit_attr = einops.reduce(
                scaled_neuron_logit_attr, "batch pos d_mlp -> batch d_mlp", "max"
            )
            self.stores[layer].batch_update(max_logit_attr)

    def save(self):
        self.save_dir.mkdir(exist_ok=True)
        for layer, store in enumerate(self.stores):
            store.save(folder_name=str(layer), dir=self.save_dir)
        print(f"Saved {self.name} stores to:", self.save_dir)


class ActivationStats:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.name = "activation_stats"

        self.debug = self.cfg.debug
        if self.debug:
            self.base_dir = Path(f"/workspace/solu_outputs/debug/{self.name}") / cfg.data_name / cfg.model_name  # type: ignore
        else:
            self.base_dir = Path(f"/workspace/solu_outputs/{self.name}") / cfg.data_name / cfg.model_name  # type: ignore
        self.base_dir.mkdir(exist_ok=True, parents=True)

        if self.debug:
            self.save_dir = self.base_dir
        else:
            self.save_dir = self.base_dir / f"v{self.cfg.version}"
            assert (not self.cfg.overwrite) or (
                not self.save_dir.exists()
            ), f"Trying to overwrite existing dir: {self.save_dir}"

        self.mean_cache = {}
        self.sq_cache = {}

        def caching_hook(act, hook):
            self.mean_cache[hook.name] = act.mean(0)
            self.sq_cache[hook.name] = act.pow(2).mean(0)

        for hook_point in model.hook_dict.values():
            hook_point.add_hook(caching_hook)

    def step(self, logits, tokens):
        pass

    def save(self):
        self.save_dir.mkdir(exist_ok=True)
        torch.save(self.mean_cache, self.save_dir / "mean_act.pth")
        torch.save(self.sq_cache, self.save_dir / "sqaure_act.pth")
        print("Saved activation stats to:", self.save_dir)

    def finish(self):
        self.save()
