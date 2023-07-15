# %%
from neel.imports import *
import solu.utils as sutils
from transformer_lens import HookedTransformer
import torch
from .config import (IN_IPYTHON)
from .utils import (
    Config,
    PredLogProbs,
    BaseMaxTracker,
    NeuronMaxAct,
    HeadLogitAttr,
    NeuronLogitAttr,
    ActivationStats,
    get_dataset_with_local_cache,
)

def scan_over_data(use_wandb=False, **cfg_kwargs):
    torch.set_grad_enabled(False)

    default_cfg = Config(**cfg_kwargs)

    if not IN_IPYTHON:
        print("Updating config")
        cfg = sutils.arg_parse_update_cfg(default_cfg.to_dict())
        cfg = Config.from_dict(cfg)
        print(cfg)
    else:
        print("In IPython, skipping config")
        new_config = {
            "debug": True,
            "use_activation_stats": True,
            "model_name":"gpt2-small",
            "data_name":"openwebtext"
        }
        cfg = dict(default_cfg.to_dict())
        cfg.update(new_config)
        cfg = Config.from_dict(cfg)
        cfg.debug = True

    if cfg.debug:
        cfg.max_tokens = int(1e6)
        cfg.batch_size = 2
    print(cfg)

    """
    Test:
    tens = torch.load("/workspace/solu_outputs/debug/full_pred_log_probs/code/solu-3l/pred_log_probs.pth")

    i = 870
    j = 532
    print(tens[i, j])
    model = HookedTransformer.from_pretrained("solu-3l")
    dataset, tokens_name = sutils.get_dataset("c4")
    tokens = dataset[i:i+1]['tokens'].cuda()
    with torch.autocast("cuda", torch.bfloat16):
        logits = model(tokens)
        plps = model.loss_fn(logits, tokens, per_token=True)
    print(plps[0, j])
    """

    if not cfg.debug:
        if use_wandb:
            wandb.init(config=cfg.to_dict())
    model = HookedTransformer.from_pretrained(cfg.model_name)  # type: ignore
    dataset = get_dataset_with_local_cache(cfg.data_name)

    if len(dataset) * model.cfg.n_ctx < cfg.max_tokens or cfg.max_tokens < 0:
        print("Resetting max tokens:", cfg.max_tokens, "to", len(dataset) * model.cfg.n_ctx)
        cfg.max_tokens = len(dataset) * model.cfg.n_ctx

    trackers = []
    if cfg.use_head_logit_attr:
        trackers.append(HeadLogitAttr(cfg, model))

    if cfg.use_max_neuron_act:
        trackers.append(NeuronMaxAct(cfg, model))

    if cfg.use_neuron_logit_attr:
        trackers.append(NeuronLogitAttr(cfg, model))

    if cfg.use_pred_log_probs:
        trackers.append(PredLogProbs(cfg, model))

    if cfg.use_activation_stats:
        trackers.append(ActivationStats(cfg, model))

    try:
        with torch.autocast("cuda", torch.bfloat16):
            for index in tqdm.tqdm(range(0, cfg.max_tokens // model.cfg.n_ctx, cfg.batch_size)):  # type: ignore
                tokens = dataset.dataset[index : index + cfg.batch_size]["tokens"].cuda()  # type: ignore
                logits = model(tokens).detach()
                for tracker in trackers:
                    tracker.step(logits, tokens)
                if not cfg.debug:
                    if use_wandb:
                        wandb.log({"tokens": index * model.cfg.n_ctx}, step=index)
    finally:
        for tracker in trackers:
            tracker.finish()
        if not cfg.debug:
            if use_wandb:
                wandb.finish()
