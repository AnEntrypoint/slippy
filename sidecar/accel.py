import torch


def xla_model():
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return None
    return xm


def is_xla(device):
    return getattr(device, "type", str(device)) == "xla"


def is_cuda(device):
    return getattr(device, "type", str(device)) == "cuda"


def pick_device(accelerator):
    if accelerator == "tpu":
        xm = xla_model()
        if xm is None:
            raise RuntimeError("accelerator=tpu but torch_xla is not importable")
        return xm.xla_device()
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator == "gpu":
        raise RuntimeError("accelerator=gpu but torch.cuda.is_available() is False")
    return torch.device("cpu")


def device_dtype(device):
    if is_xla(device):
        return torch.bfloat16
    if is_cuda(device):
        return torch.float16
    return torch.float32


def attn_implementation(device):
    if is_cuda(device):
        return None
    return "sdpa"


def xla_version():
    try:
        import torch_xla
    except Exception:
        return "?"
    return getattr(torch_xla, "__version__", "?")


def device_label(device):
    if is_cuda(device):
        return f"cuda {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability()}"
    xm = xla_model() if is_xla(device) else None
    if xm is None:
        return "cpu"
    kind = xm.xla_device_hw(device) if hasattr(xm, "xla_device_hw") else "tpu"
    supported = getattr(xm, "get_xla_supported_devices", None)
    try:
        count = len(supported()) if supported is not None else 0
    except Exception:
        count = 0
    return f"xla {kind} devices {count} torch_xla {xla_version()}"


def mark_step(device):
    xm = xla_model() if is_xla(device) else None
    if xm is not None:
        xm.mark_step()


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize()
        return
    xm = xla_model() if is_xla(device) else None
    if xm is not None:
        xm.wait_device_ops()


def empty_cache(device):
    if is_cuda(device):
        torch.cuda.empty_cache()
        return
    mark_step(device)


def reset_peak(device):
    if is_cuda(device):
        torch.cuda.reset_peak_memory_stats()


def memory_stats(device):
    if is_cuda(device):
        return {"alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    xm = xla_model() if is_xla(device) else None
    if xm is None:
        return {}
    info = xm.get_memory_info(device)
    return {"alloc_gb": info.get("bytes_used", 0) / 1e9,
            "peak_gb": info.get("peak_bytes_used", 0) / 1e9,
            "limit_gb": info.get("bytes_limit", 0) / 1e9}


def is_oom(exc):
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = f"{type(exc).__name__} {exc}".lower().replace("_", " ")
    return "resource exhausted" in text or "out of memory" in text


def optimizer_step(optimizer, device, params=None, grad_clip=0.0):
    if grad_clip and params is not None:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    xm = xla_model() if is_xla(device) else None
    if xm is not None:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def to_device(tensor, device):
    if is_cuda(device):
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def manual_hidden_states(model, x, lm_head=None):
    hidden = model.embed_tokens(x)
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    rotary = getattr(model, "pos_emb", None)
    layer_kwargs = {"position_ids": positions, "cache_position": positions[0]}
    if rotary is not None:
        layer_kwargs["position_embeddings"] = rotary(hidden, positions)
    states = [hidden]
    for layer in model.layers:
        produced = layer(hidden, **layer_kwargs)
        hidden = produced[0] if isinstance(produced, tuple) else produced
        states.append(hidden)
    final_norm = getattr(model, "norm", None)
    if final_norm is None:
        final_norm = getattr(model, "embedding_norm", None)
    if final_norm is not None:
        hidden = final_norm(hidden)
        states[-1] = hidden
    if lm_head is not None:
        reference = model(x).last_hidden_state
        mine = lm_head(hidden.to(lm_head.weight.dtype)).float().argmax(-1)
        theirs = lm_head(reference.to(lm_head.weight.dtype)).float().argmax(-1)
        if (mine != theirs).float().mean().item() > 1e-3:
            raise RuntimeError("manual hidden states disagree with the model forward")
    return states


def model_hidden_states(model, x, lm_head=None, log=None):
    produced = model(x, output_hidden_states=True)
    states = getattr(produced, "hidden_states", None)
    if states is None:
        if log is not None:
            log("output_hidden_states unsupported by this transformers build, walking layers manually")
        states = manual_hidden_states(model, x, lm_head)
    return states
