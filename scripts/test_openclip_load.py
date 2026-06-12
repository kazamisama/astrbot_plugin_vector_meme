from __future__ import annotations

import time
import traceback
from pathlib import Path

print("[test_openclip] start", flush=True)
try:
    import torch
    import open_clip

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("open_clip", getattr(open_clip, "__version__", "unknown"), flush=True)
    print("creating model ViT-B-32 / laion2b_s34b_b79k ...", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = tokenizer(["happy anime expression"]).to("cpu")
    with torch.no_grad():
        feat = model.encode_text(text)
    print("text feature shape", tuple(feat.shape), flush=True)
    print("OK", flush=True)
except Exception:
    traceback.print_exc()
    raise
