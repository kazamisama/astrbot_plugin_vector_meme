from __future__ import annotations

"""离线建立 OpenCLIP 索引。

用途：不启动 AstrBot，直接把 vector_meme/memes 建成 SQLite + FAISS。

示例：
    python scripts/build_openclip_index.py
    python scripts/build_openclip_index.py --pretrained openai
    python scripts/build_openclip_index.py --meme-dir memes --data-dir .cache
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.database import MemeDatabase
from core.embedder import EmbedderFactory
from core.indexer import MemeIndexer, IndexProgress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme-dir", type=Path, default=ROOT / "memes")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="open_clip", choices=["open_clip", "dummy"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reset", action="store_true", help="删除旧 db/faiss 后重建")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.data_dir / "memes.db"
    index_path = args.data_dir / "memes.faiss"

    if args.reset:
        for p in [db_path, index_path]:
            if p.exists():
                p.unlink()
                print(f"removed {p}")

    print("=== vector_meme OpenCLIP index builder ===", flush=True)
    print(f"root       : {ROOT}", flush=True)
    print(f"meme_dir   : {args.meme_dir}", flush=True)
    print(f"data_dir   : {args.data_dir}", flush=True)
    print(f"backend    : {args.backend}", flush=True)
    print(f"model      : {args.model}", flush=True)
    print(f"pretrained : {args.pretrained}", flush=True)
    print(f"device     : {args.device}", flush=True)
    print()

    t0 = time.time()
    try:
        print("loading embedder...", flush=True)
        if args.backend == "open_clip":
            embedder = EmbedderFactory.create(
                "open_clip",
                model_name=args.model,
                pretrained=args.pretrained,
                device=args.device,
            )
        else:
            embedder = EmbedderFactory.create("dummy")
        print(f"embedder loaded. dim={embedder.dim}, time={time.time()-t0:.1f}s", flush=True)

        db = MemeDatabase(db_path=db_path, index_path=index_path, dim=embedder.dim)
        indexer = MemeIndexer(db, embedder, use_subdir_as_tag=True, default_tag="misc")

        last_report = 0
        def on_progress(done: int, total: int, fp: Path):
            nonlocal last_report
            if total and (done == total or done - last_report >= 10):
                last_report = done
                print(f"indexing {done}/{total}: {fp.name}", flush=True)

        progress = IndexProgress(on_progress=on_progress)
        progress = indexer.index_directory(
            args.meme_dir,
            recursive=True,
            batch_size=args.batch_size,
            progress=progress,
        )
        print()
        print("=== done ===", flush=True)
        print(f"total   : {progress.total}", flush=True)
        print(f"added   : {progress.added}", flush=True)
        print(f"updated : {progress.updated}", flush=True)
        print(f"skipped : {progress.skipped}", flush=True)
        print(f"failed  : {progress.failed}", flush=True)
        print(f"stats   : {db.stats()}", flush=True)
        print(f"tags    : {db.count_by_tag()}", flush=True)
        print(f"elapsed : {time.time()-t0:.1f}s", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
