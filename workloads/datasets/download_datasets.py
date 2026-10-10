#!/usr/bin/env python3
"""
FairRoute dataset downloader (parallel).

Downloads every source dataset / repo / tokenizer for the FairRoute workload suite
into  ./fairroute_workloads/data/raw/<dataset>/  (change with --out), all at once.

Setup:
    pip install -U huggingface_hub
    export HF_TOKEN=hf_xxx            # Windows PowerShell: $env:HF_TOKEN="hf_xxx"

Usage:
    python download_datasets.py                 # everything except optional sets
    python download_datasets.py --optional      # also WildChat-1M (very large, gated)
    python download_datasets.py --list          # show what would be downloaded
    python download_datasets.py --only sharegpt loogle
    python download_datasets.py --skip burstgpt --workers 8
    python download_datasets.py --force         # ignore .done markers, re-download

Gated repos (Llama-3.1 tokenizer, lmsys/chatbot_arena_conversations, WildChat) need
you to accept the terms once on the Hugging Face web page while logged in with the
same account that owns HF_TOKEN. A failure in one dataset never stops the others.

Per dataset it writes  _provenance.json  (source, revision SHA when resolvable,
date, file list with sizes, sha256 for directly downloaded files) to help fill in
DATA_SOURCES.md. Nothing is fabricated: if a source can't be fetched it is reported
as FAILED with the reason.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Run:  pip install -U huggingface_hub")

try:  # faster transfers if available
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
UA = {"User-Agent": "fairroute-downloader/1.0"}
PRINT_LOCK = threading.Lock()


def log(name: str, msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {name:<18} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    name: str                       # folder name under data/raw/
    kind: str                       # hf | hf_files | urls | github | manual
    desc: str
    repo: str = ""                  # HF repo id or "owner/repo" for github
    repo_type: str = "dataset"      # for hf kinds
    allow_patterns: list[str] | None = None
    # urls kind: list of (filename, [candidate urls tried in order])
    files: list[tuple[str, list[str]]] = field(default_factory=list)
    gated: bool = False
    optional: bool = False
    note: str = ""


SOURCES: list[Source] = [
    # ---- Hugging Face datasets ------------------------------------------- #
    Source("sharegpt", "hf", "ShareGPT_Vicuna_unfiltered (NexusSched)",
           repo="anon8231489123/ShareGPT_Vicuna_unfiltered"),
    Source("lmsys_chatbot_arena", "hf", "LMSYS Chatbot Arena conversations (VTC, D2LPM)",
           repo="lmsys/chatbot_arena_conversations", gated=True),
    Source("loogle", "hf", "LooGLE long-document QA (Preble, D2LPM)",
           repo="bigai-nlco/LooGLE"),
    Source("arxiv_summarization", "hf", "arxiv_summarization_postprocess (NexusSched)",
           repo="whu9/arxiv_summarization_postprocess"),
    Source("code_reasoning", "hf", "Code-Reasoning (NexusSched)",
           repo="efficientscaling/Z1-Code-Reasoning-107K"),
    Source("wildchat", "hf", "WildChat-1M (optional; real client structure)",
           repo="allenai/WildChat-1M", gated=True, optional=True,
           note="Very large. Check licence before use."),
    # ---- Tokenizer (gated) ----------------------------------------------- #
    Source("llama31_8b_tokenizer", "hf", "Llama-3.1-8B tokenizer files only",
           repo="meta-llama/Llama-3.1-8B", repo_type="model", gated=True,
           allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                           "generation_config.json", "LICENSE*", "USE_POLICY*"]),
    # ---- Azure LLM inference traces (no text) ---------------------------- #
    Source("azure_llm_inference", "urls", "Azure LLM Inference Trace (2023 + 2024, conv & code)",
           files=[
               ("AzureLLMInferenceTrace_conv.csv", [
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv.csv",
                   "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv"]),
               ("AzureLLMInferenceTrace_code.csv", [
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code.csv",
                   "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv"]),
               ("AzureLLMInferenceTrace_conv_1week.csv", [
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace2024/AzureLLMInferenceTrace_conv_1week.csv",
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv"]),
               ("AzureLLMInferenceTrace_code_1week.csv", [
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace2024/AzureLLMInferenceTrace_code_1week.csv",
                   "https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv"]),
           ],
           note="2024 URLs are best-effort; if they fail, grab them from "
                "https://github.com/Azure/AzurePublicDataset (AzureLLMInferenceDataset2024.md)."),
    # ---- BurstGPT (no text) ---------------------------------------------- #
    Source("burstgpt", "urls", "BurstGPT trace (AdaGen)",
           files=[
               ("BurstGPT_1.csv", [
                   "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv",
                   "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv",
                   "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_without_fails_1.csv"]),
           ],
           note="If this fails, see https://github.com/HPMLL/BurstGPT (releases / data folder)."),
    # ---- GitHub repos (code read for schemas / workload construction) ---- #
    Source("repo_LLMServingSim", "github", "Simulator: casys-kaist/LLMServingSim",
           repo="casys-kaist/LLMServingSim"),
    Source("repo_VTC_artifact", "github", "VTC artifact (workload construction)",
           repo="Ying1123/VTC-artifact"),
    Source("repo_preble", "github", "Preble (Toolbench/LooGLE/video QA generators)",
           repo="WukLab/preble"),
    Source("repo_ToolBench", "github", "ToolBench code (OpenBMB)", repo="OpenBMB/ToolBench"),
    Source("repo_StableToolBench", "github", "StableToolBench code (THUNLP-MT)",
           repo="THUNLP-MT/StableToolBench"),
    # ---- Manual ----------------------------------------------------------- #
    Source("toolbench_data", "manual", "ToolBench / StableToolBench query data",
           note="Data is hosted on Google Drive (and HF mirrors), linked from the READMEs of "
                "repo_ToolBench and repo_StableToolBench. Download it and place it in "
                "data/raw/toolbench_data/."),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(dest: Path, with_hash: bool = False) -> list[dict]:
    out = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and not p.name.startswith(("_provenance", ".done")) \
                and ".cache" not in p.parts:
            rec = {"path": str(p.relative_to(dest)), "bytes": p.stat().st_size}
            if with_hash:
                rec["sha256"] = sha256_file(p)
            out.append(rec)
    return out


def http_download(url: str, dest: Path, name: str, retries: int = 3) -> None:
    """Download with resume + retries. Raises FileNotFoundError on 404/403 (try next URL)."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            part.replace(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():      # already complete
                part.replace(dest)
                return
            if e.code in (401, 403, 404):
                raise FileNotFoundError(f"HTTP {e.code} for {url}")
            err = f"HTTP {e.code}"
        except Exception as e:  # network hiccup
            err = repr(e)
        log(name, f"retry {attempt}/{retries} after error: {err}")
        time.sleep(2 * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}")


# --------------------------------------------------------------------------- #
# Download implementations (each returns provenance dict)
# --------------------------------------------------------------------------- #
def dl_hf(src: Source, dest: Path) -> dict:
    prov: dict = {"source": f"https://huggingface.co/"
                            f"{'datasets/' if src.repo_type == 'dataset' else ''}{src.repo}"}
    api = HfApi(token=HF_TOKEN)
    try:
        info = (api.dataset_info if src.repo_type == "dataset" else api.model_info)(src.repo)
        prov["revision_sha"] = info.sha
        prov["last_modified"] = str(getattr(info, "last_modified", ""))
        prov["license_tag"] = next((t for t in (info.tags or []) if t.startswith("license:")), None)
    except Exception as e:
        prov["revision_sha"] = None
        prov["revision_lookup_error"] = type(e).__name__
    snapshot_download(
        repo_id=src.repo, repo_type=src.repo_type, local_dir=str(dest),
        token=HF_TOKEN, allow_patterns=src.allow_patterns,
        revision=prov.get("revision_sha"), max_workers=8,
    )
    return prov


def dl_urls(src: Source, dest: Path) -> dict:
    prov = {"files": {}}
    errors = []
    for fname, candidates in src.files:
        target = dest / fname
        if target.exists() and target.stat().st_size > 0:
            prov["files"][fname] = {"url": "(already present)"}
            continue
        for url in candidates:
            try:
                log(src.name, f"GET {fname}  <- {url}")
                http_download(url, target, src.name)
                prov["files"][fname] = {"url": url}
                break
            except FileNotFoundError as e:
                log(src.name, f"not available: {e}")
            except Exception as e:
                log(src.name, f"error: {e}")
        else:
            errors.append(fname)
    if errors:
        prov["missing_files"] = errors
        if len(errors) == len(src.files):
            raise RuntimeError(f"no file could be downloaded ({', '.join(errors)}). {src.note}")
    return prov


def dl_github(src: Source, dest: Path) -> dict:
    owner, repo = src.repo.split("/")
    prov: dict = {"source": f"https://github.com/{src.repo}"}
    refs: list[str] = []
    for branch in ("main", "master"):
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
                headers={**UA, "Accept": "application/vnd.github.sha"})
            with urllib.request.urlopen(req, timeout=30) as r:
                sha = r.read().decode().strip()
            if len(sha) == 40:
                refs.append(sha)
                prov["commit_sha"] = sha
                break
        except Exception:
            continue
    refs += [f"refs/heads/{b}" for b in ("main", "master")]
    zpath = dest / "_repo.zip"
    last = None
    for ref in refs:
        url = f"https://codeload.github.com/{owner}/{repo}/zip/{ref}"
        try:
            http_download(url, zpath, src.name)
            prov["archive_url"] = url
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"could not fetch repo archive: {last}")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        top = names[0].split("/")[0] + "/"
        for n in names:
            rel = n[len(top):] if n.startswith(top) else n
            if not rel:
                continue
            out = dest / rel
            if n.endswith("/"):
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as s, open(out, "wb") as d:
                    shutil.copyfileobj(s, d)
    zpath.unlink(missing_ok=True)
    return prov


DISPATCH: dict[str, Callable[[Source, Path], dict]] = {
    "hf": dl_hf, "urls": dl_urls, "github": dl_github,
}


def explain_error(src: Source, e: Exception) -> str:
    name = type(e).__name__
    text = str(e)
    if "Gated" in name or "gated" in text.lower() or "401" in text or "403" in text:
        base = f"https://huggingface.co/{'datasets/' if src.repo_type == 'dataset' else ''}{src.repo}"
        tok = "HF_TOKEN is not set. " if not HF_TOKEN else ""
        return (f"ACCESS DENIED. {tok}Log in on huggingface.co with the account that owns "
                f"your token and accept the terms at {base}, then re-run.")
    return f"{name}: {text[:300]}"


def run_one(src: Source, root: Path, force: bool) -> dict:
    dest = root / src.name
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".done"
    t0 = time.time()
    if src.kind == "manual":
        log(src.name, f"MANUAL: {src.note}")
        return {"name": src.name, "status": "MANUAL", "detail": src.note, "secs": 0, "bytes": 0}
    if marker.exists() and not force:
        log(src.name, "already complete (use --force to redo)")
        return {"name": src.name, "status": "SKIPPED", "detail": "already complete",
                "secs": 0, "bytes": dir_size(dest)}
    log(src.name, f"START  {src.desc}")
    try:
        prov = DISPATCH[src.kind](src, dest)
        prov.update({
            "name": src.name, "description": src.desc,
            "accessed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "files": list_files(dest, with_hash=(src.kind == "urls")),
        })
        (dest / "_provenance.json").write_text(json.dumps(prov, indent=2))
        partial = bool(prov.get("missing_files"))
        if not partial:
            marker.write_text(prov["accessed_utc"])
        size = dir_size(dest)
        secs = time.time() - t0
        status = "PARTIAL" if partial else "OK"
        detail = (f"missing: {prov['missing_files']}. {src.note}" if partial else
                  f"{len(prov['files'])} files")
        log(src.name, f"{status}   {human(size)} in {secs:.0f}s")
        return {"name": src.name, "status": status, "detail": detail, "secs": secs, "bytes": size}
    except Exception as e:
        msg = explain_error(src, e)
        log(src.name, f"FAILED  {msg}")
        return {"name": src.name, "status": "FAILED", "detail": msg,
                "secs": time.time() - t0, "bytes": dir_size(dest)}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel FairRoute dataset downloader")
    ap.add_argument("--out", default="fairroute_workloads/data/raw",
                    help="output folder (default: ./fairroute_workloads/data/raw)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel downloads (default: one per dataset)")
    ap.add_argument("--only", nargs="*", default=None, help="only these names")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these names")
    ap.add_argument("--optional", action="store_true", help="include optional (WildChat)")
    ap.add_argument("--force", action="store_true", help="re-download even if complete")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    args = ap.parse_args()

    chosen = [s for s in SOURCES
              if (args.optional or not s.optional)
              and (args.only is None or s.name in args.only)
              and s.name not in args.skip]
    if args.only:
        unknown = set(args.only) - {s.name for s in SOURCES}
        if unknown:
            print(f"Unknown names: {sorted(unknown)}\nValid: {[s.name for s in SOURCES]}")
            return 2

    if args.list:
        for s in SOURCES:
            flags = ("gated " if s.gated else "") + ("optional" if s.optional else "")
            print(f"{s.name:<24} {s.kind:<7} {flags:<14} {s.desc}")
        return 0

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Output folder : {root}")
    print(f"HF_TOKEN      : {'set' if HF_TOKEN else 'NOT SET (gated sets will fail)'}")
    print(f"Datasets      : {', '.join(s.name for s in chosen)}\n")

    workers = args.workers or max(1, len(chosen))
    results: list[dict] = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, s, root, args.force): s for s in chosen}
        pending = set(futs)
        try:
            while pending:
                done, pending = cf.wait(pending, timeout=20, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    results.append(f.result())
                if pending:
                    running = ", ".join(
                        f"{futs[f].name}({human(dir_size(root / futs[f].name))})" for f in pending)
                    log("progress", f"running: {running}")
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run to resume (partial files are kept).")
            pool.shutdown(wait=False, cancel_futures=True)
            return 130

    results.sort(key=lambda r: r["name"])
    print("\n" + "=" * 78)
    print(f"{'DATASET':<24}{'STATUS':<9}{'SIZE':>10}  DETAIL")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<24}{r['status']:<9}{human(r['bytes']):>10}  {r['detail'][:60]}")
    print("=" * 78)
    print(f"Total wall time: {time.time() - t0:.0f}s")
    (root / "_download_summary.json").write_text(json.dumps(results, indent=2))

    bad = [r for r in results if r["status"] in ("FAILED", "PARTIAL")]
    if bad:
        print("\nNeeds attention:")
        for r in bad:
            print(f" - {r['name']}: {r['detail']}")
    manual = [r for r in results if r["status"] == "MANUAL"]
    for r in manual:
        print(f"\nManual step ({r['name']}): {r['detail']}")
    return 1 if any(r["status"] == "FAILED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())