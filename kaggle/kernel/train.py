"""Kaggle entrypoint — fully self-contained.

Load order for the base model:
  1. MODEL_PATH env (explicit local folder)
  2. Known Kaggle input dataset folders
  3. Any /kaggle/input/**/config.json that has model weights
  4. HuggingFace hub (needs working Internet) — fail fast, no long waits
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

INPUT_CSV = os.environ.get(
    "INPUT_CSV", "/kaggle/input/approved-parallel/parallel.csv"
)
OUT = Path("/kaggle/working/santali-output")
DATA_DIR = OUT / "data"
ADAPTER_DIR = OUT / "adapter"
HF_MODEL_ID = os.environ.get(
    "MODEL_ID", "ai4bharat/indictrans2-en-indic-dist-200M"
)
SRC_LANG = "eng_Latn"
TGT_LANG = "sat_Olck"
OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")
EPOCHS = float(os.environ.get("EPOCHS", "3"))

CRITICAL_PKGS = [
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "sentencepiece",
    "pandas",
]

LOCAL_MODEL_CANDIDATES = [
    os.environ.get("MODEL_PATH", "").strip(),
    "/kaggle/input/indictrans2-en-indic-200m",
    "/kaggle/input/indictrans2-indic-indic-dist-320m",
    "/kaggle/input/ai4bharat-indictrans2-en-indic-200m",
    "/kaggle/input/indictrans2",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def debug_layout() -> None:
    log("=" * 60)
    log("Kaggle kernel starting (self-contained)")
    log(f"cwd      : {Path.cwd()}")
    log(f"__file__  : {globals().get('__file__', 'N/A')}")
    for root in [Path("/kaggle/src"), Path("/kaggle/working"), Path("/kaggle/input")]:
        if root.exists():
            log(f"{root}:")
            for p in sorted(root.rglob("*"))[:60]:
                if p.is_file():
                    log(f"  {p}")
    log("=" * 60)


def _import_ok(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _pip_install(spec: str, retries: int = 2) -> bool:
    for attempt in range(1, retries + 1):
        try:
            log(f"  pip install {spec} (attempt {attempt}/{retries})")
            subprocess.check_call(
                [
                    sys.executable, "-m", "pip", "install", "-q",
                    "--disable-pip-version-check", "--no-input", spec,
                ],
                timeout=180,
            )
            return True
        except Exception as e:
            log(f"  pip failed for {spec}: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def install_transformers_compat() -> None:
    """Keep IndicTrans2 loadable on Kaggle images without network access."""
    import types
    try:
        import transformers
        import transformers.onnx  # type: ignore[attr-defined]
        log("  transformers.onnx is available")
        return
    except Exception as error:
        log(f"  transformers.onnx unavailable; installing local shim: {error}")
    module = types.ModuleType("transformers.onnx")
    module.__path__ = []
    class OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8
    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        use_past = False
        def fill_with_past_key_values_(self, inputs, direction="inputs"):
            return inputs
    module.OnnxConfig = OnnxConfig
    module.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    sys.modules["transformers.onnx"] = module
    utils = types.ModuleType("transformers.onnx.utils")
    def compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):
        if dimension is None or dimension < 0:
            return fixed_dimension + num_token_to_add
        return dimension + num_token_to_add
    utils.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = utils
    log("  local transformers.onnx shim installed")


def install_deps() -> None:
    log("Checking training dependencies...")
    missing = []
    for pkg in CRITICAL_PKGS:
        if _import_ok(pkg):
            log(f"  OK preinstalled: {pkg}")
        else:
            log(f"  missing: {pkg}")
            missing.append(pkg)
    for pkg in missing:
        if not _pip_install(pkg) or not _import_ok(pkg):
            raise RuntimeError(
                f"Critical package '{pkg}' missing and pip failed. "
                "Turn Internet ON and re-run, or wait for Kaggle DNS."
            )
    install_transformers_compat()
    log("Dependency check done (no optional pip installs).")


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    has_weights = (
        (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors").exists()
        or any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
    )
    return has_weights


def resolve_model_source() -> str:
    """Local Kaggle input first, else HuggingFace id."""
    for raw in LOCAL_MODEL_CANDIDATES:
        if not raw:
            continue
        p = Path(raw)
        if _looks_like_model_dir(p):
            log(f"Using LOCAL model: {p}")
            return str(p)
        if p.is_dir():
            for sub in p.iterdir():
                if _looks_like_model_dir(sub):
                    log(f"Using LOCAL model: {sub}")
                    return str(sub)

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for cfg in input_root.rglob("config.json"):
            parent = cfg.parent
            if _looks_like_model_dir(parent):
                log(f"Using LOCAL model (auto): {parent}")
                return str(parent)

    log(f"No local model under /kaggle/input — HF id: {HF_MODEL_ID}")
    log("Attach dataset janaiworkspace/indictrans2-en-indic-200m to avoid network dependency.")
    return HF_MODEL_ID


def network_ok(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as e:
        log(f"Network probe {host}:{port} failed: {e}")
        return False


def patch_local_model_for_transformers(model_ref: str) -> str:
    """Use symlinked weights with a tiny source-only compatibility patch."""
    import shutil
    patched = Path("/kaggle/working/indictrans2-en-indic-200m-patched")
    if patched.exists():
        return str(patched)
    shutil.copytree(model_ref, patched, symlinks=True)
    modeling = patched / "modeling_indictrans.py"
    if modeling.exists():
        source = modeling.read_text(encoding="utf-8")
        updated = re.sub(
            r"def tie_weights\(self(?:, [^)]*)?\):",
            "def tie_weights(self, *args, **kwargs):",
            source,
            count=1,
        )
        if updated != source:
            modeling.write_text(updated, encoding="utf-8")
            log("  patched IndicTrans2 tie_weights signature for local loader")
    return str(patched)


def disable_incompatible_torchao() -> None:
    """Use ordinary LoRA when Kaggle ships an unsupported torchao build."""
    try:
        import peft.import_utils as import_utils
        import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao as torchao_dispatcher
        torchao_dispatcher.is_torchao_available = lambda: False
        log("  disabled incompatible torchao dispatcher; using standard LoRA")
    except Exception as error:
        log(f"  torchao compatibility probe skipped: {error}")


def load_tokenizer_and_model(model_ref: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    is_local = Path(model_ref).exists()
    last_err: Exception | None = None

    if is_local:
        log(f"Loading LOCAL model (no network wait): {model_ref}")
        compatible_ref = patch_local_model_for_transformers(model_ref)
        tokenizer = AutoTokenizer.from_pretrained(compatible_ref, trust_remote_code=True, local_files_only=True)
        # The Kaggle model dataset contains both safetensors and PyTorch weights.
        # Prefer the PyTorch checkpoint here: it avoids a known NaN forward-pass
        # failure observed with the local safetensors shard on the Kaggle image.
        model = AutoModelForSeq2SeqLM.from_pretrained(
            compatible_ref,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=False,
            torch_dtype=torch.float32,
        )
        bad_parameters = [
            name for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter.detach()).all()
        ]
        if bad_parameters:
            preview = ", ".join(bad_parameters[:5])
            raise RuntimeError(f"Local model contains non-finite weights: {preview}")
        log("Model loaded successfully from local disk.")
        return tokenizer, model

    # Remote HF path — fail fast, do NOT wait minutes for DNS
    if not network_ok():
        raise RuntimeError(
            "No local model attached and huggingface.co is unreachable (DNS/network).\n"
            "Do NOT wait — fix one of these:\n"
            "  1) Run GitHub Actions with HF_TOKEN so janaiworkspace/indictrans2-en-indic-200m is uploaded\n"
            "  2) Kernel → Add Input → dataset janaiworkspace/indictrans2-en-indic-200m\n"
            "  3) Settings → Internet ON, then re-run when Kaggle DNS works\n"
            f"  Current /kaggle/input: {list(Path('/kaggle/input').iterdir()) if Path('/kaggle/input').exists() else []}"
        )

    for attempt in range(1, 4):
        try:
            log(f"Loading from HuggingFace {model_ref!r} (attempt {attempt}/3)")
            tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_ref, trust_remote_code=True)
            log("Model loaded successfully from HuggingFace.")
            return tokenizer, model
        except Exception as e:
            last_err = e
            log(f"  load failed: {type(e).__name__}: {e}")
            if attempt < 3:
                time.sleep(5)

    raise RuntimeError(
        f"Failed to load model {model_ref!r}.\nLast error: {last_err}\n"
        "Prefer attaching local dataset janaiworkspace/indictrans2-en-indic-200m."
    ) from last_err


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def normalize_rows(rows: list[dict]) -> list[dict]:
    for r in rows:
        for k in ("source", "target"):
            if r.get(k):
                r[k] = unicodedata.normalize("NFC", r[k]).strip()
    return rows


def validate_rows(rows: list[dict]) -> list[dict]:
    kept, seen = [], set()
    stats = {"read": 0, "kept": 0, "duplicates": 0, "empty": 0, "no_olchiki": 0}
    for r in rows:
        stats["read"] += 1
        source = (r.get("source") or "").strip()
        target = (r.get("target") or "").strip()
        if not source or not target:
            stats["empty"] += 1
            continue
        key = (source, target)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        tlang = r.get("target_lang", "sat_Olck")
        if tlang == "sat_Olck" and not OL_CHIKI.search(target):
            stats["no_olchiki"] += 1
            continue
        kept.append(r)
        stats["kept"] += 1
    log(f"validate: {json.dumps(stats)}")
    return kept


def split_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n == 0:
        return {"train": [], "validation": [], "test": []}
    return {
        "train": rows[: int(0.8 * n)],
        "validation": rows[int(0.8 * n) : int(0.9 * n)],
        "test": rows[int(0.9 * n) :],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["source", "target"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare_data() -> Path:
    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        alts = list(Path("/kaggle/input").rglob("parallel.csv"))
        if alts:
            csv_path = alts[0]
            log(f"Using alternate input: {csv_path}")
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {INPUT_CSV}\n"
                "Attach dataset janaiworkspace/approved-parallel."
            )

    log(f"Reading {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    log(f"Raw rows: {len(rows)}")

    if rows and ("source" not in rows[0] or "target" not in rows[0]):
        for r in rows:
            if "src" in r and "source" not in r:
                r["source"] = r.get("src", "")
            if "tgt" in r and "target" not in r:
                r["target"] = r.get("tgt", "")
            if "en" in r and "source" not in r:
                r["source"] = r.get("en", "")
            if "sat" in r and "target" not in r:
                r["target"] = r.get("sat", "")

    rows = normalize_rows(rows)
    rows = validate_rows(rows)
    if not rows:
        raise RuntimeError("No valid parallel pairs after validation.")

    splits = split_rows(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        write_csv(DATA_DIR / f"{name}.csv", data)
        log(f"  {name}: {len(data)}")
    return DATA_DIR


# ---------------------------------------------------------------------------
# LoRA training
# ---------------------------------------------------------------------------
def train_lora(data_dir: Path) -> None:
    import pandas as pd
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model
    disable_incompatible_torchao()

    def load_csv(path: Path) -> Dataset:
        df = pd.read_csv(path).fillna("")
        if not {"source", "target"}.issubset(df.columns):
            raise ValueError(f"Need source/target columns, got {list(df.columns)}")
        return Dataset.from_pandas(df[["source", "target"]], preserve_index=False)

    for split in ("train", "validation", "test"):
        p = data_dir / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    ds = DatasetDict({s: load_csv(data_dir / f"{s}.csv") for s in ("train", "validation", "test")})
    log(f"Dataset sizes: { {k: len(v) for k, v in ds.items()} }")

    model_ref = resolve_model_source()
    tokenizer, model = load_tokenizer_and_model(model_ref)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def encode(batch):
        sources = [f"{SRC_LANG} {TGT_LANG} {t}" for t in batch["source"]]
        model_inputs = tokenizer(sources, max_length=128, truncation=True, padding=False)
        labels = tokenizer(
            text_target=batch["target"], max_length=128, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        encode, batched=True, remove_columns=ds["train"].column_names, desc="Tokenizing"
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Training must not run on CPU: the 320M model is too slow "
            "and the previous CPU run produced loss=0/NaN. Enable Kaggle GPU in Settings > Accelerator "
            "and rerun."
        )
    sanity_batch = collator([tokenized["train"][0], tokenized["train"][1]])
    label_tokens = int((sanity_batch["labels"] != -100).sum().item())
    if label_tokens == 0:
        raise RuntimeError("Sanity check failed: all label tokens are masked; refusing to train.")
    model.eval()
    with torch.no_grad():
        sanity_outputs = model(**{key: value.to(model.device) for key, value in sanity_batch.items()})
    sanity_loss = float(sanity_outputs.loss.detach().float().cpu())
    log(
        f"Sanity check: label_tokens={label_tokens}, initial_loss={sanity_loss:.6f}, "
        f"cuda={torch.cuda.get_device_name(0)}"
    )
    if not torch.isfinite(sanity_outputs.loss):
        raise RuntimeError(
            f"Sanity check failed: initial loss is {sanity_loss}; refusing to start training. "
            "Check model weights, dtype, and tokenizer compatibility."
        )

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    use_fp16 = torch.cuda.is_available()
    args = Seq2SeqTrainingArguments(
        output_dir=str(ADAPTER_DIR),
        learning_rate=5e-5,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        logging_steps=25,
        predict_with_generate=True,
        fp16=use_fp16,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    log("Starting training...")
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    log("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized["test"])
    (ADAPTER_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps(metrics, indent=2))
    log(f"Artifacts saved to {ADAPTER_DIR}")


def main() -> None:
    debug_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    install_deps()
    data_dir = prepare_data()
    train_lora(data_dir)
    log("Done.")


if __name__ == "__main__":
    main()
