"""Fine-tuning de un BERT ligero (DistilBERT) como regresor de años de experiencia.

Configuración
    - AutoModelForSequenceClassification(num_labels=1, problem_type="regression"): una sola salida, pérdida MSE.
      Variante Huber (pregunta 3) con delta = 1 año en la escala del target.
    - Target escalado: y / 20 durante el entrenamiento; las predicciones se vuelven a años y se recortan a [0, 20].
    - Longitud: max_length = 256 con truncado de cabeza. Los CVs limpios tienen como máximo 177 tokens de DistilBERT,
      así que ningún CV se trunca y todas las fechas llegan al modelo (se verifica y se reporta n_truncated).
    - Early stopping sobre MAE de validación (en años), paciencia 3; se restaura el mejor checkpoint.
    - Mismo split y mismos escenarios que los baselines (src/experiments.py).

Pensado para correr en GPU (Colab T4): notebooks/02_colab_bert.ipynb.

Salidas: output/predictions/bert*.csv, output/metrics/bert*.json, models/bert/<escenario>/ (pesos, no se suben)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile

import numpy as np
import torch

from src.experiments import SETUPS, run
from src.utils import BERT_DIR, SEED, TARGET_CAP, get_logger

log = get_logger("bert")

MODEL_NAME = "distilbert-base-uncased"
HPARAMS = {
    "learning_rate": 5e-5,
    "batch_size": 16,
    "max_epochs": 20,
    "patience": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 256,
    "seed": SEED,
}
HUBER_DELTA = 1 / TARGET_CAP  # 1 año en la escala y / 20


class BertPredictor:
    def __init__(self, model_dir, max_length: int = HPARAMS["max_length"], batch_size: int = 64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    @torch.no_grad()
    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[i:i + self.batch_size]), truncation=True, padding=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            out.append(self.model(**batch).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) * TARGET_CAP


def make_fit(loss: str = "mse", save_dir=None):
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    class HuberTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.huber_loss(outputs.logits.squeeze(-1), labels, delta=HUBER_DELTA)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.clip(np.asarray(preds).reshape(-1) * TARGET_CAP, 0, TARGET_CAP)
        return {"mae": float(np.mean(np.abs(preds - np.asarray(labels) * TARGET_CAP)))}

    def fit(train_texts, y_train, val_texts, y_val):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1,
                                                                   problem_type="regression")

        def to_dataset(texts, y):
            enc = tokenizer(texts, truncation=True, max_length=HPARAMS["max_length"])
            return Dataset.from_dict({**enc, "labels": (np.asarray(y, dtype=np.float32) / TARGET_CAP).tolist()})

        lengths = [len(ids) for ids in tokenizer(train_texts + val_texts)["input_ids"]]
        n_truncated = int(sum(n > HPARAMS["max_length"] for n in lengths))

        out_dir = tempfile.mkdtemp(prefix="bert_")
        args = TrainingArguments(
            output_dir=out_dir,
            learning_rate=HPARAMS["learning_rate"],
            per_device_train_batch_size=HPARAMS["batch_size"],
            per_device_eval_batch_size=64,
            num_train_epochs=HPARAMS["max_epochs"],
            weight_decay=HPARAMS["weight_decay"],
            warmup_steps=HPARAMS["warmup_ratio"],  # en transformers 5 un float < 1 es la fracción de pasos
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="mae",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            seed=HPARAMS["seed"],
            report_to="none",
            logging_strategy="epoch",
        )
        trainer = (HuberTrainer if loss == "huber" else Trainer)(
            model=model, args=args, processing_class=tokenizer,
            train_dataset=to_dataset(train_texts, y_train), eval_dataset=to_dataset(val_texts, y_val),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=HPARAMS["patience"])],
        )
        trainer.train()

        history = [{"epoch": h["epoch"], "val_mae": round(h["eval_mae"], 4)}
                   for h in trainer.state.log_history if "eval_mae" in h]
        final_dir = save_dir or tempfile.mkdtemp(prefix="bert_best_")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        shutil.rmtree(out_dir, ignore_errors=True)

        info = {
            "model_name": MODEL_NAME, "loss": loss, **HPARAMS,
            "huber_delta_years": HUBER_DELTA * TARGET_CAP if loss == "huber" else None,
            "best_epoch": min(history, key=lambda h: h["val_mae"])["epoch"] if history else None,
            "epochs_run": len(history), "history": history,
            "n_truncated_train_val": n_truncated, "max_tokens_train_val": int(max(lengths)),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        log.info("Mejor época %s | val MAE %.3f | truncados %d", info["best_epoch"],
                 min(h["val_mae"] for h in history), n_truncated)
        return BertPredictor(final_dir), info

    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-main", action="store_true", help="solo el escenario principal (prueba rápida)")
    parser.add_argument("--skip-huber", action="store_true")
    args = parser.parse_args()

    # Escenario principal: el modelo se guarda en models/bert/main para la app y el probe de fechas
    run("bert", make_fit("mse", save_dir=str(BERT_DIR / "main")), setups={"main": SETUPS["main"]}, save_model=False)
    if args.only_main:
        return
    # Pregunta 3: sensibilidad a la limpieza del OCR (mismo setup, otro texto)
    run("bert_cleaning", make_fit("mse"), setups={k: v for k, v in SETUPS.items() if k != "main"},
        reference=False, save_model=False)
    # Pregunta 3: pérdida Huber vs MSE
    if not args.skip_huber:
        run("bert_huber", make_fit("huber"), setups={"main": SETUPS["main"]}, reference=False, save_model=False)


if __name__ == "__main__":
    main()
