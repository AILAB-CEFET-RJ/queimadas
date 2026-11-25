"""Treino do STConvS2S com restrições físicas fracas (PIML), sem alterar o `stconv_train.py` da linha de base.

Este script importa o `stconv_train.py` e troca só a função de época de treino por uma que soma, à BCE mascarada
da linha de base, penalidades físicas ponderadas:

    L = BCE + λ_nb·L_nb + λ_vento·L_vento + λ_long·L_long + λ_mono·L_mono

Todo o resto é o do baseline: receita, dados, validação, escolha do checkpoint (pela `val_loss` = BCE pura, para que o
critério seja o mesmo dos runs `baseline_es_*`), limiar, relatório e avaliação única do teste. Com todos os λ = 0 o
treino é idêntico ao do baseline (mesma semente, mesma ordem dos lotes).

Penalidades (sobre p = σ(logit), sem usar o rótulo; campos e convenções de `physics.py`):

- `nonburnable` (L_nb): probabilidade média nos pixels avaliados de classes não combustíveis do MCD12Q1.
- `wind` (L_vento): fração da massa de fogo novo perto da frente que avança CONTRA o vento, ponderada por max(0, −cos)
  entre a normal externa da frente e o transporte do vento em t. Não exige alinhamento: só cobra o lado de barlavento.
- `far` (L_long): fração da massa de fogo novo a mais de `radius` px da frente. No dado ~25–37% do fogo novo está
  longe, então este termo empurra além do observado: use λ pequeno ou deixe em 0.
- `monotone` (L_mono): média de max(0, p − p') perto do fogo, em que p' é a previsão com ERC e temperatura máxima
  aumentados e precipitação reduzida em `delta` desvios-padrão. Custa uma segunda passada da rede por micro-lote.

Os runs saem com nome próprio (prefixo obrigatório `piml_`) e o relatório ganha a seção `piml` com os λ e a média
de cada termo por época. Exemplos:

    python scripts/stconv_train_piml.py --recipe paper_es --fold 2 --run-name piml_mono_f02 --lambda-monotone 1
    python scripts/stconv_train_piml.py --evaluate-test --run-name piml_mono_f02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stconv_train as base  # noqa: E402
from scripts.core import grid_geometry  # noqa: E402
from scripts.physics import (  # noqa: E402
    DEFAULT_DELTA_STD,
    DEFAULT_RADIUS_PX,
    MONOTONE_DRIVERS,
    PhysicsLayout,
    _plane,
    gradient,
    inside_image,
    perturb,
    proximity,
    wind_transport_grid,
)

RUN_PREFIX = "piml_"
TERMS = ("nonburnable", "wind", "far", "monotone")
EPS = 1e-6


def add_piml_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("PIML")
    group.add_argument("--lambda-nonburnable", type=float, default=0.0)
    group.add_argument("--lambda-wind", type=float, default=0.0)
    group.add_argument("--lambda-far", type=float, default=0.0)
    group.add_argument("--lambda-monotone", type=float, default=0.0)
    group.add_argument("--piml-radius", type=int, default=DEFAULT_RADIUS_PX, help="Raio da frente, em px.")
    group.add_argument("--piml-delta", type=float, default=DEFAULT_DELTA_STD,
                       help="Perturbação da monotonicidade, em desvios-padrão.")


def split_argv(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    add_piml_arguments(parser)
    return parser.parse_known_args(argv)


# --------------------------------------------------------------------------- penalidades


def penalty_terms(
    prob: Any, logits_shifted: Any | None, batch: dict[str, Any], x: Any, layout: PhysicsLayout, gamma_deg: Any,
    radius: int,
) -> dict[str, Any]:
    """Termos escalares (diferenciáveis em `prob`) do micro-lote. `prob`: `[B, 1, 1, P, P]` em float32."""
    mask = _plane(batch["mask"])
    fire_t = (_plane(batch["fire_t"]) > 0.5).float()
    p = _plane(prob) * mask
    inside = inside_image(x, layout)
    nonburnable = x[:, list(layout.nonburnable), -1].float().sum(dim=1).clamp(max=1.0)
    prox = proximity(fire_t, radius)
    near = (prox > 0).float()
    new = p * (1.0 - fire_t)
    near_new = new * near
    n_col, n_row = (-c for c in gradient(prox, inside))
    w_col, w_row = wind_transport_grid(x, layout, gamma_deg)
    n_norm = (n_col ** 2 + n_row ** 2).sqrt()
    w_norm = (w_col ** 2 + w_row ** 2).sqrt()
    valid = ((n_norm > EPS) & (w_norm > EPS)).float()
    cosine = (n_col * w_col + n_row * w_row) / (n_norm * w_norm).clamp(min=EPS)
    terms = {
        "nonburnable": (p * nonburnable).sum() / mask.sum().clamp(min=1.0),
        "wind": (near_new * valid * (-cosine).clamp(min=0.0)).sum() / (near_new * valid).sum().clamp(min=EPS),
        "far": (new * (1.0 - near)).sum() / new.sum().clamp(min=EPS),
    }
    if logits_shifted is not None:
        region = near * mask
        shifted = _plane(base_sigmoid(logits_shifted))
        terms["monotone"] = ((p - shifted * mask).clamp(min=0.0) * region).sum() / region.sum().clamp(min=1.0)
    return terms


def base_sigmoid(logits: Any) -> Any:
    import torch

    return torch.sigmoid(logits.float())


def _physics_context(loader: Any) -> tuple[PhysicsLayout, Any]:
    """Layout dos canais e γ por evento a partir do dataset do loader (o mesmo do treino)."""
    import torch

    dataset = loader.dataset
    layout = PhysicsLayout.from_channels(dataset.channel_names, dataset.normalization)
    gammas = dataset.gamma_deg or [grid_geometry(e.lnglat)["gamma_deg"] for e in dataset.events]
    return layout, torch.tensor(gammas, dtype=torch.float32)


def make_train_one_epoch(settings: argparse.Namespace, log: list[dict[str, Any]]) -> Any:
    """Época de treino do baseline com as penalidades somadas à BCE. Mesma acumulação, AMP e corte por `max_steps`."""
    lambdas = {term: float(getattr(settings, f"lambda_{term}")) for term in TERMS}
    drivers = tuple(name for name, _ in MONOTONE_DRIVERS)

    def train_one_epoch(
        model: Any, loader: Iterable[dict[str, Any]], optimizer: Any, scaler: Any, *, device: Any, use_amp: bool,
        accumulate: int, pos_weight: float, step: int, max_steps: int | None,
    ) -> dict[str, Any]:
        import torch

        layout, gammas = _physics_context(loader)
        model.train()
        losses, pending, skipped, batches = [], 0, 0, 0
        term_sums = {term: 0.0 for term in TERMS}
        bce_sum = 0.0
        stopped_by_max_steps = False

        def optimizer_step() -> None:
            nonlocal skipped
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.is_enabled() and scaler.get_scale() < scale)
            optimizer.zero_grad(set_to_none=True)

        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            fields = {k: batch[k].to(device) for k in ("y", "mask", "fire_t")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)[:, :, -1:]
                logits_shifted = (model(perturb(x, layout, drivers, settings.piml_delta))[:, :, -1:]
                                  if lambdas["monotone"] > 0 else None)
            bce = base.masked_bce_with_logits(logits.float(), fields["y"], fields["mask"], pos_weight)
            loss = bce
            if any(value > 0 for value in lambdas.values()):
                gamma = gammas[[int(e) for e in batch["event"]]].to(device)
                terms = penalty_terms(torch.sigmoid(logits.float()), logits_shifted, fields, x, layout, gamma,
                                      settings.piml_radius)
                for term, value in terms.items():
                    term_sums[term] += float(value.detach())
                    if lambdas[term] > 0:
                        loss = loss + lambdas[term] * value
            scaler.scale(loss / accumulate).backward()
            losses.append(float(loss.detach()))
            bce_sum += float(bce.detach())
            pending, batches = pending + 1, batches + 1
            if pending == accumulate:
                optimizer_step()
                pending, step = 0, step + 1
                if max_steps is not None and step >= max_steps:
                    stopped_by_max_steps = True
                    break
        if pending:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulate / pending)
            optimizer_step()
            step += 1
        complete = not stopped_by_max_steps or batches == len(loader)  # type: ignore[arg-type]
        n = max(batches, 1)
        log.append({"batches": batches, "bce": bce_sum / n, **{term: term_sums[term] / n for term in TERMS}})
        print("  [piml] " + " ".join(f"{k}={v:.4g}" for k, v in log[-1].items() if k != "batches"), flush=True)
        return {"losses": losses, "step": step, "epoch_complete": complete, "skipped_steps": skipped,
                "batches": batches}

    return train_one_epoch


# --------------------------------------------------------------------------- execução


def main(argv: list[str] | None = None) -> dict[str, Any]:
    settings, rest = split_argv(argv if argv is not None else sys.argv[1:])
    args = base.parse_args(rest)
    if args.evaluate_test:
        # A avaliação do teste é a do baseline: mesmo modelo, mesmo livro-razão, mesma regra de teste único.
        return base.run_test_evaluation(args)
    if not (args.run_name or "").startswith(RUN_PREFIX):
        raise SystemExit(f"Runs com penalidade física precisam de --run-name começando por {RUN_PREFIX!r}, para não se "
                         "misturarem com os da linha de base.")
    if any(getattr(settings, f"lambda_{t}") < 0 for t in TERMS):
        raise SystemExit("Os λ precisam ser >= 0.")
    # Os parâmetros PIML entram no `config` do run (o treino o monta de vars(args)): ficam no relatório, no best.pt e
    # na impressão digital do last.pt, então retomar com outros λ é recusado como qualquer mudança de configuração.
    for name, value in vars(settings).items():
        setattr(args, name, value)
    log: list[dict[str, Any]] = []
    base.train_one_epoch = make_train_one_epoch(settings, log)
    report = base.run_training(args)
    report_path = args.report or (base.RUNS_REPORT_DIR / f"{args.run_name}.json")
    if report.get("kind") in {"full", "demo", "smoke"} and Path(report_path).exists():
        report["piml"] = {
            "loss": "BCE mascarada + Σ λ_k·L_k (stconv_train_piml.py)",
            "lambdas": {term: getattr(settings, f"lambda_{term}") for term in TERMS},
            "radius_px": settings.piml_radius,
            "delta_std": settings.piml_delta,
            "checkpoint_criterion_note": "val_loss = BCE pura, como no baseline",
            "per_epoch_this_process": log,
        }
        base.write_json(report, Path(report_path))
    return report


if __name__ == "__main__":
    main()
