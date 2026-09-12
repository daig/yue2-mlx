"""Offline generation, serial batches, replay and local runtime diagnostics."""
from __future__ import annotations

import argparse
import json
import importlib
import importlib.metadata
import os
import platform
import re
from pathlib import Path
import sys
import time
import uuid

from yue2.protocol import GenerationConfig
from yue2.storage import identity, write_json



def _pipeline(args, generation_config=None):
    from .pipeline import YuE2Pipeline
    from .conversion import MODEL_REPO, VAE_REPO

    return YuE2Pipeline.from_pretrained(
        args.model or MODEL_REPO, vae=args.vae or VAE_REPO,
        converted_dir=args.converted_dir, precision=args.precision,
        local_files_only=args.offline, memory_budget_gib=args.memory_budget_gib,
        vae_core_frames=args.vae_core_frames, progress=not args.quiet,
        generation_config=generation_config, require_ac=args.require_ac,
    )


def _request(args, data=None, base=None):
    from yue2.cli import request_kwargs

    if data is None:
        path = getattr(args, "request_file", None) or getattr(args, "request", None)
        data = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        base = Path.cwd() if path is None else Path(path).parent
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    data = dict(data)
    for key in ("id", "style", "seed", "cfg_scale", "lyrics"):
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
            if key == "style":
                data.pop("tags", None)
            if key == "lyrics":
                data.pop("lyrics_path", None)
    if getattr(args, "mode", None) is not None:
        data["cot"] = args.mode
    for field in ("lyrics", "abc"):
        path = getattr(args, field + "_file", None)
        if path is not None:
            data[field] = Path(path).read_bytes().decode("utf-8")
            data.pop(field + "_path", None)
        elif field + "_path" in data:
            if data.get(field) is not None:
                raise ValueError(f"Pass {field} or {field}_path, not both")
            data[field] = (base / data.pop(field + "_path")).read_bytes().decode("utf-8")
    generation = data.pop("generation_config", None)
    result = request_kwargs(data, base or Path.cwd())
    if generation is not None:
        result["generation_config"] = generation
    return result


def _failure(path, exc, **metadata):
    receipt = {"status": "failed", "type": type(exc).__name__, "reason": str(exc), **metadata}
    try:
        write_json(path, receipt)
    except Exception as error:
        receipt["receipt_error"] = f"{type(error).__name__}: {error}"
        print(f"Could not write failure receipt {path}: {error}", file=sys.stderr)
    return receipt


def _managed_artifact(name, allowed):
    if name in allowed:
        return True
    if not name.endswith(".tmp"):
        return False
    stem = name[:-4]
    base, separator, pid = stem.rpartition(".")
    return bool(separator and pid.isdigit() and base in allowed)


def _generate(args, request):
    """Use the native result protocol; attempt receipts only authorize safe retries."""
    from .artifacts import load_artifacts

    request = dict(request)
    generation = request.pop("generation_config", None)
    config = None if generation is None else GenerationConfig.from_dict(generation)
    directory = args.output
    attempt = None
    with _pipeline(args, config) as pipe:
        normalized = pipe._request(**{
            k: v for k, v in request.items() if k not in {"abc_sampling", "semantic_sampling"}
        })
        effective = pipe.effective_config(
            normalized, request.get("abc_sampling"), request.get("semantic_sampling"),
        )
        expected = identity({"request": normalized.to_dict(), "config": effective, "weights": pipe.weights})
        if directory is None:
            directory = Path("runs/default") / normalized.id
            args.output = directory
        attempt = directory.with_name(directory.name + ".attempt.json")
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise FileExistsError(f"Output is not a regular directory: {directory}")
        if args.resume and (directory / "result.json").exists():
            saved_result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if saved_result.get("status") == "complete" and not attempt.exists():
                saved = load_artifacts(directory)
                if saved.result["identity"] != expected:
                    raise ValueError("Request/config/weight identity changed; use a new output directory")
                return {"output": str(directory), "resumed": True, **saved.result}
        if directory.exists() and any(directory.iterdir()):
            if not args.resume or not attempt.is_file() or attempt.is_symlink():
                raise FileExistsError(f"Nonempty output {directory}; use a new output directory")
            receipt = json.loads(attempt.read_text(encoding="utf-8"))
            if receipt.get("identity") != expected or receipt.get("status") not in {"running", "failed"}:
                raise ValueError("Prior attempt identity changed; use a new output directory")
            allowed = {
                "request.json", "config.json", "result.json", "failure.json", "noise.npy",
                "audio.flac", "prefix.npy", "semantic.npy", "latent.npy", "plan.json",
                "plan_manifest.json", "score.abc", "abc_tokens.npy",
            }
            if any(
                p.is_symlink() or not p.is_file() or not _managed_artifact(p.name, allowed)
                for p in directory.iterdir()
            ):
                raise FileExistsError("Interrupted output contains unrelated artifacts; use a new output directory")
            archive = directory.with_name(directory.name + ".interrupted-" + uuid.uuid4().hex)
            directory.rename(archive)
        if attempt.exists():
            if not args.resume or attempt.is_symlink():
                raise FileExistsError(f"Prior attempt receipt exists: {attempt}; use --resume or a new output")
            prior = json.loads(attempt.read_text(encoding="utf-8"))
            if prior.get("identity") != expected:
                raise ValueError("Prior attempt identity changed; use a new output directory")
        write_json(attempt, {"status": "running", "identity": expected})
        try:
            monitor_args = argparse.Namespace(**vars(args))
            if args.resume:
                monitor_args.output = directory.with_name(directory.name + ".retry-" + uuid.uuid4().hex)
            with _resource_monitor(monitor_args):
                result = pipe(**request)
                receipt = result.save_artifacts(directory)
        except BaseException as exc:
            _failure(attempt, exc, identity=expected)
            _failure(directory / "failure.json", exc, identity=expected)
            raise
        try:
            attempt.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Could not remove completed attempt receipt {attempt}: {exc}", file=sys.stderr)
        return {"output": str(directory), "resumed": False, **receipt}


def _batch(args):
    path = Path(args.input)
    rows = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            song_id = row.get("id") if isinstance(row, dict) else None
            if (not isinstance(song_id, str) or not song_id or song_id in {".", ".."}
                    or any(c in song_id for c in "/\\\x00")
                    or song_id.endswith((".json", ".jsonl")) or song_id in ids):
                raise ValueError("Every batch row needs a unique, single-component string id")
            ids.add(song_id)
            rows.append((line_number, row, None))
        except (ValueError, TypeError) as exc:
            rows.append((line_number, None, exc))
    output = args.output
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError(f"Batch output is not a regular directory: {output}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty batch output; use --resume or a new output directory")
    if output.exists() and any(output.iterdir()):
        previous_path = output / "batch.json"
        if previous_path.is_symlink() or not previous_path.is_file():
            raise FileExistsError("Nonempty batch output lacks a batch receipt; use a new directory")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or not isinstance(previous.get("results"), list):
            raise ValueError("Invalid prior batch receipt; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    report = {"complete": False, "expected": len(rows), "failed": 0, "results": receipts}
    write_json(output / "batch.json", report)
    for line_number, row, error in rows:
        song_id = None if row is None else row["id"]
        if not args.quiet:
            print(f"Batch line {line_number}: {song_id or 'invalid request'}", file=sys.stderr, flush=True)
        try:
            if error is not None:
                raise error
            row_args = argparse.Namespace(**vars(args))
            row_args.output = output / song_id
            request = _request(row_args, row, path.parent)
            receipt = _generate(row_args, request)
            receipts.append({"line": line_number, "id": song_id, "status": "complete",
                             "identity": receipt["identity"], "resumed": receipt["resumed"]})
        except Exception as exc:
            report["failed"] += 1
            # Keep failures outside song directories: rejected outputs must remain untouched.
            receipts.append(_failure(output / f"line-{line_number}.{uuid.uuid4().hex}.failure.json", exc,
                                     line=line_number, id=song_id))
        report["complete"] = len(receipts) == len(rows) and not report["failed"]
        try:
            write_json(output / "batch.json", report)
        except Exception as exc:
            print(f"Could not update batch receipt: {exc}", file=sys.stderr)
            return 1
    if not rows:
        report["complete"] = True
        write_json(output / "batch.json", report)
    return int(bool(report["failed"]))

def _doctor(args):
    versions, errors = {}, {}
    packages = {
        "mlx": "mlx.core", "mlx-lm": "mlx_lm", "torch": "torch",
        "transformers": "transformers", "huggingface-hub": "huggingface_hub",
        "safetensors": "safetensors", "tiktoken": "tiktoken", "numpy": "numpy",
        "soundfile": "soundfile", "psutil": "psutil",
    }
    modules = {}
    for package, module in packages.items():
        try:
            versions[package] = importlib.metadata.version(package)
            modules[package] = importlib.import_module(module)
        except Exception as exc:
            versions.setdefault(package, None)
            errors[package] = f"{type(exc).__name__}: {exc}"
    backends = {"mlx_metal": False, "torch_mps": False}
    for name, package, probe in (
        ("mlx_metal", "mlx", lambda m: m.metal.is_available()),
        ("torch_mps", "torch", lambda m: m.backends.mps.is_available()),
    ):
        if package in modules:
            try:
                backends[name] = bool(probe(modules[package]))
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
    mac_version = platform.mac_ver()[0]
    mac_parts = tuple(int(v) for v in mac_version.split(".")[:2]) if mac_version else ()
    runtime = {
        "system": platform.system(), "macos": mac_version, "machine": platform.machine(),
        "python": platform.python_version(),
        "supported_os": platform.system() == "Darwin" and mac_parts >= (26, 2),
        "supported_arch": platform.machine() == "arm64",
        "supported_python": sys.version_info[:2] == (3, 12),
        "unsafe_environment": {key: os.environ[key] for key in
            ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH") if os.environ.get(key) == "1"},
    }
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        runtime["unsafe_environment"]["MLX_ENABLE_TF32"] = os.environ.get("MLX_ENABLE_TF32")
    report = {
        "dependencies_ready": not errors, "versions": versions, "errors": errors,
        "runtime": runtime, "backends": backends,
        "model": str(Path(args.model or args.converted_dir).expanduser()),
        "vae": None if args.vae is None else str(Path(args.vae).expanduser()),
        "hashes_verified": False, "validated": False,
        "note": "Readiness is not quality, performance, or real-memory acceptance. Diagnostics never download models.",
    }
    if args.verify_hashes:
        try:
            from .conversion import verify_conversion
            from yue2.storage import model_identity

            if args.vae is None:
                raise ValueError("--verify-hashes requires --vae pointing to a local checkpoint")
            model, vae = Path(report["model"]), Path(report["vae"])
            if not model.is_dir() or not vae.is_dir():
                raise FileNotFoundError("Hash verification requires local model and VAE directories")
            weights = {"mot": verify_conversion(model), "vae": model_identity(vae)}
            if not (model / f"ar-{args.precision}.safetensors").is_file():
                raise FileNotFoundError(f"Converted AR precision is unavailable: {args.precision}")
            report["weights"] = weights
            report["hashes_verified"] = True
        except Exception as exc:
            errors["weights"] = f"{type(exc).__name__}: {exc}"
    report["ready"] = (
        report["dependencies_ready"] and all(backends.values())
        and all(runtime[key] for key in ("supported_os", "supported_arch", "supported_python"))
        and not runtime["unsafe_environment"] and (not args.verify_hashes or report["hashes_verified"])
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(not report["ready"])


def _save(result, output):
    result.save_artifacts(output)
    print(json.dumps({"output": str(output), "sample_rate": result.sample_rate,
                      "audio_seconds": len(result.audio) / result.sample_rate,
                      "truncated": result.truncated, "timing": result.timing}, indent=2))


def _resource_monitor(args):
    from .measure import ResourceMonitor

    output = Path(args.output)
    return ResourceMonitor(
        require_ac=args.require_ac,
        log_path=output.with_name(output.name + ".resources.jsonl"),
        report_path=output.with_name(output.name + ".resources.json"),
        metadata={
            "command": args.command,
            "memory_budget_gib": args.memory_budget_gib,
            "vae_core_frames": args.vae_core_frames,
            "precision": args.precision,
        },
    )


def _render_plan(pipe, plan, semantic_sampling=None):
    from .pipeline import SongResult, initial_noise

    start = time.perf_counter()
    semantic = pipe.generate_semantic(plan, sampling=semantic_sampling)
    noise = initial_noise(len(semantic.tokens), plan.request.seed)
    nar_start = time.perf_counter()
    latents = pipe.synthesize(semantic, noise=noise)
    nar_seconds = time.perf_counter() - nar_start
    vae_start = time.perf_counter()
    audio = pipe.decode(latents)
    config = pipe.effective_config(plan.request, semantic_sampling=semantic_sampling)
    stamp = identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights})
    timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
              "vae_seconds": time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
              "e2e_seconds": time.perf_counter() - start}
    return SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, stamp, noise)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Fetch pinned checkpoints and convert the generator")
    prep.add_argument("--source")
    prep.add_argument("--output", default="models/converted")
    prep.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
    prep.add_argument("--offline", action="store_true")
    prep.add_argument("--cache-dir")
    for name in ("generate", "plan", "render-plan", "replay", "batch", "doctor"):
        command = commands.add_parser(name)
        if name in {"generate", "plan"}:
            command.add_argument("request", nargs="?", help="Request JSON (optional with inline flags)")
            command.add_argument("--request", dest="request_file", help="Request JSON")
        elif name in {"render-plan", "replay"}:
            command.add_argument("request", help="Saved plan directory or saved song directory")
        command.add_argument(
            "--output",
            required=name in {"render-plan", "replay"},
            default=Path("runs/batch") if name == "batch" else None,
            type=Path,
        )
        command.add_argument("--model", help="Converted directory or pinned source checkpoint")
        command.add_argument("--vae", help="Local default VAE checkpoint directory")
        command.add_argument("--converted-dir", default="models/converted")
        command.add_argument("--precision", choices=("bf16", "8bit", "4bit"), default="bf16")
        command.add_argument("--offline", action="store_true")
        command.add_argument("--quiet", "--no-progress", action="store_true")
        command.add_argument("--memory-budget-gib", type=float)
        command.add_argument("--vae-core-frames", type=int, default=256)
        command.add_argument("--require-ac", action="store_true")
        if name in {"generate", "plan", "batch"}:
            command.add_argument("--mode", "--cot", choices=("full", "melody", "off"))
        if name in {"generate", "plan"}:
            command.add_argument("--abc", "--abc-file", dest="abc_file",
                                 help="Supplied ABC file; UTF-8 bytes are preserved")
            command.add_argument("--id")
            command.add_argument("--style", "--tags", dest="style")
            command.add_argument("--lyrics")
            command.add_argument("--lyrics-file")
            command.add_argument("--seed", type=int)
            command.add_argument("--cfg-scale", type=float)
        if name in {"generate", "batch"}:
            command.add_argument("--resume", action="store_true")
        if name == "batch":
            command.add_argument("--input", required=True)
            command.add_argument("--concurrency", type=int, choices=(1,), default=1)
        if name == "doctor":
            command.add_argument("--verify-hashes", action="store_true")
        if name == "replay":
            command.add_argument("--stage", choices=("synthesize", "decode"), default="decode")
    return parser


def main(argv=None):
    cli_parser = parser()
    args = cli_parser.parse_args(argv)
    if args.command in {"generate", "plan"} and args.request and args.request_file:
        cli_parser.error("Pass request JSON either positionally or with --request, not both")
    if args.command == "doctor":
        return _doctor(args)
    if args.command != "prepare" and args.memory_budget_gib is None:
        from .measure import DEFAULT_MEMORY_BUDGET_GIB

        args.memory_budget_gib = DEFAULT_MEMORY_BUDGET_GIB
    if args.command == "batch":
        return _batch(args)
    if args.command in {"generate", "plan"}:
        try:
            request = _request(args)
            if request.get("style", request.get("tags")) is None or request.get("lyrics") is None:
                raise ValueError("Provide style and lyrics in request JSON or inline flags")
            if request.get("generation_config") is not None:
                GenerationConfig.from_dict(request["generation_config"])
            if args.output is None:
                song_id = request.get("id", "song")
                if (
                    not isinstance(song_id, str)
                    or song_id in {".", ".."}
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", song_id)
                ):
                    raise ValueError("id must be a filename-safe identifier")
                args.output = Path("runs/default") / song_id
        except (OSError, ValueError, TypeError) as exc:
            cli_parser.error(str(exc))
    if args.command == "generate":
        if not args.resume and args.output.exists() and (
            not args.output.is_dir() or any(args.output.iterdir())
        ):
            cli_parser.error("Output directory must be empty; recordings are never silently overwritten")
        print(json.dumps(_generate(args, request), indent=2, ensure_ascii=False))
        return 0
    if args.command == "prepare":
        from .conversion import fetch_models, prepare
        model, vae = fetch_models(cache_dir=args.cache_dir, local_files_only=args.offline)
        output = prepare(args.source or model, args.output, precision=args.precision)
        print(json.dumps({"model": str(output), "vae": str(vae)}))
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        cli_parser.error("Output directory must be empty; recordings are never silently overwritten")
    with _resource_monitor(args):
        if args.command == "plan":
            generation = request.pop("generation_config", None)
            config = None if generation is None else GenerationConfig.from_dict(generation)
            with _pipeline(args, config) as pipe:
                from .artifacts import save_plan_artifacts

                abc_sampling = request.pop("abc_sampling", None)
                semantic_sampling = request.pop("semantic_sampling", None)
                plan = pipe.plan(**request, abc_sampling=abc_sampling)
                effective = pipe.effective_config(
                    plan.request, abc_sampling, semantic_sampling,
                )
                save_plan_artifacts(
                    plan, args.output, GenerationConfig.from_dict(effective["generation"]),
                )
            print(json.dumps({"output": str(args.output), "truncated": plan.truncated,
                              "timing": plan.timing}, indent=2))
        elif args.command == "render-plan":
            from .artifacts import load_plan_artifacts

            plan, config = load_plan_artifacts(args.request)
            with _pipeline(args, config) as pipe:
                result = _render_plan(pipe, plan)
            _save(result, args.output)
        else:
            from .artifacts import load_artifacts
            from .pipeline import SongResult, initial_noise

            saved = load_artifacts(args.request)
            config = GenerationConfig.from_dict(saved.config["generation"])
            with _pipeline(args, config) as pipe:
                start = time.perf_counter()
                noise = saved.noise
                noise_source = "source_artifact" if noise is not None else "unavailable"
                latents = saved.latents
                nar_seconds = 0.0
                if args.stage == "synthesize":
                    if noise is None:
                        noise = initial_noise(
                            len(saved.semantic.tokens), saved.semantic.plan.request.seed,
                        )
                        noise_source = "regenerated_from_request_seed"
                    nar_start = time.perf_counter()
                    latents = pipe.synthesize(saved.semantic, noise=noise)
                    nar_seconds = time.perf_counter() - nar_start
                vae_start = time.perf_counter()
                audio = pipe.decode(latents)
                timing = {
                    "abc": saved.semantic.plan.timing,
                    "semantic": saved.semantic.timing,
                    "nar_seconds": nar_seconds,
                    "vae_seconds": time.perf_counter() - vae_start,
                    "load": dict(pipe.load_timing),
                    "e2e_seconds": time.perf_counter() - start,
                    "replay": {
                        "stage": args.stage,
                        "reused_latents": args.stage == "decode",
                    },
                }
                effective = pipe.effective_config(saved.semantic.plan.request)
                effective["replay"] = {
                    "stage": args.stage,
                    "source_identity": saved.result["identity"],
                    "source_config": saved.config,
                    "source_weights": saved.result["weights"],
                    "source_artifacts": saved.result["artifacts"],
                    "source_timing": saved.result["timing"],
                    "source_truncated": saved.result["truncated"],
                    "noise_source": noise_source,
                    "noise_used_for_synthesis": args.stage == "synthesize",
                }
                stamp = identity({
                    "request": saved.semantic.plan.request.to_dict(),
                    "config": effective,
                    "weights": pipe.weights,
                })
                result = SongResult(
                    audio, 48000, saved.semantic, latents, effective,
                    pipe.weights, timing, stamp, noise,
                )
            _save(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
