"""Candidate generation: local Hugging Face (default), OpenAI, Anthropic, or SMOKE_TEST mutants.

Each sample carries the generating model's token log-probabilities (mean / min) as a confidence
signal. Anthropic's API does not expose log-probabilities, so those are NaN for Claude samples.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import log
from .astutils import make_mutants, parse_quietly

NAN = float("nan")
SYSTEM_PROMPT = "You are an expert Python programmer. Reply with Python code only."


def build_prompt(problem):
    return f"""Write a Python function that solves the following problem.

Problem: {problem['text']}

The function must be named `{problem['function_name']}` and pass this test:
{problem['visible_tests'][0]}

Return only the code (the function plus any imports it needs). Do not include tests, usage examples, or explanations."""


def extract_code(raw_text):
    """Take the first fenced block that defines a function (or the first block, or the raw text)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n?(.*?)```", raw_text, re.DOTALL)
    if blocks:
        with_def = [b for b in blocks if "def " in b]
        return (with_def[0] if with_def else blocks[0]).strip()
    return raw_text.strip()


def _logprob_stats(logprobs):
    vals = [x for x in logprobs if x is not None and math.isfinite(x)]
    if not vals:
        return NAN, NAN
    return float(np.mean(vals)), float(np.min(vals))


def _sample(raw, logprobs, error="", num_tokens=None, input_tokens=NAN):
    """`input_tokens` is this sample's share of the request's prompt tokens (prompts are sent once per
    request), so summing over samples gives the request totals."""
    mean_lp, min_lp = _logprob_stats(logprobs)
    return {"raw": raw, "code": extract_code(raw), "mean_logprob": mean_lp, "min_logprob": min_lp,
            "num_tokens": len(logprobs) if num_tokens is None else num_tokens,
            "input_tokens": input_tokens, "error": error}


_hf_cache = {}


def _dtype_kwarg():
    """`dtype=` on transformers >= 4.56 (where `torch_dtype` is deprecated), `torch_dtype=` before."""
    import transformers
    from packaging.version import Version
    return "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"


def _approx_download(model_id):
    match = re.search(r"(\d+(?:\.\d+)?)B", model_id)
    return f"about {2 * float(match.group(1)):.0f} GB" if match else "the model weights"


def _load_hf(model_id):
    if model_id not in _hf_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.quiet_third_party()
        cuda = torch.cuda.is_available()
        device = f"GPU ({torch.cuda.get_device_name(0)})" if cuda else "CPU (slow: use a GPU runtime)"
        log.info(f"Loading {model_id} on {device}. The first time, this downloads "
                 f"{_approx_download(model_id)} from Hugging Face; later runs use the local cache.")
        start = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if cuda else None,
            **{_dtype_kwarg(): torch.float16 if cuda else torch.float32},
        )
        model.eval()
        _hf_cache[model_id] = (tok, model)
        log.info(f"Model ready: {model_id} ({log.format_seconds(time.time() - start)}).")
    return _hf_cache[model_id]


def generate_local_hf(prompt, model_id, temperature, n, max_new_tokens=512):
    import torch
    tok, model = _load_hf(model_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=0.95,
            max_new_tokens=max_new_tokens, num_return_sequences=n, pad_token_id=pad_id,
            return_dict_in_generate=True, output_scores=True,
        )
    scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    scores = scores.float().cpu().numpy()
    generated = out.sequences[:, inputs["input_ids"].shape[1]:].cpu().numpy()
    stop_ids = set(np.atleast_1d(model.generation_config.eos_token_id).tolist())
    stop_ids |= {tok.eos_token_id, pad_id}
    stop_ids.discard(None)
    prompt_tokens = inputs["input_ids"].shape[1]
    samples = []
    for i in range(n):
        ids = generated[i].tolist()
        end = next((j for j, t in enumerate(ids) if t in stop_ids), len(ids))
        raw = tok.decode(ids[:end], skip_special_tokens=True)
        samples.append(_sample(raw, scores[i][:end].tolist(), input_tokens=prompt_tokens / n))
    return samples


def generate_openai(prompt, model, temperature, n):
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=model, temperature=temperature, n=n, logprobs=True,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    )
    usage = getattr(resp, "usage", None)
    input_share = usage.prompt_tokens / len(resp.choices) if usage and resp.choices else NAN
    samples = []
    for choice in resp.choices:
        content = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
        samples.append(_sample(choice.message.content or "", [t.logprob for t in content],
                               input_tokens=input_share))
    return samples


def generate_anthropic(prompt, model, temperature, n):
    import anthropic
    client = anthropic.Anthropic()
    samples = []
    for _ in range(n):
        resp = client.messages.create(
            model=model, max_tokens=1024, temperature=temperature, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = getattr(resp, "usage", None)
        samples.append(_sample(text, [], num_tokens=usage.output_tokens if usage else NAN,
                               input_tokens=usage.input_tokens if usage else NAN))
    return samples


def generate_samples(model_cfg, prompt, n, max_new_tokens=512):
    try:
        provider = model_cfg["provider"]
        if provider == "local_hf":
            return generate_local_hf(prompt, model_cfg["model"], model_cfg["temperature"], n, max_new_tokens)
        if provider == "openai":
            return generate_openai(prompt, model_cfg["model"], model_cfg["temperature"], n)
        if provider == "anthropic":
            return generate_anthropic(prompt, model_cfg["model"], model_cfg["temperature"], n)
        raise ValueError(f"Unknown provider: {provider}")
    except ImportError as e:
        extra = "llm" if model_cfg.get("provider") == "local_hf" else "api"
        error = f"{e!r} -- install the backend with `uv sync --extra {extra}`"
        return [_sample("", [], error=error[:400]) for _ in range(n)]
    except Exception as e:
        return [_sample("", [], error=repr(e)[:300]) for _ in range(n)]


def candidate_row(problem, model_name, provider, idx, sample, provenance, mutation_type=""):
    return {
        "candidate_id": f"{problem['task_id']}::{model_name}::{idx}",
        "task_id": problem["task_id"],
        "model_name": model_name,
        "provider": provider,
        "provenance": provenance,
        "mutation_type": mutation_type,
        "problem_text": problem["text"],
        "generated_code": sample.get("code", ""),
        "gen_error": sample.get("error", ""),
        "gen_mean_logprob": sample.get("mean_logprob", NAN),
        "gen_min_logprob": sample.get("min_logprob", NAN),
        "gen_num_tokens": sample.get("num_tokens", NAN),
        "gen_input_tokens": sample.get("input_tokens", NAN),
        "gen_seconds": sample.get("seconds", NAN),
    }


def mutant_rows(problem, n, model_name, provider, seed, start_idx=0):
    mutants = make_mutants(problem["reference_code"], n, seed=f"{seed}-{problem['task_id']}")
    return [candidate_row(problem, model_name, provider, start_idx + j, {"code": m}, "mutant", kind)
            for j, (m, kind) in enumerate(mutants)]


def generate_candidates(problems, cfg, cache_path):
    """Returns (candidates DataFrame, number of rows dropped because the generation call failed).

    LLM samples are cached in `cache_path` (JSONL) so interrupted runs resume; failed calls are not
    cached. Every row records its provenance: `llm`, `reference` or `mutant`.
    """
    records = []
    if cfg.smoke_test:
        log.info(f"SMOKE_TEST: no LLM calls. Using each problem's reference solution plus "
                 f"{cfg.smoke_mutants_per_problem} AST mutants of it as candidates.")
        start = time.time()
        for p in problems:
            # Normalise the reference through ast.unparse so it is formatted like its mutants.
            reference = ast.unparse(parse_quietly(p["reference_code"]))
            records.append(candidate_row(p, "smoke", "smoke", 0, {"code": reference}, "reference"))
            records += mutant_rows(p, cfg.smoke_mutants_per_problem, "smoke", "smoke", cfg.seed, start_idx=1)
        session = {"smoke": {"problems": len(problems), "attempts_requested": len(records),
                             "new_samples": len(records), "reused_samples": 0, "failed_attempts": 0,
                             "session_seconds": time.time() - start}}
    else:
        records, session = _generate_llm_candidates(problems, cfg, cache_path)
        if cfg.use_mutant_augmentation:
            log.info(f"Adding {cfg.mutants_per_problem} reference mutants per problem "
                     "(training-only augmentation).")
            for p in problems:
                records += mutant_rows(p, cfg.mutants_per_problem, "augmentation", "mutation", cfg.seed)

    df = pd.DataFrame(records)
    failed = df["gen_error"] != ""
    if failed.any():
        log.warning(f"dropping {int(failed.sum())} rows whose generation call failed, e.g.: "
                    f"{df.loc[failed, 'gen_error'].iloc[0]}")
    df = df[~failed].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No candidates were generated -- fix the generation error printed above.")
    # Per-config counts for this session (cache reuse, failures); read by the run report.
    df.attrs["generation_session"] = session
    return df, int(failed.sum())


def _read_cache(cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache.setdefault((r["task_id"], r["model_name"]), []).append(r)
    return cache


def _generate_llm_candidates(problems, cfg, cache_path):
    """One pass per model config, so each model is loaded once and only if something is missing."""
    cache = _read_cache(cache_path)
    configs = cfg.resolved_model_configs()
    n = cfg.samples_per_config
    if cache:
        log.info(f"Found a generation cache at {cache_path}; already generated samples are reused.")
    start_all = time.time()
    session = {}
    with open(cache_path, "a") as f:
        for i, model_cfg in enumerate(configs, 1):
            name = model_cfg["name"]
            todo = [p for p in problems if len(cache.get((p["task_id"], name), [])) < n]
            cached = len(problems) - len(todo)
            stats = session[name] = {"problems": len(problems), "attempts_requested": len(problems) * n,
                                     "new_samples": 0, "reused_samples": cached * n,
                                     "failed_attempts": 0, "session_seconds": 0.0}
            temperature = model_cfg.get("temperature", 0.0)
            log.info(f"[{i}/{len(configs)}] {name}: {model_cfg.get('model', model_cfg['provider'])} "
                     f"at temperature {temperature}, {n} samples per problem. "
                     f"{cached}/{len(problems)} problems cached, {len(todo)} to generate.")
            if not todo:
                continue
            start, errors = time.time(), 0
            prompts = (build_prompt(p) for p in todo)
            bar = tqdm(zip(todo, prompts), total=len(todo), desc=f"  generating {name}", unit="problem",
                       file=sys.stdout, mininterval=1.0, disable=not log.progress_enabled())
            for p, prompt in bar:
                call_start = time.time()
                samples = generate_samples(model_cfg, prompt, n, cfg.max_new_tokens)
                per_sample = (time.time() - call_start) / max(len(samples), 1)
                rows = [candidate_row(p, name, model_cfg["provider"], j, {**s, "seconds": per_sample}, "llm")
                        for j, s in enumerate(samples)]
                if any(r["gen_error"] for r in rows):
                    if errors == 0:
                        log.warning(f"{name} failed on task {p['task_id']}: {rows[0]['gen_error'][:200]} "
                                    "(failed calls are not cached and will be retried on the next run)")
                    errors += 1
                    stats["failed_attempts"] += len(rows)
                else:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                    stats["new_samples"] += len(rows)
                cache[(p["task_id"], name)] = rows
                bar.set_postfix(errors=errors, refresh=False)
            bar.close()
            stats["session_seconds"] = time.time() - start
            log.info(f"[{i}/{len(configs)}] {name}: generated {len(todo) - errors} problems "
                     f"({stats['new_samples']} samples), {errors} failed, "
                     f"in {log.format_seconds(stats['session_seconds'])}.")
    log.info(f"Generation finished in {log.format_seconds(time.time() - start_all)}.")
    records = [r for p in problems for model_cfg in configs
               for r in cache.get((p["task_id"], model_cfg["name"]), [])[:n]]
    return records, session
