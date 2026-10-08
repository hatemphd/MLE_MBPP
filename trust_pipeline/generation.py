"""Candidate generation: local Hugging Face (default), OpenAI, Anthropic, or SMOKE_TEST mutants.

Each sample carries the generating model's token log-probabilities (mean / min) as a confidence
signal. Anthropic's API does not expose log-probabilities, so those are NaN for Claude samples.
"""
from __future__ import annotations

import ast
import io
import json
import math
import os
import re
import sys
import time
import tokenize

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import log
from .astutils import expected_call, make_mutants, parse_quietly

NAN = float("nan")
SYSTEM_PROMPT = "You are an expert Python programmer. Reply with Python code only."


# Prompt styles. `standard` is the core sensor; the others are an optional extra (--prompt-styles).
# Only `inject_bug` asks for a defect; its samples get provenance `injected` (training-only).
STANDARD = "standard"
INJECT_BUG = "inject_bug"
NATURAL_STYLES = [STANDARD, "signature_only", "step_by_step", "constrained"]
PROMPT_STYLES = NATURAL_STYLES + [INJECT_BUG]
PROMPT_STYLE_PRESETS = {"all": PROMPT_STYLES, "natural": NATURAL_STYLES}


def resolve_prompt_styles(spec):
    """'standard,step_by_step' / 'all' / 'natural' / list -> ordered list of known style names."""
    items = spec.split(",") if isinstance(spec, str) else list(spec or [STANDARD])
    styles = []
    for item in (s.strip() for s in items):
        for style in PROMPT_STYLE_PRESETS.get(item, [item]):
            if style not in PROMPT_STYLES:
                raise ValueError(f"unknown prompt style {style!r}; choose from {PROMPT_STYLES} "
                                 f"or a preset {sorted(PROMPT_STYLE_PRESETS)}")
            if style not in styles:
                styles.append(style)
    return styles or [STANDARD]


def provenance_for(style):
    return "injected" if style == INJECT_BUG else "natural"


def function_signature(problem):
    """`def name(arg1, arg2):` from the call in the visible test (name and arity only)."""
    name, call = expected_call(problem["visible_tests"][0])
    name = name or problem["function_name"]
    params = [f"arg{i + 1}" for i in range(len(call.args))] if call is not None else []
    params += [k.arg for k in (call.keywords if call is not None else []) if k.arg]
    return f"def {name}({', '.join(params)}):"


_RETURN_CODE_ONLY = ("Return only the code (the function plus any imports it needs). "
                     "Do not include tests, usage examples, or explanations.")


def build_prompt(problem, style=STANDARD):
    task = f"Write a Python function that solves the following problem.\n\nProblem: {problem['text']}\n\n"
    test = (f"The function must be named `{problem['function_name']}` and pass this test:\n"
            f"{problem['visible_tests'][0]}\n\n")
    if style == STANDARD:
        return task + test + _RETURN_CODE_ONLY
    if style == "signature_only":
        return (task + f"Use exactly this signature:\n{function_signature(problem)}\n\n" + _RETURN_CODE_ONLY)
    if style == "step_by_step":
        return (task + test + "First reason briefly, step by step, about the approach and the edge cases "
                "(a few short sentences). Then give the complete final function in a single ```python "
                "code block at the end.")
    if style == "constrained":
        return (task + test + "Write a concise solution: a single function using only the Python standard "
                "library, preferring built-ins, with no comments, no docstrings and no print statements. "
                + _RETURN_CODE_ONLY)
    if style == INJECT_BUG:
        return (task + test + "For a software-testing study, write a solution that looks correct and "
                "plausible but contains exactly one subtle, realistic bug of the kind programmers really "
                "make: an off-by-one error, a missing edge case, a wrong comparison or arithmetic operator, "
                "or a wrong loop boundary. Prefer a bug that still passes the test above but fails on some "
                "other inputs. Do not mention, hint at or comment on the bug anywhere: no comments, no "
                "docstrings. " + _RETURN_CODE_ONLY)
    raise ValueError(f"unknown prompt style {style!r}")


def extract_code(raw_text, prefer_last=False):
    """Take the first (or, for reasoning-style answers, the last) fenced block that defines a
    function, else the first/last block, else the raw text."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n?(.*?)```", raw_text, re.DOTALL)
    if blocks:
        with_def = [b for b in blocks if "def " in b] or blocks
        return (with_def[-1] if prefer_last else with_def[0]).strip()
    return raw_text.strip()


def strip_comments(code):
    """Remove comments and docstrings (for injected samples, so nothing gives the bug away).
    Returns the input unchanged if it does not tokenize/parse."""
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                  if t.type != tokenize.COMMENT]
        code = tokenize.untokenize(tokens)
        tree = parse_quietly(code)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return code
    drop = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and len(body) > 1 and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    lines = [line.rstrip() for i, line in enumerate(code.splitlines(), 1) if i not in drop]
    return "\n".join(line for line in lines if line.strip())


def _is_main_guard(node):
    test = node.test
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__")


def _calls_any(node, names):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
               for n in ast.walk(node))


def strip_test_code(code):
    """Remove the model's own top-level test code (asserts, prints and other bare expressions,
    `if __name__ == "__main__":` blocks, loops, and assignments that call the candidate's functions)
    so a wrong self-written example can't crash an otherwise valid solution at import time.
    Imports, definitions and constant assignments are kept. Returns the input unchanged if it
    does not parse."""
    try:
        tree = parse_quietly(code)
    except (SyntaxError, ValueError):
        return code
    defined = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    drop = set()
    for node in tree.body:
        is_test = (
            isinstance(node, (ast.Assert, ast.Expr, ast.For, ast.AsyncFor, ast.While))
            or (isinstance(node, ast.If) and _is_main_guard(node))
            or (isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and _calls_any(node, defined))
        )
        if is_test:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            drop.update(range(start, node.end_lineno + 1))
    if not drop:
        return code
    source = code.splitlines()
    for start in sorted(drop):
        i = start - 1
        while i >= 1 and i not in drop and source[i - 1].strip().startswith("#"):
            drop.add(i)
            i -= 1
    lines = [line for i, line in enumerate(source, 1) if i not in drop]
    return "\n".join(lines).strip()


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


def candidate_row(problem, model_name, provider, idx, sample, provenance, mutation_type="",
                  prompt_style="none", model="", temperature=NAN):
    """One candidate. `prompt_style` is `none` for reference solutions and mutants."""
    style_part = "" if prompt_style in (STANDARD, "none") else f"{prompt_style}::"
    return {
        "candidate_id": f"{problem['task_id']}::{model_name}::{style_part}{idx}",
        "task_id": problem["task_id"],
        "model_name": model_name,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "prompt_style": prompt_style,
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
    return [candidate_row(problem, model_name, provider, start_idx + j, {"code": m}, "mutant", kind,
                          model="reference+mutation")
            for j, (m, kind) in enumerate(mutants)]


def postprocess_sample(sample, style):
    """Style-specific code extraction: last code block for step_by_step answers; comments and
    docstrings stripped from injected samples so nothing gives the bug away."""
    if sample.get("error"):
        return sample
    if style == "step_by_step":
        sample = {**sample, "code": extract_code(sample.get("raw", ""), prefer_last=True)}
    if style == INJECT_BUG:
        sample = {**sample, "code": strip_comments(sample.get("code", ""))}
    return sample


def generate_candidates(problems, cfg, cache_path):
    """Returns (candidates DataFrame, number of rows dropped because the generation call failed).

    LLM samples are cached in `cache_path` (JSONL) so interrupted runs resume; failed calls are not
    cached. Every row records its provenance: `natural` (LLM, normal prompt), `injected` (LLM asked
    for a subtle bug; training-only), `reference` or `mutant` (training-only outside SMOKE_TEST).
    """
    records = []
    if cfg.smoke_test:
        log.info(f"SMOKE_TEST: no LLM calls. Using each problem's reference solution plus "
                 f"{cfg.smoke_mutants_per_problem} AST mutants of it as candidates.")
        if cfg.resolved_prompt_styles() != [STANDARD]:
            log.info("Prompt styles are ignored in SMOKE_TEST (no prompts are sent).")
        start = time.time()
        for p in problems:
            # Normalise the reference through ast.unparse so it is formatted like its mutants.
            reference = ast.unparse(parse_quietly(p["reference_code"]))
            records.append(candidate_row(p, "smoke", "smoke", 0, {"code": reference}, "reference",
                                         model="reference"))
            records += mutant_rows(p, cfg.smoke_mutants_per_problem, "smoke", "smoke", cfg.seed, start_idx=1)
        session = {"smoke": {"model_config": "smoke", "prompt_style": "none", "problems": len(problems),
                             "attempts_requested": len(records), "new_samples": len(records),
                             "reused_samples": 0, "failed_attempts": 0,
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
    """Cache key (task_id, model_name, prompt_style). Entries written before prompt styles existed
    have no `prompt_style` and are the standard prompt."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                style = r.get("prompt_style") or STANDARD
                cache.setdefault((r["task_id"], r["model_name"], style), []).append(r)
    return cache


def config_label(model_name, style):
    """`qwen0.5b-t0.3` for the standard prompt, `qwen0.5b-t0.3/step_by_step` otherwise."""
    return model_name if style in (STANDARD, "none", "", None) else f"{model_name}/{style}"


def _generate_llm_candidates(problems, cfg, cache_path):
    """One pass per (model config, prompt style), so each model is loaded once and only if
    something is missing."""
    cache = _read_cache(cache_path)
    jobs = [(m, s) for m in cfg.resolved_model_configs() for s in cfg.resolved_prompt_styles()]
    n = cfg.samples_per_config
    if cache:
        log.info(f"Found a generation cache at {cache_path}; already generated samples are reused.")
    start_all = time.time()
    session = {}
    with open(cache_path, "a") as f:
        for i, (model_cfg, style) in enumerate(jobs, 1):
            name = model_cfg["name"]
            label = config_label(name, style)
            provenance = provenance_for(style)
            todo = [p for p in problems if len(cache.get((p["task_id"], name, style), [])) < n]
            cached = len(problems) - len(todo)
            stats = session[label] = {"model_config": name, "prompt_style": style, "problems": len(problems),
                                      "attempts_requested": len(problems) * n, "new_samples": 0,
                                      "reused_samples": cached * n, "failed_attempts": 0,
                                      "session_seconds": 0.0}
            temperature = model_cfg.get("temperature", 0.0)
            log.info(f"[{i}/{len(jobs)}] {label}: {model_cfg.get('model', model_cfg['provider'])} "
                     f"at temperature {temperature}, prompt style '{style}', {n} samples per problem. "
                     f"{cached}/{len(problems)} problems cached, {len(todo)} to generate.")
            if not todo:
                continue
            start, errors = time.time(), 0
            prompts = (build_prompt(p, style) for p in todo)
            bar = tqdm(zip(todo, prompts), total=len(todo), desc=f"  generating {label}", unit="problem",
                       file=sys.stdout, mininterval=1.0, disable=not log.progress_enabled())
            for p, prompt in bar:
                call_start = time.time()
                samples = generate_samples(model_cfg, prompt, n, cfg.max_new_tokens)
                per_sample = (time.time() - call_start) / max(len(samples), 1)
                rows = [candidate_row(p, name, model_cfg["provider"], j,
                                      {**postprocess_sample(s, style), "seconds": per_sample}, provenance,
                                      prompt_style=style, model=model_cfg.get("model", ""),
                                      temperature=temperature)
                        for j, s in enumerate(samples)]
                if any(r["gen_error"] for r in rows):
                    if errors == 0:
                        log.warning(f"{label} failed on task {p['task_id']}: {rows[0]['gen_error'][:200]} "
                                    "(failed calls are not cached and will be retried on the next run)")
                    errors += 1
                    stats["failed_attempts"] += len(rows)
                else:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                    stats["new_samples"] += len(rows)
                cache[(p["task_id"], name, style)] = rows
                bar.set_postfix(errors=errors, refresh=False)
            bar.close()
            stats["session_seconds"] = time.time() - start
            log.info(f"[{i}/{len(jobs)}] {label}: generated {len(todo) - errors} problems "
                     f"({stats['new_samples']} samples), {errors} failed, "
                     f"in {log.format_seconds(stats['session_seconds'])}.")
    log.info(f"Generation finished in {log.format_seconds(time.time() - start_all)}.")
    # Older cache entries lack model/temperature/prompt_style and say provenance `llm`; the
    # (model config, style) they were generated under is known from the key, so fill them in.
    # Self-written test code is stripped here, so cached samples get it too without regenerating.
    records = [{**r, "model": model_cfg.get("model", ""), "temperature": model_cfg.get("temperature", NAN),
                "prompt_style": style, "provenance": provenance_for(style),
                "generated_code": strip_test_code(r.get("generated_code") or "")}
               for p in problems for model_cfg, style in jobs
               for r in cache.get((p["task_id"], model_cfg["name"], style), [])[:n]]
    return records, session
