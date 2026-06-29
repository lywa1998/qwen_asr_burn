#!/usr/bin/env python3
"""Grade eval outputs against assertions. Run from burn-workspace/."""

import json, os, re, sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
ITER = f"{WORKSPACE}/iteration-1"

def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""

def grade_eval(eval_dir, config):
    """Run assertions for one (eval, config) pair. Returns grading dict."""
    meta_path = f"{eval_dir}/eval_metadata.json"
    meta = json.loads(read_file(meta_path))
    assertions = meta["assertions"]
    outputs_dir = f"{eval_dir}/{config}/outputs"
    all_files = read_file(f"{outputs_dir}/ALL_FILES")  # will be empty
    # Collect all output file contents for grep
    files = {}
    for fname in os.listdir(outputs_dir):
        path = f"{outputs_dir}/{fname}"
        if os.path.isfile(path):
            files[fname] = read_file(path)
    combined = "\n".join(files.values())

    expectations = []
    passed = 0
    failed = 0

    # Eval 1 assertions (cnn-scaffold)
    if "cnn-scaffold" in eval_dir:
        a_idx = 0
        # Assertion 1: Cargo.toml includes burn with features wgpu, train, vision
        a = "Cargo.toml includes burn with features wgpu, train, vision"
        cargo = files.get("Cargo.toml", "")
        ok = bool(re.search(r'burn.*=.*\{', cargo)) and (
            all(f in cargo for f in ["wgpu", "train", "vision"])
        )
        expectations.append({"text": a, "passed": ok, "evidence": f"Cargo.toml features check: {'PASS' if ok else 'MISSING one or more of wgpu/train/vision'}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 2: Model struct uses #[derive(Module, Debug)] without B: Backend generic
        a = "Model struct uses #[derive(Module, Debug)] without B: Backend generic"
        model = files.get("model.rs", "")
        has_derive = "#[derive(Module" in model
        no_b_backend = ": Backend" not in model
        ok = has_derive and no_b_backend
        expectations.append({"text": a, "passed": ok, "evidence": f"derive found={has_derive}, B:Backend absent={no_b_backend}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 3: model.init() takes &Device, not &B::Device
        a = "model.init() takes &Device, not &B::Device"
        ok = "&Device" in combined and ": &B::Device" not in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"&Device found={'&Device' in combined}, &B::Device absent={': &B::Device' not in combined}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 4: main.rs has #![recursion_limit = "256"]
        a = 'main.rs has #![recursion_limit = "256"]'
        main = files.get("main.rs", "")
        ok = 'recursion_limit' in main and '256' in main
        expectations.append({"text": a, "passed": ok, "evidence": f"recursion_limit found in main.rs: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 5: Device is created with Device::wgpu(DeviceKind::DefaultDevice)
        a = "Device is created with Device::wgpu(DeviceKind::DefaultDevice)"
        ok = "Device::wgpu" in combined and "DeviceKind" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"Device::wgpu found: {'Device::wgpu' in combined}, DeviceKind: {'DeviceKind' in combined}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 6: No Tensor<B, D> style — uses Tensor<D> without backend generic
        a = "No Tensor<B, D> style — uses Tensor<D> without backend generic"
        # Only check .rs source files to avoid false positives in .md explanations
        rs_content = "\n".join(v for k, v in files.items() if k.endswith(".rs"))
        old_style = bool(re.search(r'Tensor<\w+\s*,\s*\d+>', rs_content))
        ok = not old_style
        expectations.append({"text": a, "passed": ok, "evidence": f"Tensor<Backend, D> pattern absent: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 7: Conv2d/Linear configs use .init(device) pattern
        a = "Conv2d/Linear configs use .init(device) pattern"
        has_init_device = ".init(device)" in combined or ".init(&device)" in combined
        ok = has_init_device
        expectations.append({"text": a, "passed": ok, "evidence": f".init(device) found: {has_init_device}"})
        if ok: passed += 1
        else: failed += 1

    # Eval 2 assertions (migrate-api)
    elif "migrate-api" in eval_dir:
        # Assertion 1: Explains removing B: Backend generic
        a = "Explains removing B: Backend generic from struct and impl"
        ok = "Backend" in combined and ("remove" in combined.lower() or "drop" in combined.lower() or "without" in combined.lower() or "no longer" in combined.lower())
        expectations.append({"text": a, "passed": ok, "evidence": f"Backend removal discussed: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 2: Shows Tensor<B, D> → Tensor<D> conversion
        a = "Shows Tensor<B, D> → Tensor<D> conversion"
        ok = "Tensor<" in combined and "Tensor<3>" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"Tensor<3> (new style) present: {'Tensor<3>' in combined}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 3: Covers Ignored<T> → #[module(skip)]
        a = "Covers Ignored<T> → #[module(skip)] replacement"
        ok = "#[module(skip)]" in combined or "module(skip)" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"#[module(skip)] mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 4: Covers PaddingConfig1d::Explicit(1) → Explicit(1, 1)
        a = "Covers PaddingConfig1d::Explicit(1) → Explicit(1, 1)"
        ok = "Explicit(1" in combined and ("1, 1" in combined or "left" in combined.lower() or "right" in combined.lower())
        expectations.append({"text": a, "passed": ok, "evidence": f"Explicit padding change mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 5: Mentions Gelu → Gelu::new()
        a = "Mentions Gelu → Gelu::new() or Gelu::default()"
        ok = "Gelu" in combined and ("::new()" in combined or "::default()" in combined)
        expectations.append({"text": a, "passed": ok, "evidence": f"Gelu change mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 6: Covers DType::Bool → DType::Bool(_)
        a = "Covers DType::Bool → DType::Bool(_) match arm update"
        ok = "DType::Bool" in combined and ("BoolStore" in combined or "Bool(_" in combined)
        expectations.append({"text": a, "passed": ok, "evidence": f"DType::Bool storage discriminator mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 7: BinFileRecorder warning
        a = "Mentions BinFileRecorder forward-compat or record migration warning"
        ok = "BinFileRecorder" in combined or "NamedMpk" in combined or "forward-compat" in combined or "record" in combined.lower() and "compat" in combined.lower()
        expectations.append({"text": a, "passed": ok, "evidence": f"Record format migration mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

    # Eval 3 assertions (wasm-backend)
    elif "wasm-backend" in eval_dir:
        # Assertion 1: Recommends burn-flex
        a = "Recommends burn-flex backend for WASM/embedded"
        ok = "flex" in combined.lower() and ("burn-flex" in combined or "Flex" in combined)
        expectations.append({"text": a, "passed": ok, "evidence": f"burn-flex recommended: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 2: Mentions burn-ndarray deprecated
        a = "Mentions burn-ndarray is deprecated in 0.21"
        ok = "ndarray" in combined.lower() and ("deprecated" in combined.lower() or "deprecat" in combined.lower())
        expectations.append({"text": a, "passed": ok, "evidence": f"ndarray deprecation mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 3: Cargo features flex, store, default-features = false
        a = "Cargo features: flex, store, default-features = false"
        ok = "flex" in combined and "store" in combined and "default-features" in combined and "false" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"Correct features mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 4: Uses include_bytes!
        a = "Uses include_bytes! to embed model weights at compile time"
        ok = "include_bytes!" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"include_bytes! found: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 5: Uses ModuleRecord::from_bytes
        a = "Uses ModuleRecord::from_bytes for no-filesystem loading"
        ok = "from_bytes" in combined and "ModuleRecord" in combined
        expectations.append({"text": a, "passed": ok, "evidence": f"ModuleRecord::from_bytes found: {ok}"})
        if ok: passed += 1
        else: failed += 1

        # Assertion 6: Mentions Device::flex()
        a = "Mentions Device::flex() for device selection"
        ok = "Device::flex" in combined or "flex()" in combined.lower()
        expectations.append({"text": a, "passed": ok, "evidence": f"Device::flex() mentioned: {ok}"})
        if ok: passed += 1
        else: failed += 1

    total = passed + failed
    pass_rate = passed / total if total > 0 else 0

    # Read timing
    timing_path = f"{eval_dir}/{config}/timing.json"
    timing = json.loads(read_file(timing_path)) if os.path.exists(timing_path) else {}

    return {
        "expectations": expectations,
        "summary": {"passed": passed, "failed": failed, "total": total, "pass_rate": round(pass_rate, 3)},
        "timing": timing
    }

def main():
    evals = ["eval-1-cnn-scaffold", "eval-2-migrate-api", "eval-3-wasm-backend"]
    configs = ["with_skill", "without_skill"]

    all_results = {}
    for ev in evals:
        for cfg in configs:
            key = f"{ev}/{cfg}"
            result = grade_eval(f"{ITER}/{ev}", cfg)
            all_results[key] = result
            # Write grading.json
            out_dir = f"{ITER}/{ev}/{cfg}"
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/grading.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"{key}: {result['summary']['passed']}/{result['summary']['total']} passed ({result['summary']['pass_rate']:.0%})")

    print("\nDone. All grading.json files written.")

if __name__ == "__main__":
    main()
