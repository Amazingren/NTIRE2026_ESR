# Team 22 — SPANV2_ESR Integration Guide

## Files

```
team22_submission/
├── README.md
├── models/
│   └── team22_SPANV2_ESR.py        # Model architecture
├── model_zoo/
│   └── team22_spanv2_c2.pth        # Checkpoint (~553 KB)
├── span_attention_op/              # Custom CUDA operator
│   ├── build_span_attn.sh          # Build/install script
│   ├── setup.py
│   └── csrc/
│       ├── span_attention.cpp
│       ├── span_attention_kernel.cuh
│       ├── span_attention_kernel_general.cu
│       ├── span_attention_kernel_opt2.cu
│       ├── span_attention_kernel_optimized.cu
│       └── span_attention_kernel_templated.cu
└── test_demo_team22.py             # Standalone integration demo
```

---

## Step 1 — Copy files into the evaluation framework

```
ntire_official/
├── models/
│   └── team22_SPANV2_ESR.py        ← copy here
├── model_zoo/
│   └── team22_spanv2_c2.pth        ← copy here
├── span_attention_op/              ← copy entire directory here
└── test_demo_team22.py             ← copy here (or see Step 3)
```

---

## Step 2 — Build the span_attention CUDA operator

The model uses a custom CUDA attention kernel for inference speed.
Build and install it into site-packages with:

```bash
PYTHON=/path/to/python bash span_attention_op/build_span_attn.sh
```

**Why `--no-build-isolation` is used internally:**
All PyTorch CUDA extensions require `torch` at build time.
`pip install .` runs in an isolated sandbox that has no `torch`,
so the build would silently fail. The script passes
`--no-build-isolation` automatically — no extra flags needed.

**Verify installation:**
```python
import torch
import span_attention
print(span_attention.__file__)  # should print path inside site-packages
```

> Note: `import torch` must precede `import span_attention` in any script
> because `libc10.so` / `libtorch.so` must be loaded first.
> In `test_demo.py`, `import torch` is already at the top, so this is
> handled automatically.

---

## Step 3 — Integrate into test_demo.py

Add the following block into the `select_model()` function in `test_demo.py`:

```python
elif model_id == 22:
    # Team 22: SPANV2_ESR
    import sys, subprocess
    span_attn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'span_attention_op')
    span_attn_dir = os.path.normpath(span_attn_dir)
    if span_attn_dir not in sys.path:
        sys.path.insert(0, span_attn_dir)
    # Auto-build span_attention if not installed
    try:
        import span_attention
    except ImportError:
        print("[team22] span_attention not found, building from source ...")
        build_result = subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=span_attn_dir,
            capture_output=True, text=True
        )
        if build_result.returncode != 0:
            raise RuntimeError(
                f"[team22] Failed to build span_attention:\n"
                f"{build_result.stdout}\n{build_result.stderr}"
            )
        print("[team22] span_attention built successfully.")
        import span_attention  # noqa: F811
    from models.team22_SPANV2_ESR import SPANV2_ESR
    name, data_range = f"{model_id:02}_SPANV2_ESR_C2", 1.0
    model_path = os.path.join('model_zoo', 'team22_spanv2_c2.pth')
    model = SPANV2_ESR(3, 3, feature_channels=32, upscale=4, bias=False, use_span_attn=True)
    state = torch.load(model_path, map_location='cpu')
    for key in ['model', 'state_dict', 'params', 'params_ema']:
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
```

### Params / FLOPs measurement note

`span_attention` is a custom CUDA op that **fvcore cannot trace**.
For params/FLOPs/activations measurement, instantiate the model with
`use_span_attn=False` (pure PyTorch fallback, same architecture):

```python
if args.model_id == 22:
    from models.team22_SPANV2_ESR import SPANV2_ESR as _SPANV2_ESR
    _model_cpu = _SPANV2_ESR(3, 3, feature_channels=32, upscale=4, bias=False, use_span_attn=False)
    _state = torch.load(os.path.join('model_zoo', 'team22_spanv2_c2.pth'), map_location='cpu')
    for _key in ['model', 'state_dict', 'params', 'params_ema']:
        if isinstance(_state, dict) and _key in _state:
            _state = _state[_key]
            break
    _model_cpu.load_state_dict(_state, strict=True)
    _model_cpu.eval()
    _model_for_stats = _model_cpu.to(device)
else:
    _model_for_stats = model

# then pass _model_for_stats to get_model_activation() and FlopCountAnalysis()
```

A complete working demo is provided in `test_demo_team22.py`.

---

## Step 4 — Run

```bash
# valid set only
python test_demo_team22.py --model_id 22 --data_dir /path/to/datasets/

# valid + test set
python test_demo_team22.py --model_id 22 --data_dir /path/to/datasets/ --include_test
```

---

## Results (A100 80GB)

| Metric            | Value       |
|-------------------|-------------|
| Val PSNR          | 26.92 dB    |
| Val Runtime       | 7.12 ms     |
| Test Runtime      | 5.67 ms     |
| Params            | 0.1391 M    |
| FLOPs             | 9.11 G      |
| Activations       | 51.38 M     |
| #Conv2d           | 22          |
| Max Memory (test) | 685.67 MB   |

---

## Design notes

### `use_span_attn` flag

`SPANV2_ESR` and `SPABV2` accept `use_span_attn: bool`:

| Value  | Behavior | When to use |
|--------|----------|-------------|
| `True` | CUDA kernel (`span_attention.span_attention()`) | Speed benchmark |
| `False`| Pure PyTorch `(x + f3) * guidance_map_conv(f3)` | Params / FLOPs / Activations |

The two paths are **numerically different** (optimized kernel fuses ops)
but the model weights are identical — no separate checkpoints needed.

### Lazy import

`import span_attention` is placed **inside** `SPABV2.forward()` under the
`use_span_attn=True` branch, not at module top-level.
This means importing `team22_SPANV2_ESR` never fails even when the CUDA
operator is not installed, enabling safe params/FLOPs measurement without
the operator present.

### Auto-build fallback

If `span_attention` is not installed, `select_model()` automatically runs
`setup.py build_ext --inplace` (reuses compiled `.o` files from `build/`,
typically completes in a few seconds) and then imports the freshly built `.so`.
