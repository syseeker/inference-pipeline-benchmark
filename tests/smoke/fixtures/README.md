# Smoke-test fixtures

Three end-to-end scenarios for the VLM-to-action pipeline. Each one is
self-contained: a synthetic image, the user instruction, a short context
history, and the **golden** action sequence the pipeline is expected to
produce.

## On-disk shape

```
tests/smoke/fixtures/<name>/
├── input.json       # FixtureInput  — image path, instruction, context history
├── image.jpg        # visual the request points at
└── expected.json    # FixtureExpected — golden ActionSequence + ValidationReport
```

Pydantic models in [schema.py](schema.py); loader in [loader.py](loader.py).

## The three fixtures

| Name | Scenario | Expected commands |
| --- | --- | --- |
| `01_click_start_button` | Game launcher with a Start button. Empty history. | `move`, `click(left)` |
| `02_dismiss_update_popup` | A "Software Update" modal interrupts a recording session. | `keypress(escape)`, `say` confirming |
| `03_pause_then_resume_video` | Video player; user wants a 5-second pause. | `keypress(space)`, `wait(5000)`, `keypress(space)`, `say` |

Coordinates and pixel offsets in the gold sequences match the synthetic
images produced by [_generate_images.py](_generate_images.py). Re-run that
script if you ever change image dimensions or button positions; the goal
is that the gold matches the literal pixels in the JPEG.

## How they're consumed

- **Offline parametrised test** (default CI lane):
  `pytest -m smoke tests/smoke/test_fixtures.py`
  Uses a `_GoldReasoner` that returns the gold JSON, then asserts the
  decoder + validator accept it. Proves the schema is sane and the gold
  is internally consistent.

- **Live NIM run** (opt-in):
  `NIM_API_KEY=... python -m examples.run_fixture 01_click_start_button`
  Sends the real fixture to a NIM-hosted Qwen-VL endpoint and prints
  actual vs. expected. Useful for eyeballing model behaviour on a known
  visual.

- **Benchmark feed** (future):
  the runner can iterate over `load_all()` to produce a deterministic
  workload across frameworks/GPUs.

## Adding a new fixture

1. `mkdir tests/smoke/fixtures/04_<short_name>/`.
2. Add a synthetic-image renderer to `_generate_images.py` and run it.
3. Write `input.json` and `expected.json` matching the schemas.
4. Done — the parametrised smoke test discovers it via `list_fixtures()`.
