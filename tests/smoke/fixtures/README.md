# Smoke-test fixtures

Three real game-screen scenarios for the VLM-to-action pipeline. Each
fixture is one (visual + short context history + high-level instruction)
→ (low-level action sequence) example, captured straight from a running
game so the visuals look like what Razer's customers will actually see.

## On-disk shape

```
tests/smoke/fixtures/<name>/
├── request.json     # FixtureRequest — what the pipeline receives (instruction, history, image ref)
├── screen.<ext>     # the visual the request points at (png or jpeg)
└── expected.json    # FixtureExpected — gold ActionSequence + ValidationReport
```

Pydantic models in [schema.py](schema.py); loader in [loader.py](loader.py).
`FixtureRequest` is the on-disk mirror of the production `PipelineRequest`.

## The three fixtures

| Name | Game | Expected commands |
| --- | --- | --- |
| `01_clash_of_clans_start_attack` | Clash of Clans (mobile, 2001×923) | `move(95,825)` → `click(left)` → `say(...)` |
| `02_catan_open_menu` | Catan online (1024×494) | `move(50,437)` → `click(left)` → `say(...)` |
| `03_fps_engage_and_reload` | Sci-fi FPS (1796×975) | `click(left)` × 2 → `keypress(r)` → `say(...)` |

`MOVE.dx/dy` are absolute pixel coordinates against the source image
referenced in each `request.json`. Each `expected.json` documents the
tolerant hitbox a grader should accept.

## How they're consumed

- **Offline parametrised test** (default CI lane):
  `pytest -m smoke tests/smoke/test_fixtures.py`
  Uses a `_GoldReasoner` that returns the gold JSON, then asserts the
  decoder + validator accept it. Proves each fixture is internally
  consistent.

- **Live NIM run** (opt-in):
  `NIM_API_KEY=... python -m examples.run_fixture 01_clash_of_clans_start_attack --backend nim`
  Sends the real fixture to a NIM-hosted Qwen-VL endpoint and prints
  actual vs. expected. Useful for eyeballing model behaviour on a known
  visual.

- **Benchmark feed** (future):
  the runner can iterate over `load_all()` to produce a deterministic
  workload across frameworks/GPUs.

## Adding a new fixture

1. `mkdir tests/smoke/fixtures/04_<short_name>/`.
2. Drop the screenshot in as `screen.png` (or `.jpeg`).
3. Write `request.json` (instruction + history + `image_path`) and
   `expected.json` (gold `ActionSequence` + `ValidationReport`).
4. Done — the parametrised smoke test discovers it via
   `list_fixtures()`.
