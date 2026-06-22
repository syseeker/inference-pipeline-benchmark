# Scenarios — the common input format

Every benchmark in this harness consumes the same on-disk shape. One
scenario is **one (frame + optional instruction + optional context) → one
ground truth**. We use the same shape across the VLM backends
(vLLM / SGLang / TRT-LLM serving Qwen3-VL / Gemma 4 / Nemotron-Omni) and
the policy backend (NitroGen), so a single sweep can compare them
apples-to-apples on the same input.

This doc explains:

1. The on-disk shape and what each file is for.
2. How scenarios from the `nvidia/NitroGen` dataset are converted into
   that shape (and why we convert at all).
3. How to add your own dataset as a scenario source — no fork needed.

---

## 1. The on-disk shape

```
tests/smoke/scenarios_nitrogen/<N>_<game>_<chunk>/
├── screen.png         (always — the input image)
├── request.json       (always — the input envelope)
├── gold_action.json   (optional — policy ground truth: faithful gamepad vector)
└── expected.json      (optional — VLM ground truth: schema-validated ActionSequence)
```

`request.json` is the input every model receives — see
[ScenarioRequest](../tests/smoke/scenarios/schema.py) for the schema.

```json
{
  "name": "01_other_-N72WiKxmfg_chunk_0000",
  "description": "NitroGen dataset frame from 'other' (-N72WiKxmfg_chunk_0000).",
  "image_path": "screen.png",
  "instruction": "",
  "context_history": [],
  "deadline_ms": 1500,
  "game_id": "other"
}
```

For a NitroGen-derived scenario, `instruction` is empty and
`context_history` is `[]` — NitroGen is conditioned on `(frame, game_id)`
only. For a human-authored VLM scenario, `instruction` carries the
high-level command the model should follow ("attack the closest enemy",
"open inventory") and `context_history` carries the last few turns.

### Two ground-truth shapes, one runtime

A scenario can carry **either or both** ground-truth files. The runner
dispatches on file presence:

| File present | Grader used | What it measures |
|---|---|---|
| `expected.json` only | VLM grader | model emits an ActionSequence — does it match `expected.actions`? Pass/fail goes into the `grammar_valid` / `exec_accept` columns of `summary.md`. |
| `gold_action.json` only | Policy grader | model emits a gamepad vector — what's `action_mse`, `button_agreement_rate`, `joystick_mae` against `gold_action.{buttons, j_left, j_right}`? |
| **Both** | Both graders | same frame + instruction graded against both ground truths. Lets you ask "on this gameplay moment, which family is closer to the intended action?" |

`bench scenarios list --json` reports `has_expected` / `has_gold_action`
per scenario so an agent can pick the right grader without opening files.

The fields inside each ground-truth file:

```json
// gold_action.json — emitted by prepare-nitrogen-dataset
{
  "game_id": "other",
  "buttons": { "south": 0.0, "north": 0.0, "dpad_up": 0.0, ... 17 keys },
  "j_left":  [-0.50, 0.05],   // [x, y] in [-1, 1]
  "j_right": [0.0, 0.0],
  "provenance": {
    "chunk":       "SHARD_0000/-N72WiKxmfg/-N72WiKxmfg_chunk_0000",
    "url":         "https://www.youtube.com/watch?v=-N72WiKxmfg",
    "frame_index": 600,
    "game":        "other"
  }
}
```

```json
// expected.json — schema-validated ActionSequence (see vlm_pipeline/schemas.py)
{
  "actions": {
    "commands": [
      { "type": "move",     "args": { "dx": 511, "dy": 47 }, "confidence": 1.0 },
      { "type": "keypress", "args": { "key": "right_trigger" }, "confidence": 1.0 }
    ],
    "rationale": "..."
  },
  "validation": { "schema_valid": true, "safe": true, "rejected_command_indices": [], "notes": [] },
  "notes": null
}
```

---

## 2. NitroGen dataset → scenario, field by field

`nvidia/NitroGen` ships **action annotations only — no pixels.** Each chunk
on the HF dataset is:

```
nvidia/NitroGen (HF dataset, .tar.gz per shard)
└── actions/SHARD_####/<video_id>/<video_id>_chunk_####/
    ├── actions_processed.parquet   17 button cols + j_left/j_right per frame
    ├── actions_raw.parquet         pre-processing input
    └── metadata.json               url, game, resolution, start_frame, end_frame, bboxes
```

`scripts/build_nitrogen_scenarios.py` turns one chunk into one scenario:

```
                       chunk dir
   ┌─────────────────────┬───────────────────────────┬──────────────────────┐
   │ metadata.json       │ actions_processed.parquet │   <video URL>        │
   │  - original_video.url│  - 1200 rows (one per   │   via yt-dlp (or     │
   │  - start_frame/end   │    frame)                │   --synthetic-frames)│
   │  - game              │  - 17 button cols, j_*   │                      │
   │  - bbox_game_area    │                          │                      │
   └─────────┬───────────┴─────────┬─────────────────┴──────────┬───────────┘
             │                     │                            │
             │                     │ row at midpoint frame      │ decode + crop
             │                     │                            │ to bbox + resize
             ▼                     ▼                            ▼
       request.json           gold_action.json                screen.png
   ┌──────────────────┐  ┌──────────────────────┐   (256×256 RGB)
   │ name             │  │ buttons[17]          │
   │ description      │  │ j_left  = [x, y]     │
   │ image_path       │  │ j_right = [x, y]     │
   │ game_id ←────────┼──┤ game_id ←────────────┼── metadata.json "game"
   │ instruction = "" │  │ provenance = {       │
   │ context = []     │  │   chunk, url,        │
   │ deadline_ms      │  │   frame_index, game  │
   └──────────────────┘  └──────────────────────┘
```

Frame choice: midpoint of `[start_frame, end_frame]`. One representative
frame per chunk; the parquet's other ~1199 rows aren't used today (they
become useful when you want temporal sequences — see [docs/nitrogen.md](nitrogen.md)).

`expected.json` is **not** auto-generated for NitroGen scenarios. The
dataset doesn't carry a human-authored VLM-style intended command list,
and synthesising one from the gamepad vector would inflate the validator
columns by construction. If you want to grade VLMs on these frames too,
write `expected.json` by hand (the workflow that authors instruction +
expected action belongs to the user).

### Offline / synthetic mode

When the box can't reach YouTube/Twitch (cloud IPs are routinely bot-blocked
by YouTube, and 2-year-old Twitch VODs rot), pass `--synthetic-frames`.
We then write a deterministic noise tile per scenario instead of a decoded
frame. The **gamepad ground truth is still real** (from the parquet) —
only the pixel input is a placeholder. Latency, throughput, and the
fp8-vs-bf16 numerical delta stay measurable; policy quality (does the
model emit the right action for *this* gameplay moment) does not.

---

## 3. Why we convert, instead of reading the dataset directly

Three reasons. The first is the load-bearing one.

**3.1. The same scenarios benchmark VLMs and policies side by side.**
This harness runs vLLM + SGLang + TRT-LLM (serving Qwen3-VL / Gemma 4 /
Nemotron-Omni) on the *same* `Pipeline.run(request) → ActionSequence` path
as it runs NitroGen. They share `LoadedScenario.pipeline_request()` and
the same `summary.md` columns. If the loader read NitroGen format
directly, **only NitroGen could run** — you'd lose the cross-family
comparison that's the point of the harness.

**3.2. The dataset has no pixels.** `metadata.json:original_video.url`
points at a YouTube/Twitch VOD. Reading "directly" still means yt-dlp +
ffmpeg decode + bbox crop + resize. You don't want that in the
latency-measurement path — it's slow, non-deterministic, and depends on
whether the box has a JS runtime for yt-dlp. Pre-extracting `screen.png`
once makes every subsequent run fast and reproducible.

**3.3. Frame choice and cropping are benchmark-design decisions.** Each
chunk is 1200 frames; which one represents the chunk? `bbox_game_area`
isolates the game pixels from the controller overlay; which crop? These
belong in the extractor (so they're explicit and reviewable), not in the
runtime.

---

## 4. Adding your own dataset as a scenario source

Don't fork. Register a Python entry-point in your own package and
`bench scenarios build --source <your-name>` discovers it automatically.

```toml
# your-package's pyproject.toml
[project.entry-points."pipeline_bench.scenario_sources"]
my-gameplay = "my_pkg.scenarios:build"
```

Your `build` function follows this contract:

```python
def build(*, n: int, out: Path, **kwargs) -> int:
    """Write `n` scenarios under `out/`. Each scenario directory has at
    minimum:

        <out>/<name>/
            screen.png        (or another image referenced by request.image_path)
            request.json      (ScenarioRequest — see tests/smoke/scenarios/schema.py)

    Plus either or both ground-truth files:
            gold_action.json  (your policy ground truth in the format
                               gold_action_schema.md documents — buttons,
                               j_left, j_right, game_id, provenance)
            expected.json     (a ScenarioExpected with the intended
                               ActionSequence + validation verdict)

    Return the count successfully written (the harness may ask for more
    than you can produce — partial writes are fine).
    """
```

Then:

```bash
bench scenarios sources --json   # confirms my-gameplay is discovered
bench scenarios build --source my-gameplay --n 10 --out path/to/dir --json
bench smoke --gpu rtx_pro6000 --backend nitrogen-eager --model nitrogen-500m-bf16 \
            --scenarios-dir path/to/dir --json
```

The runner doesn't care which source produced the scenarios — once the
files are on disk in the shape above, every backend can consume them.
That's the value of the standardization: **the dataset-shape problem is
yours; the benchmark contract is shared.**

### When to write a custom source instead of using `nitrogen`

- You have **your own gameplay clips** + actions in a different format.
- You're building a **VLM benchmark** (write `expected.json`, skip `gold_action.json`).
- You want **streaming / lazy frame extraction** instead of the upfront
  one-time decode (your source can write a placeholder `screen.png` and
  store the video pointer in `request.json` provenance for runtime
  fetching — at the cost of slower, less reproducible runs).

---

## Pinned references

- The extractor: [scripts/build_nitrogen_scenarios.py](../scripts/build_nitrogen_scenarios.py)
- Scenario schema: [tests/smoke/scenarios/schema.py](../tests/smoke/scenarios/schema.py)
- Loader (presence-dispatch on ground-truth files): [tests/smoke/scenarios/loader.py](../tests/smoke/scenarios/loader.py)
- NitroGen model context (what it is, how it relates to Cosmos 3 / GR00T): [docs/nitrogen.md](nitrogen.md)
- Metric definitions including accuracy-vs-gold: [docs/metrics.md](metrics.md)
- The five agent skills: [skills/](../skills/)
