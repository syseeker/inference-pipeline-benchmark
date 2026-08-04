"""Triton Python-backend model for microsoft/kosmos-2.5 (the ILM tenant).

Why this exists at all: `Kosmos2_5ForConditionalGeneration` is in neither
vLLM's nor SGLang's model registry, so unlike every other LLM/VLM tenant in
the study it cannot be served by a native inference server. Triton's Python
backend is the only path, and this file is the "hand-authored model.py" the
repo layout has always referenced.

It runs inside the Triton container, so it needs torch + transformers +
pillow, which the stock `tritonserver:26.07-py3` image does NOT ship (numpy
and nothing else). Use the derived image from docker/Dockerfile.triton-python.

## The input is a trigger, not the image

perf_analyzer synthesises tensors matching the declared input shape. kosmos-2.5
consumes pix2struct-style `flattened_patches` of (4096, 770); random floats
there are not a document, and generation against them terminates early and
erratically — the tenant would be "running" while producing nothing like the
real workload, and its contention contribution would be meaningless.

So the declared input is a 1-element INT32 trigger, and the real document and
prompts are preprocessed ONCE at load. Each request replays them. Beyond making
the load realistic, this is what the design asks for anyway: hold a CV tenant's
input fixed so the variance you measure is contention rather than content
(SKILL.md failure-recovery, "CV latency varies with image content").

Configured through `parameters` in config.pbtxt, all optional:

    document_path   absolute path to the document image
    prompt_path     .jsonl with one {"text": ...} per line
    output_tokens   max_new_tokens (default 256, matching the ilm_document workload)
"""

from __future__ import annotations

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils  # provided by Triton at runtime


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in (cfg.get("parameters") or {}).items()}

        hf_id = params.get("hf_id", "microsoft/kosmos-2.5")
        doc = params.get("document_path", "")
        prompts_file = params.get("prompt_path", "")
        self._max_new_tokens = int(params.get("output_tokens", "256"))

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 to match the yaml's declared quantization. The card is Blackwell;
        # fp16 would also work but would not be what the config says is running.
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(hf_id)
        self._model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            hf_id, dtype=self._dtype,
        ).to(self._device).eval()

        prompts = ["<ocr>"]
        if prompts_file and os.path.exists(prompts_file):
            loaded = []
            with open(prompts_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line).get("text")
                    except json.JSONDecodeError:
                        continue
                    if t:
                        loaded.append(t)
            if loaded:
                prompts = loaded

        if doc and os.path.exists(doc):
            image = Image.open(doc).convert("RGB")
        else:
            # A blank page still exercises the full patch grid, so a missing
            # document degrades the realism of the workload without taking the
            # tenant out of the study. The pre-flight in coloc.py is what stops
            # a run reaching here with its payload missing.
            image = Image.new("RGB", (1024, 1024), "white")

        # Preprocess every prompt once. Requests round-robin over them, so the
        # tenant's load is deterministic across repeats and across the solo
        # baseline / contention pair — which is what makes their ratio mean
        # something.
        self._batches = []
        for text in prompts:
            enc = self._processor(images=image, text=text, return_tensors="pt")
            enc.pop("width", None)
            enc.pop("height", None)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            if "flattened_patches" in enc:
                enc["flattened_patches"] = enc["flattened_patches"].to(self._dtype)
            self._batches.append(enc)
        self._next = 0

    def execute(self, requests):
        torch = self._torch
        responses = []
        for _ in requests:
            enc = self._batches[self._next % len(self._batches)]
            self._next += 1
            with torch.inference_mode():
                out = self._model.generate(**enc, max_new_tokens=self._max_new_tokens)
            text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
            tensor = pb_utils.Tensor(
                "TEXT", np.array([text.encode("utf-8")], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses

    def finalize(self):
        self._model = None
        self._batches = None
