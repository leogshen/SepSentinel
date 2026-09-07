#!/usr/bin/env python
"""Feasibility smoke test for TimesFM on this machine.

Answers three practical questions before any experiment is designed:
  1. does a checkpoint load and run on this GPU (Turing, 11 GB, no bf16)?
  2. what does the API actually expose -- is there any classification path?
  3. what is the context/patch geometry, given our sequences are ~24-72 hourly
     steps and TimesFM patches at 32?

Not a model evaluation. It forecasts a synthetic series and reports shapes.
"""

import sys
import time

import numpy as np
import torch


def main():
    import timesfm
    print("timesfm module:", timesfm.__file__)
    print("public API:", [n for n in dir(timesfm) if not n.startswith("_")])
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "| bf16 supported:", torch.cuda.is_bf16_supported()
          if torch.cuda.is_available() else "n/a")

    t0 = time.time()
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
    except Exception as exc:
        print("CHECKPOINT LOAD FAILED: %s: %s" % (type(exc).__name__, exc))
        return 1
    print("checkpoint loaded in %.1fs" % (time.time() - t0))

    try:
        model.compile(timesfm.ForecastConfig(
            max_context=512, max_horizon=24, normalize_inputs=True))
    except Exception as exc:
        print("compile note: %s: %s" % (type(exc).__name__, exc))

    # Our regime: a single univariate channel, ~48 hourly steps of context.
    ctx = [np.sin(np.arange(48) / 6.0).astype(np.float32)]
    try:
        point, quantiles = model.forecast(horizon=12, inputs=ctx)
        print("forecast OK: point %s, quantiles %s"
              % (np.asarray(point).shape, np.asarray(quantiles).shape))
    except Exception as exc:
        print("FORECAST FAILED: %s: %s" % (type(exc).__name__, exc))
        return 1

    has_cls = [n for n in dir(model)
               if "class" in n.lower() or "embed" in n.lower()]
    print("classification/embedding entry points on the model object:",
          has_cls or "NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
