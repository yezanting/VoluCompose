# Evaluation protocol and provenance

This release starts from the author-supplied `med3d2-final` snapshot. A ZIP/extracted folder does not contain the Gitee branch history; no Gitee commit SHA is asserted. The launchers use the supplied generation settings, with environment overrides and corrected method selection/retention metadata.

## Hulu-Med comparison

Reference: [ZJUI-AI4H/Hulu-Med](https://github.com/ZJUI-AI4H/Hulu-Med), commit [`6d4d47aa014342b1beda70695ab7c799edc7ade6`](https://github.com/ZJUI-AI4H/Hulu-Med/tree/6d4d47aa014342b1beda70695ab7c799edc7ade6), retrieved on 2026-10-12. This is a recorded reference snapshot, not a claim about which historical commit the author originally used.

| Component | Observed relationship to the recorded Hulu-Med snapshot |
| --- | --- |
| `utils/question_formats.py` | Identical in the supplied final snapshot. |
| `utils/eval_3d.py` | Identical metric function definitions; the supplied file additionally raises Python's recursion limit. Closed-answer extraction, text normalization, metric formulas and category aggregation are retained. |
| M3D loader | Uses `<datasets_path>/M3D_test.json`; adds evenly spaced, ordered slice-file sampling with `MAX_IMAGE_NUM`. The recorded upstream loader passes the video directory directly. This input difference must be controlled in comparisons. |
| 3D-RAD loader | Uses `<datasets_path>/Rad3D_test.json` rather than a direct annotation-file path. Retains `load_video(..., fps=1, max_frames=1800)` and the upstream prompt construction. |
| AMOS loader | Uses `<datasets_path>/amos_val_mrg_sft.json` instead of the recorded upstream's hard-coded annotation path. Returns generation plus retention metadata and initializes its report output accumulator. |
| Base dataset | Stores both responses and retention metadata and retains the supplied shard aggregation behavior. Compression wrappers return `(responses, retention_records)` rather than only responses. |
| Report metric script | Wraps the inherited report metric calculations in an explicit results/output CLI. Uses local RaTE model paths. This release retains those calculations and preprocessing. |

The evaluation is **adapted from Hulu-Med**, with the differences above. It is not byte-for-byte identical to the whole upstream framework. No dataset-loader or metric formula changes are introduced during this publication pass.

## Release fixes

- The four MedPruner++ model wrappers now recognize `medpruner++` when recording measured retention; the supplied snapshot checked obsolete `our/auto` labels and otherwise wrote `1.0`.
- Both shell launchers honor environment overrides, locate the repository independently of the caller's working directory, pass the correct judge-model variable, and default to `COMPRESSION=medpruner++`.
- M3D shards remain sequential. `CHUNKS` is configurable and must be a positive integer. Shards use the same visible GPU set; this is not per-GPU data parallelism.
- The supplied package configuration included only `medcompose/__init__.py` in a wheel. Package discovery now includes the nested backend modules. Package exports load backends on demand without changing their model implementations.
- Transformers is pinned to `4.57.3`: the supplied old `4.51.2` list lacks Qwen3-VL APIs, and `4.57.1` exposes a different `check_model_inputs` decorator signature. No compatibility patch to the compression mathematics is applied.

## Matched-run settings

The reference and compressed run must share:

1. Model checkpoint and processor revision, dtype, attention implementation and GPU/software environment.
2. Dataset split, annotation JSON, valid case set, filename order and slice-sampling limits.
3. Prompt template, seed, generation length, temperature, repetition penalty and decoding mode.
4. Metric implementation, normalization, aggregation and NLTK resource availability.

The launchers default to seed 42, temperature 0, top-p 0.95, repetition penalty 1, and Transformers inference. Qwen/Hulu wrappers use greedy decoding when temperature is zero. The MedGemma wrapper retains its supplied generation call, which sets top-k and top-p to `None`; do not infer that every CLI decoding parameter affects every adapter equally. `TEST_TIMES` is an inherited argument and does not implement repeated runs in this entrypoint.

### Input sampling

- M3D reads `video[0]` and sorts all files in that directory. If there are more than `MAX_IMAGE_NUM`, it chooses evenly spaced indices using `numpy.linspace`. Keep those directories free of unrelated files.
- AMOS opens all supported slice-image filenames in its referenced directory. It does not apply the M3D sampling limit.
- 3D-RAD uses the inherited video-directory loader, `fps=1`, and a hard-coded cap of 1,800 frames. `MAX_IMAGE_NUM` does not override that cap.
- All three loaders use the annotation's image path; putting JSON files under `data/Eval` alone does not repair stale image paths.

### Token capacity

`retain_ratio` is relative to the native visual-token count before slice grouping. The final count is bounded by support-unit capacities and integer allocation. `auto_compress=true` uses mean normalized attention entropy followed by `clamp(H_mean^3 / 2, 0, 0.3)`; the realized rate is input-dependent. Report the mean and distribution of measured `compose_rates.compose_rate` values over the evaluated case set.

The code exports `compression_rate` even though its meaning is **retention fraction**. Do not present it as the removed fraction. Token counts alone do not measure end-to-end latency, encoder savings or observed memory savings.

### Shards and outputs

The inherited aggregator lets chunk 0 remove previous shard files, then combines results when all shard files exist. Run the M3D launcher sequentially as provided and use a separate output directory for each method/configuration. Changing to concurrent workers requires redesigning and testing the cleanup/merge behavior first.

M3D/3D-RAD save `results.json`, `metrics.json` and `wrong_answers.json`; chunked runs also save `results_<index>.json`. The CLI's aggregate `total_results.json` write is commented out in the supplied `eval.py`; use each dataset's own output files.

For open-ended M3D/3D-RAD questions, metrics include exact match, F1, precision/recall, BLEU, ROUGE and METEOR. Overall metrics are category averages. Categories present only in the closed-question subset can contribute empty/zero-valued open metrics under the retained implementation. Use the same category set when comparing runs; do not substitute sample-weighted aggregation without labeling it as a different protocol.

The 3D VQA metric path does not use a GPT judge; `USE_LLM_JUDGE=False` suffices for these local metrics. METEOR requires NLTK WordNet resources. The inherited code catches METEOR exceptions and returns zero, so install those resources before interpreting the score.

## AMOS report scoring

AMOS first exports reports to `results.json` and `predictions_<chunk>.csv` / `ground_truth_<chunk>.csv`. Its initial `metrics.json` directs the user to the separate report scorer; it does not contain final report metrics.

Install the optional dependencies from the repository root:

```bash
python -m pip install -r requirements-report.txt
python -m nltk.downloader wordnet omw-1.4
```

Place the RaTE checkpoints at `workspace/model/RaTE-NER-Deberta` and `workspace/model/BioLORD-2023-C`, matching the supplied scorer's local paths. Then run:

```bash
cd MedUniEval
python utils/Metrics_Compute/cal_report_metrics.py \
  --results_path ../workspace/eval/hulu-amos-medprunerpp/Amos/results.json \
  --output_metrics_path ../workspace/eval/hulu-amos-medprunerpp/Amos/report_metrics.json
```

The inherited scorer computes BLEU-1/2/3/4, ROUGE-1/2/L, METEOR and RaTE. It removes `Findings:` / `Impression:` from predictions, truncates predictions to 2,048 characters for ROUGE, and skips empty predictions while retaining the total input-sample count as the lexical-metric denominator. RaTE averages over the nonempty prediction set. These details are retained for comparability. Despite its printed “Green score” message and its `BERTScorer` import, this script does not compute GREEN or BERTScore. Full report scoring requires the actual checkpoint files and was not reproduced in the CPU release checks.

## Verification scope

The publication checks cover:

- CPU mathematical/output-contract checks for entropy capacity, ordered grouping and allocated token/mask counts.
- Regression tests for measured retention on the four wrappers, environment propagation, sequential chunk arguments and lazy package imports.
- Wheel contents, including all nested model modules.
- Imports of all four backbones for MedPruner++, VisionZip, HoliTom and UniComp under Transformers 4.57.3 and PyTorch 2.6.0 on CPU.
- Python/Bash syntax, local documentation/figure links and preservation of the supplied metric and preprocessing source.

These checks do not establish numerical benchmark reproduction, CUDA/FlashAttention compatibility on a particular server, clinical validity, or an observed end-to-end speedup. The dataset/model-dependent evaluation remains a separate reproducibility step.
