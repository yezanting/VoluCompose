# Evaluation logic and usage

The evaluation framework is adapted from [Hulu-Med](https://github.com/ZJUI-AI4H/Hulu-Med). The entrypoint connects dataset loading, model inference, shard aggregation and task-specific metric calculation.

## Execution path

1. `MedUniEval/eval.sh` or `eval_M3D_chunked.sh` reads environment variables, resolves the repository root and invokes `MedUniEval/eval.py`.
2. `eval.py` loads the selected model wrapper and compression parameters from `config/config.json`, then dispatches the requested datasets through `benchmarks.py`.
3. Each dataset loader reads annotations, loads ordered slice images and constructs the task prompt.
4. The model wrapper preprocesses the images and text, invokes generation, and returns responses together with visual-token retention metadata.
5. The dataset handler writes predictions, merges completed shards and invokes the appropriate metric logic. AMOS report scoring runs separately.

## Data loading

| Dataset | Annotation path relative to its dataset directory | Image loading |
| --- | --- | --- |
| M3D | `M3D_test.json` | Reads `video[0]`, sorts slice filenames, and samples evenly spaced indices if the count exceeds `MAX_IMAGE_NUM`. |
| AMOS | `amos_val_mrg_sft.json` | Opens supported slice-image files in the referenced directory without applying the M3D sampling limit. |
| 3D-RAD | `Rad3D_test.json` | Uses the inherited video-directory loader with its internal frame limit. `MAX_IMAGE_NUM` does not override that limit. |

Annotation paths must point to actual image directories. Filenames must sort in anatomical slice order, and the directories should contain only the intended slices. The launchers resolve relative paths from the repository root.

## Compression and generation

`COMPRESSION=volucompose` selects the VoluCompose adapters; `COMPRESSION=none` selects the uncompressed model. `visionzip`, `holitom` and `unicomp` select the included comparison adapters.

VoluCompose first groups contiguous recurrent slices into ordered support units. It distributes the token budget across units, selects distinctive anchors using Lorentz distance, refines them with neighboring features, and preserves source order in the compact sequence. `retain_ratio` sets a budget relative to the original visual-token count. With `auto_compress=true`, normalized attention entropy determines the budget instead. Unit capacities and integer allocation constrain the final sequence length.

`compression_rate` and `compose_rates.compose_rate` record retained visual tokens divided by original visual tokens. These fields describe retention rather than the removed fraction.

Keep the model checkpoint, processor, input sampling, prompt, decoding settings and metric implementation consistent when comparing configurations. Qwen/Hulu wrappers use greedy decoding when temperature is zero. The MedGemma wrapper sets top-k and top-p to `None` in its generation call. `TEST_TIMES` is an inherited argument and does not implement repeated runs in this entrypoint.

## Shards and output files

The M3D launcher executes shards sequentially. Set `CHUNKS` to a positive integer; each shard uses the configured visible GPU set. Chunk zero clears previous shard files, and the handler merges predictions when all shard files exist. Use a separate output directory for each method and configuration.

M3D and 3D-RAD write `results.json`, `metrics.json` and `wrong_answers.json`; chunked runs also write `results_<index>.json`. The aggregate `total_results.json` write is disabled in `eval.py`, so read the files in each dataset's output directory.

Open-ended VQA uses exact match, F1, precision/recall, BLEU, ROUGE and METEOR. Closed-answer questions use answer extraction and category aggregation. The 3D VQA metric path does not use a GPT judge, so `USE_LLM_JUDGE=False` is sufficient. Install NLTK WordNet resources for METEOR.

## AMOS report scoring

AMOS generation writes `results.json` and prediction/reference CSV files. Its initial `metrics.json` points to the separate report scorer.

Install the optional dependencies from the repository root:

```bash
python -m pip install -r requirements-report.txt
python -m nltk.downloader wordnet omw-1.4
```

Place the RaTE checkpoints at `workspace/model/RaTE-NER-Deberta` and `workspace/model/BioLORD-2023-C`, matching the scorer's local paths. Then run:

```bash
cd MedUniEval
python utils/Metrics_Compute/cal_report_metrics.py \
  --results_path ../workspace/eval/hulu-amos-volucompose/Amos/results.json \
  --output_metrics_path ../workspace/eval/hulu-amos-volucompose/Amos/report_metrics.json
```

The scorer computes BLEU, ROUGE, METEOR and RaTE. It removes report-section prefixes from predictions, truncates predictions for ROUGE, and skips empty predictions. Lexical metrics use the total input count as their denominator; RaTE averages over nonempty predictions. The script does not compute GREEN or BERTScore.

## Local tests

From the repository root:

```bash
python -m pip install build wheel
python -m unittest discover -s tests -v
```

The test suite exercises entropy-based capacity, ordered grouping, token allocation, wrapper retention fields, launcher argument propagation, lazy imports and wheel contents. It uses synthetic features and simulated generation calls. Mathematical tests require PyTorch and Geoopt; package tests require `build`. Model inference and dataset evaluation require the actual checkpoints, data and GPU environment described in the README.
