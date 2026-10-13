# MedPruner++

**Constructing Compact Volumetric Visual Representations for Medical Image Understanding**

[English](README.md) · [简体中文](README_CN.md) · [Evaluation protocol](docs/EVALUATION.md) · [MIT license](LICENSE)

MedPruner++ is a **training-free visual-token compression framework** for volumetric medical vision–language models. It takes the ordered visual features of a CT/MRI slice sequence and produces a shorter, ordered visual sequence for the language model. All backbone parameters remain frozen; no pruning policy is trained.

For example, a study represented by 32 slices with 256 visual tokens per slice has 8,192 visual tokens. With `retain_ratio=0.1`, the requested capacity is `floor(8192 × 0.1) = 819` tokens. The realized count can be smaller because support-unit capacities and integer allocation constrain the output. This is a visual-token budget, not a fraction of slices or a guaranteed runtime speedup.

<p align="center">
  <img src="assets/method_overview.png" width="100%" alt="MedPruner++: ordered support construction, radial–angular token selection and local refinement, with fixed or attention-calibrated capacity">
</p>

The content path groups recurrent contiguous slices into ordered support units, allocates capacity to those units, selects distinctive anchors through radial–angular feature relations, and refines anchors with related local features. A separate capacity path supports either a fixed budget or an attention-entropy-calibrated budget. Compression is applied after native visual encoding and before the retained visual sequence enters language decoding.

## Supported backbones and tasks

The released Transformers adapters expose the following combinations. Adapter availability does not substitute for reproducing a full benchmark on your hardware.

| Backbone | `MODEL_NAME` | MedPruner++ | Included comparison adapters |
| --- | --- | --- | --- |
| Qwen2.5-VL | `Qwen2.5-VL` | ✓ | VisionZip, HoliTom, UniComp |
| Qwen3-VL | `Qwen3-VL` | ✓ | VisionZip, HoliTom, UniComp |
| Hulu-Med (Qwen2) | `Hulumed_qwen2` | ✓ | VisionZip, HoliTom, UniComp |
| MedGemma | `MedGemma` | ✓ | VisionZip, HoliTom, UniComp |

| Benchmark | Exact CLI name | Task in this repository | Annotation file |
| --- | --- | --- | --- |
| AMOS-MM | `Amos` | CT report generation | `amos_val_mrg_sft.json` |
| M3D | `M3D` | Open and closed volumetric visual question answering | `M3D_test.json` |
| 3D-RAD | `3DRad` | Open and closed volumetric visual question answering | `Rad3D_test.json` |

The evaluation inputs are **preprocessed slice-image directories** referenced by annotation JSON files. The current loaders do not directly evaluate raw NIfTI volumes, perform lesion segmentation, or guarantee support for arbitrary DICOM files.

## Installation

Use Linux with an NVIDIA GPU for model inference. The supplied adapters use FlashAttention 2. Choose the PyTorch/CUDA build appropriate to your server; the following is a CUDA 12.4 example:

```bash
git clone https://github.com/yezanting/MedPruner-plus-plus.git
cd MedPruner-plus-plus
conda create -n medprunerpp python=3.10 -y
conda activate medprunerpp

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging ninja
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip install -e ./medcompose
python -m nltk.downloader wordnet omw-1.4
```

`transformers==4.57.3` matches the Qwen3-VL modules and decorator APIs used by the vendored model backends. The original environment list is retained as `requirements-legacy.txt` for provenance; it pins an older Transformers version and is not the release installation recipe. Baseline import and package checks are described in [the evaluation notes](docs/EVALUATION.md).

## Data and model preparation

Follow the [Hulu-Med evaluation data instructions](https://github.com/ZJUI-AI4H/Hulu-Med/tree/main/MedUniEval) and its [processed evaluation resources](https://modelscope.cn/models/Med-Team/Hulu-Med). Obtain datasets and checkpoints under their respective access and license terms; this repository does not distribute them.

```text
data/Eval/
├── Amos/
│   ├── amos_val_mrg_sft.json
│   └── amos_test_crop_32/
├── M3D/
│   └── M3D_test.json
└── 3DRad/
    └── Rad3D_test.json
```

Each annotation's `video[0]` must point to an existing directory containing the ordered slice images. Update paths to your actual data location. Relative paths resolve from the repository root when using the provided launchers. Filenames must sort in anatomical slice order, for example `0001.png`, `0002.png`, … . Model paths may be local checkpoint directories or compatible model identifiers, subject to the model's access requirements.

## Compression configuration

Both the method selector and the JSON key are **`medpruner++`**. The Python module is `medcompose.medpruner_pp`.

```json
{
  "medpruner++": {
    "frame_uniqueness": 0.005,
    "retain_ratio": 0.1,
    "spatial_uniqueness": 0.2,
    "auto_compress": false
  }
}
```

This is the MedPruner++ section of [`config/config.json`](config/config.json), which also contains baseline configurations.

| Parameter | Default | Meaning in the implementation |
| --- | --- | --- |
| `frame_uniqueness` | `0.005` | Lorentz-distance threshold for grouping contiguous slices against the first slice of the active support unit. Larger values permit more grouping; aggregation averages corresponding native features. |
| `retain_ratio` | `0.1` | Requested visual-token retention fraction relative to the original encoded sequence when `auto_compress=false`. |
| `spatial_uniqueness` | `0.2` | Local refinement threshold: a patch is an anchor's neighbor when `1 - S < threshold`, where `S` is row-normalized negative Lorentz distance. It does not set the TopK anchor count. |
| `auto_compress` | `false` | If true, replace the fixed budget with `rho = clamp(mean(H_normalized)^3 / 2, 0, 0.3)` from the available visual attention. |

Use a positive retention ratio at most 1 for fixed-budget runs. Automatic capacity is input-dependent and capped at 30%; it is not an automatic 10% setting. Very small budgets and degenerate feature/attention inputs require separate validation.

The model attribute `compression_rate` and output field `compose_rates.compose_rate` both mean **retained visual tokens / original visual tokens**. Thus `0.1` means 10% retained, or approximately 90% removed. Inspect realized retention rather than assuming it exactly equals the requested capacity.

## Evaluation

From the repository root, run an AMOS report-generation evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_NAME=Hulumed_qwen2 \
MODEL_PATH=/path/to/Hulu-Med-7B \
DATASETS_PATH=/path/to/Eval \
EVAL_DATASETS=Amos \
COMPRESSION=medpruner++ \
OUTPUT_PATH=workspace/eval/hulu-amos-medprunerpp \
bash MedUniEval/eval.sh
```

For M3D, the chunked launcher runs shards **sequentially**, using the visible GPU set for each model instance. It does not launch independent GPU workers:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_NAME=Qwen3-VL \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATASETS_PATH=/path/to/Eval \
CHUNKS=8 \
COMPRESSION=medpruner++ \
OUTPUT_PATH=workspace/eval/qwen3-m3d-medprunerpp \
bash MedUniEval/eval_M3D_chunked.sh
```

Set `EVAL_DATASETS=3DRad` with `eval.sh` for 3D-RAD. Set `COMPRESSION=none` for the uncompressed reference, or `visionzip`, `holitom`, `unicomp` for the included comparison implementations. Use a **different output directory per method/configuration**. Keep the checkpoint, input sampling, prompt, generation settings and metric code matched between runs.

| Variable | `eval.sh` default | M3D launcher default |
| --- | --- | --- |
| `EVAL_DATASETS` | `3DRad,Amos` | `M3D` |
| `MODEL_NAME` | `Hulumed_qwen2` | `Hulumed_qwen2` |
| `MODEL_PATH` | `workspace/model/hulu_7B` | Same |
| `DATASETS_PATH` | `data/Eval` | Same |
| `CONFIG_PATH` | `config/config.json` | Same |
| `COMPRESSION` | `medpruner++` | Same |
| `CUDA_VISIBLE_DEVICES` | `0` | `0` |
| `MAX_NEW_TOKENS` | `16384` | `256` |
| `MAX_IMAGE_NUM` | `600` | `600` |
| `SEED`, `TEMPERATURE`, `TOP_P` | `42`, `0`, `0.95` | Same |
| `USE_VLLM`, `USE_LLM_JUDGE` | `False`, `False` | Same |
| `CHUNKS` | Single run | `8` sequential shards |

The compression release uses Transformers adapters. `MAX_IMAGE_NUM` limits sampling in the M3D loader; the inherited 3D-RAD loader uses `max_frames=1800`, and AMOS loads its supplied crop sequence. See [protocol details and Hulu-Med differences](docs/EVALUATION.md).

M3D and 3D-RAD save `results.json`, `metrics.json`, and `wrong_answers.json` under the dataset output directory. AMOS saves generated reports and CSV exports; its first `metrics.json` is an instruction to run the separate [report metric script](MedUniEval/utils/Metrics_Compute/cal_report_metrics.py), not a final report score.

## Citation and acknowledgements

The manuscript metadata below comes from the author-provided draft. No published venue or DOI is asserted. [`CITATION.cff`](CITATION.cff) also provides a repository citation.

```bibtex
@misc{ye2026medprunerpp,
  title = {MedPruner++: Constructing Compact Volumetric Visual Representations for Medical Image Understanding},
  author = {Ye, Zanting and Lin, Yunrui and Liu, Shengyuan and Hao, Jing and Yuan, Yixuan and Zheng, Yefeng and Lu, Lijun},
  year = {2026},
  note = {Research manuscript; code release},
  url = {https://github.com/yezanting/MedPruner-plus-plus}
}
```

This work builds on [MedPruner](https://github.com/Saint-lsy/MedPruner) and uses an evaluation framework adapted from [Hulu-Med](https://github.com/ZJUI-AI4H/Hulu-Med). We acknowledge [Transformers](https://github.com/huggingface/transformers), [Qwen-VL](https://github.com/QwenLM/Qwen3-VL), [MedGemma](https://github.com/google-health/medgemma), [VisionZip](https://github.com/JIA-Lab-research/VisionZip), [HoliTom](https://github.com/cokeshao/HoliTom), and [UniComp](https://github.com/TimeMarker-LLM/UniComp).

- [MedPruner: Training-Free Hierarchical Token Pruning for Efficient 3D Medical Image Understanding in Vision-Language Models](https://arxiv.org/abs/2603.11625)
- [Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding](https://arxiv.org/abs/2510.08668)

## License

MedPruner++ project contributions are released under the [MIT License](LICENSE). Vendored and adapted third-party code retains its original copyright notices and license terms, including Apache 2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset and pretrained-weight licenses remain separate.
