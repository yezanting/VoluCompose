# Third-party notices

MedPruner++ project contributions are distributed under the root MIT license. This does not replace licenses or copyright notices on vendored code, model weights or datasets.

## Hulu-Med and its evaluation framework

`MedUniEval/` is adapted from [ZJUI-AI4H/Hulu-Med](https://github.com/ZJUI-AI4H/Hulu-Med), distributed under [Apache License 2.0](licenses/Apache-2.0.txt). Hulu model adapters under `medcompose/*/Hulu/` also incorporate Hulu-Med, LLaVA and Transformers code. Original file headers are retained. Changes include volumetric compression integration and retention metadata, dataset path/sampling adaptations, and release launcher fixes; see [evaluation provenance](docs/EVALUATION.md).

## Transformers and inherited model adapters

The Qwen, Gemma, SigLIP, Qwen2 and other model implementations include code from [Hugging Face Transformers](https://github.com/huggingface/transformers) under Apache License 2.0. File headers identify their original copyright holders, including the Qwen, Google and Hugging Face teams. Compression-specific model modifications are included in `medcompose/`; those files retain their upstream headers. The inherited LLaVA/HealthGPT/Huatuo and PEFT adapters also retain applicable original notices. The root MIT license does not remove those obligations.

## Comparison methods

The included comparison adapters adapt the following research projects. Their upstream licenses were checked for this release:

| Project | Upstream license | Local notice |
| --- | --- | --- |
| [VisionZip](https://github.com/JIA-Lab-research/VisionZip) | Apache 2.0 | [VisionZip license](licenses/VisionZip.txt) |
| [HoliTom](https://github.com/cokeshao/HoliTom) | Apache 2.0 | [HoliTom license](licenses/HoliTom.txt) |
| [UniComp](https://github.com/TimeMarker-LLM/UniComp) | Apache 2.0 | [UniComp license](licenses/UniComp.txt) |
| [MedPruner](https://github.com/Saint-lsy/MedPruner) | MIT | [MedPruner license](licenses/MedPruner.txt) |

The included adaptations are not asserted to be byte-identical to the original implementations. Cite the corresponding projects when using or comparing them.

## Weights, data and images

Pretrained model weights and evaluation datasets are not released by this repository. Obtain and use them under their original licenses and access conditions, including any model-specific terms. Historical bundled benchmark answer/result files and the unrelated MMLU-Pro test annotation are excluded from the public tree.

`assets/method_overview.png` and its editable SVG come from the author-provided MedPruner++ manuscript assets. They are a method illustration, not patient imaging or a newly validated experimental result.
