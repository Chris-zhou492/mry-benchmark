# Dataset attribution and license notes

This file records the conditions found in the pinned official sources. It is a practical attribution guide, not legal advice. Follow the upstream license text and any source-specific notice when using or redistributing data.

| Dataset | Attribution / license | Release-specific note |
| --- | --- | --- |
| LongMemEval | The official repository and Hugging Face dataset identify MIT. Cite the LongMemEval paper and retain its upstream notice when redistributing. | The cleaned dataset documentation references third-party filler sources. This repository distributes only the small oracle file; larger S/M variants are retrieved from the official source. Check upstream provenance before creating or redistributing derivatives. |
| LoCoMo | CC BY-NC 4.0. Attribute Snap Research and the LoCoMo paper; non-commercial use only. [License text](https://creativecommons.org/licenses/by-nc/4.0/). | This release contains the text JSON only. Image assets are not included; embedded image references/captions do not grant rights to those images. |
| PersonaMem v1 | MIT in the official HF dataset metadata. Cite the PersonaMem paper and retain the upstream notice. | Only the pinned 32K questions and shared contexts are bundled. |
| PersonaMem v2 | CC BY 4.0. Attribute the PersonaMem v2 authors, identify modifications, and include a link to the license. [License text](https://creativecommons.org/licenses/by/4.0/). | Text benchmark and linked histories are obtained from the official pinned HF revision. |
| CL-bench Life | Custom evaluation-only terms supplied by Tencent. Use only for evaluation, testing, or model benchmarking. No training, fine-tuning, calibration, distillation, adaptation, or parameter updates. | The upstream `LICENSE.txt` is included at `datasets/bundled/cl-bench-life/LICENSE.txt` and must accompany any redistributed copy. |
| BEAM | BEAM code is MIT, while its benchmark data is CC BY-SA 4.0. Attribute the dataset authors, link the license, identify changes, and share adaptations under the same terms. [License text](https://creativecommons.org/licenses/by-sa/4.0/). | The bundled parquet files are data, not BEAM source code. |
| ScriptMem | CC BY-NC 4.0 for benchmark-created question materials. Attribute Memorax AI and the ScriptMem project; non-commercial use only. [License text](https://creativecommons.org/licenses/by-nc/4.0/). | The CC license does not cover third-party scripts. No original scripts or conversations are included in the pinned public package. |

The original MRY-Benchmark code and documentation do not yet have a separate repository-wide license. The dataset licenses above apply only to their respective upstream data.
