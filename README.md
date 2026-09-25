# MoST: Search-Based Testing of Modality Sensitivity in Vision Language Models

Replication package for the paper *Search-Based Testing of Modality Sensitivity in Vision Language Models*.

MoST is a black-box, search-based test generator for vision-language models (VLMs). For an image and a
detection prompt it searches, with NSGA-II, for combinations of continuous-severity image and text
perturbations that degrade the VLM's bounding-box predictions while keeping the inputs visually and
semantically close to the originals. Running the search image-only, text-only and jointly shows which modality a
model is sensitive to.

MoST is built on a generic multi-objective testing framework (`src/`) that provides the SUT / manipulator /
objective / optimizer interfaces. It is reused infrastructure, not a contribution of this paper. Everything
specific to the paper lives in `defaults/mmm/`, `experiments/` and `survey/`.

---

## 1. Where each part of the paper is implemented

| Paper | Code |
|---|---|
| Fig. 1, overall loop (manipulate → SUT → objectives → optimizer) | `defaults/mmm/_mmm_tester.py` (loop), `experiments/run.py` (wiring of all components) |
| §3.1 SUT, prompt template (§4.2) | `src/sut/_vlm_sut.py`, `defaults/mmm/_prompts.py`, per-model settings in `experiments/config/models.py` |
| §3.2.1 image manipulations, Table 1 | `src/manipulator/pertubation_manipulator/_image_pertubation_manipulator.py` |
| §3.2.2 text manipulations, Table 2 | `src/manipulator/pertubation_manipulator/_textual_pertubation_manipulator.py` |
| §3.3 objective vector, Eq. (1) | `defaults/objective_configs.py` (`MMM`) |
| Eq. (2) detection degradation f1 (label-free Hungarian IoU) | `src/objectives/image_criteria/_bbox_iou.py`; pixel-box conversion in `defaults/mmm/_scoring.py` |
| Eq. (3) visual fidelity f2 | `src/objectives/image_criteria/_matrix_distance.py` |
| Eq. (4) semantic fidelity f3 (Qwen3-Embedding-0.6B) | `src/objectives/text_criteria/_embedding_distance.py`, `defaults/mmm/_qwen3_embedding.py` |
| §3.3.1 early stopping | `experiments/run.py` (`--early-stop-*`), `defaults/early_termination.py` |
| §3.4 strategy κ = [κ_img, κ_txt] | `defaults/mmm/_genome.py` |
| §3.4 budget-aware sampling, budget repair | `defaults/mmm/optimizer_modules/` |
| §4.4 NSGA-II configuration | `experiments/run.py` (CLI defaults), `defaults/optimizer_configs.py` |
| §4.5.1 image selection (SC-SI / SC-MI / MC / Driving), baseline IoU ≥ 0.5 | `experiments/initialize/data_selector.py`, `defaults/mmm/_mmm_tester.py` |
| §4.5.2 homophone / synonym tables (gpt-oss:120b) | `experiments/initialize/generate_mappings.py` → `experiments/auxiliary_files/*.json` |
| §4.5.3 run outcomes, Fig. 3 | `experiments/analysis/termination.py`, `experiments/RQ1_Effectiveness.ipynb` |
| Eq. (5) ASR_B / ASR_E / ASR_M, Fig. 4 | `experiments/analysis/termination.py`, `experiments/RQ1_Effectiveness.ipynb` |
| MS-SSIM, NED, Eq. (6) SWTD, Table 3 | `experiments/analysis/metrics.py`, `experiments/RQ1_Effectiveness.ipynb` → `experiments/effectiveness.csv` |
| McNemar / Wilcoxon + Holm tests (RQ1, RQ2) | `experiments/analysis/rq12_stats.py` → `experiments/stats/rq12_tests.csv` |
| Table 4, efficiency | `experiments/RQ2_Efficiency.ipynb` → `experiments/efficiency.csv` |
| §4.6 human study, Table 5, Fig. 5, Randolph's κ | `experiments/analysis/rq3_validity.py`, `experiments/RQ3_Validity.ipynb`, `experiments/survey_responses.csv` |
| RQ4, Fig. 6 (mean final IoU) and Fig. 7 (IoU reduction over the budget split) | `experiments/RQ4_ModalitySensitivity.ipynb` |

Additional material not reported in the paper:

| Analysis | Code |
|---|---|
| Choice of the minimum box-area fraction (0.015) | `experiments/Fig_BBoxAreaDistribution.ipynb` |
| Qualitative examples of generated test cases | `experiments/Fig_Examples.ipynb` |
| Perturbation attribution per operator family | `experiments/RQ4_Correlation.ipynb` |
| Repairing malformed JSON with a local LLM | `experiments/analysis/json_recovery.py`, `experiments/RQ4_Correlation.ipynb` |
| Re-scoring test cases at different sampling temperatures | `experiments/run_ablation.py`, `experiments/RQ5_Ablation_Comparison.ipynb` |

## 2. Repository layout

```
defaults/mmm/            MoST test loop and its helpers
  _mmm_tester.py           search loop per sample: baseline check, NSGA-II, early stop, saving
  _data.py                 load a selected sample (image, prompt, ground-truth boxes)
  _parsing.py              parse the VLM's JSON answer
  _scoring.py              predicted boxes -> pixel coordinates -> IoU (Eq. 2)
  _genome.py               genome -> image/text perturbation candidates
  _results_io.py           write best_result.json / .png, baseline_fail.json
  optimizer_modules/       budget-aware sampling and budget repair (§3.4)
defaults/*.py            objective, optimizer and early-termination presets of the framework
experiments/
  01_create_env.sh, 02_*.sh, 03_run_all.sh   end-to-end pipeline (Section 4)
  run.py                   one test-generation run (one model, one regime)
  config/                  paths, data-selection settings, model specs
  initialize/              dataset download, image selection, label/synonym/homophone tables
  analysis/                analysis code imported by the notebooks
  RQ*.ipynb                one notebook per research question (RQ4 has two)
  *.csv, stats/            tables produced by the notebooks
survey/                  human-study cases: manifest, per-case metadata, the three survey variants
src/                     testing framework (reused, see src/README.md)
tests/                   unit tests
```

The other subfolders of `defaults/` hold presets for unrelated uses of the framework and are not used here.

## 3. Checking the results without re-running

Raw results are not included because they are mostly image data (the clean and the perturbed image of every
run). What is included:

- the notebooks with their saved outputs (tables and figures as in the paper);
- `experiments/effectiveness.csv` (Table 3), `experiments/efficiency.csv` (Table 4);
- `experiments/stats/rq12_tests.csv`, all significance tests of RQ1 and RQ2;
- `experiments/survey_responses.csv`, the anonymised human-study responses, and `survey/`, the case metadata (RQ3).

## 4. Full reproduction

Hardware used for the paper: one NVIDIA L40S (48 GB). All VLMs are served through vLLM.

```bash
bash experiments/01_create_env.sh            # conda env + torch, flash-attn, vLLM
bash experiments/02_data_selection.sh        # select 100 images per scene type (ImageNet DET val, Udacity)
bash experiments/02_1_generate_mappings.sh   # optional: regenerate homophone/synonym tables (needs Ollama)
bash experiments/03_run_all.sh <gpu-id>      # all models x {multi, image, text}; resumes finished samples
```

Datasets are not redistributed. `02_data_selection.sh` expects the ILSVRC 2017 DET validation split under
`dataset/2017/ILSVRC/` and the Udacity driving data under `dataset/udacity/`; fetch the latter first with
`python experiments/initialize/download_udacity.py`.
Paths are set in `experiments/config/paths.py`.

A single run:

```bash
python experiments/run.py --vlm qwen --mode multi     # --mode {multi,image,text}
```

Results are written to `defaults/mmm/results/<model>/{multimodal,unimodal/image,unimodal/text}/<scene>/<id>/`.
Each directory holds `best_result.json` (objectives, genome, VLM answer, ground truth, runtime) and
`best_result.png`, or `baseline_fail.json` when the clean input already fails (baseline IoU < 0.5).

Then, from `experiments/`, run the `RQ*.ipynb` notebooks and
`python -m experiments.analysis.rq12_stats`. MS-SSIM values are cached in `experiments/cache/` after the first run.

`03_run_all.sh` also runs Gemma-3 and DeepSeek-VL2. They are part of Fig. 3, but the later analyses exclude
them because too few of their runs end with valid boxes (§4.5.3).

## 5. Human study (RQ3)

- `experiments/survey_responses.csv`: one row per answer (62 accepted HITs × 48 tasks = 2,976 assessments),
  with pseudonymous annotator and session ids.
- `survey/samples/manifest.json` and `survey/samples/<case>/metadata.json`: for each of the 144 survey cases,
  the model, regime and sample it came from, its ground-truth boxes and the model's IoU, copied from the
  corresponding `best_result.json`.
- `survey/data/variants/variant-{1,2,3}.json`: the three survey subsets (48 cases each) with the task order shown to participants; the images themselves are not included.

`RQ3_Validity.ipynb` reads `survey/` by default; set `SURVEY_ROOT` to use another location.

## 6. Configuration used in the paper

| Setting | Value | Where |
|---|---|---|
| Algorithm | NSGA-II, SBX (p = 0.9, η = 15), PM (η = 20) | `experiments/run.py` |
| Population / generations | 50 / 100 (≤ 5,050 evaluations per run) | `run.py --pop-size`, `--num-generations` |
| Total budget B | 1.0 | `run.py --budget-max` |
| Early stop | f1 ≤ 0.25, f2 ≤ 0.1, f3 ≤ 0.3 | `run.py --early-stop-*` |
| Baseline filter | IoU ≥ 0.5 on the clean input | `run.py --baseline-iou-min` |
| Minimum ground-truth box area | 1.5% of the image | `run.py --min-bbox-area-fraction` |
| Max. image side | 1024 px | `run.py --max-resolution` |
| Seed | 42669 | `run.py --seed`, `experiments/config/experiment.py` |
| ASR threshold τ | 0.25 | `experiments/analysis/termination.py` |
| Models | Qwen3-VL-4B, Kimi-VL-A3B, InternVL3.5-8B, Nemotron-3-Nano-Omni-30B (+ Gemma-3-4B, DeepSeek-VL2-Tiny) | `experiments/config/models.py` |

## 7. Notes

- IoU ignores predicted labels (§3.3): a synonym or homophone in the prompt may change how the model names an
  object, and label matching would count a correctly localised object as a failure. Labels are only used when
  loading a sample, to find the ground-truth box of each prompt object.
- SWTD is not computed for runs that end in malformed JSON (there are no boxes to score). For Kimi-VL, where most
  runs end this way, some Table 3 cells therefore rest on very few runs.
- `experiments/analysis/box_iou.py` repeats the IoU of `src/` so that the analysis does not depend on vLLM;
  `tests/test_box_iou.py` checks that the two agree.
