"""Analysis code behind the RQ notebooks.

loader         Read every ``best_result.json`` / ``baseline_fail.json`` into one DataFrame.
box_iou        Detection IoU (same metric as ``src/objectives/image_criteria/_bbox_iou.py``).
termination    Termination classes of a run and the cumulative ASR_B / ASR_E / ASR_M indicators.
metrics        MS-SSIM, NED and SWTD per test case.
reporting      Matplotlib setup and the LaTeX table renderer.
rq12_stats     Significance tests for RQ1/RQ2 (``python -m experiments.analysis.rq12_stats``).
rq3_validity   Human-vs-model validity analysis of the survey (RQ3).
json_recovery  Repairing malformed VLM JSON with a local LLM (RQ4).
"""
