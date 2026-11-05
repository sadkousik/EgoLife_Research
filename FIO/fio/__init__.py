"""FIO memory library: extract, optimise and store fine-grained interaction objects.

Modules
-------
llm         Qwen2.5-7B backend (``call_qwen``); the model loads on first use, at a pinned revision.
prompts     FIO extraction prompt.
schema      FIO validation, document strings, recency scoring.
optimizer   ``FIOOptimizer``: filter, score and merge FIOs.
tracking    ``BuildTracker``: build reports.
exp_logger  ``start_experiment``: experiment log setup.
stats       ``analyze_chroma_dates``: per-day counts of a Chroma store.
repro       ``set_seed``, ``run_info``, ``write_json``: seeding and run records.
grounding   Grounded Accuracy (paper Sec. 4.1): ``answer_path``, ``classify``, ``summarize``.
storage     Chroma stores (``chroma``, ``dual``) and node-local staging (``staging``).

Nothing is imported here on purpose, so importing one module stays cheap.
"""
