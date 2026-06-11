X-RLM Five-Model Natural Questions RAG Guide
============================================

Project path:
/Users/htet/Desktop/Projects/X-RLM

Dataset:
/Users/htet/Desktop/Projects/X-RLM/Data/NaturalQuestions/train/train.parquet

This guide is for the five-model comparison:

1. Llama 3 8B + RAG
2. GRLM + RAG
3. RLM REPL + RAG
4. RLM REPL 8-step + RAG
5. Focused RLM REPL + RAG

The matching plot script is:

/Users/htet/Desktop/Projects/X-RLM/plot_six_rag_models.py


Setup
=====

Run this first:

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate

Install or repair packages:

pip install pandas pyarrow numpy matplotlib langgraph langchain langchain-core langchain-community langchain-chroma chromadb llama-cpp-python sentence-transformers nltk

For Mac Metal acceleration:

CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python


Model
=====

Default local model:

QuantFactory/Meta-Llama-3-8B-Instruct-GGUF
Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

The scripts load the model through llama_cpp:

from llama_cpp import Llama


The five experiments
====================

1. Llama 3 8B + RAG
-------------------

Plain LangGraph + Chroma RAG baseline.

Pipeline:
retrieve -> build prompt -> Llama answer -> F1/BLEU/ROUGE -> LLM judge -> CSV

Command:

python /Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_eval.py \
  --max-rows 10 \
  --rebuild-index

Outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_rag_llama_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_rag_llama_summary.csv


2. GRLM + RAG
-------------

Genetic prompt search plus RAG.

Pipeline:
retrieve -> prompt population -> prompt scoring/mutation -> best prompt -> Llama answer -> F1/BLEU/ROUGE -> LLM judge -> CSV

Command:

python /Users/htet/Desktop/Projects/X-RLM/nq_langgraph_genetic_prompt_eval.py \
  --max-rows 10 \
  --use-genetic-prompt \
  --rebuild-index

Outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_genetic_rag_llama_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_genetic_rag_llama_summary.csv


3. RLM REPL + RAG
-----------------

Algorithm-1 REPL-style recursive language model with external P values.

Pipeline:
retrieve -> InitREPL(P) -> AddFunction(sub_RLM) -> LLM proposes REPL code/actions -> safe REPL executes action -> stdout metadata goes back into history -> FINAL/root answer -> F1/BLEU/ROUGE -> LLM judge -> CSV

Command:

python /Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_eval.py \
  --max-rows 10 \
  --repl-max-steps 4 \
  --rebuild-index

Outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_outputs/nq_rlm_repl_rag_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_outputs/nq_rlm_repl_rag_summary.csv


4. RLM REPL 8-step + RAG
------------------------

This is a separate expanded-budget REPL experiment. It keeps the original
RLM REPL scaffold but allows up to 8 REPL actions before fallback.

Command:

python /Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_eval.py \
  --max-rows 10 \
  --rebuild-index

Outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_outputs/nq_rlm_repl8_rag_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_outputs/nq_rlm_repl8_rag_summary.csv


5. Focused RLM REPL + RAG
-------------------------

This is the focused REPL experiment. It does not replace the original
nq_rlm_repl_rag_eval.py file.

Pipeline:
retrieve -> InitREPL(P) -> understand_question -> extract_answer_evidence -> generate_final_answer -> F1/BLEU/ROUGE -> LLM judge -> CSV

Command:

python /Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_eval.py \
  --max-rows 10 \
  --rebuild-index

Outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_outputs/nq_rlm_repl_focused_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_outputs/nq_rlm_repl_focused_summary.csv

Important trace columns:

question_understanding_json
selected_evidence_handles
snippet_classifications_json
rlm_sub_results_json
rlm_root_prompt
rlm_trace_json
focused_repl_job_sequence


Run all five in sequence
========================

Use this block for the clean 10-question comparison:

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate

python nq_langgraph_rag_eval.py \
  --max-rows 10 \
  --rebuild-index

python nq_langgraph_genetic_prompt_eval.py \
  --max-rows 10 \
  --use-genetic-prompt \
  --rebuild-index

python nq_rlm_repl_rag_eval.py \
  --max-rows 10 \
  --repl-max-steps 4 \
  --rebuild-index

python nq_rlm_repl8_rag_eval.py \
  --max-rows 10 \
  --rebuild-index

python nq_rlm_repl_focused_eval.py \
  --max-rows 10 \
  --rebuild-index

python plot_six_rag_models.py


Run all five for 500 questions
==============================

This can take a long time because GRLM and the RLM variants use more LLM calls
than plain RAG.

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate

python nq_langgraph_rag_eval.py \
  --max-rows 500 \
  --rebuild-index

python nq_langgraph_genetic_prompt_eval.py \
  --max-rows 500 \
  --use-genetic-prompt \
  --rebuild-index

python nq_rlm_repl_rag_eval.py \
  --max-rows 500 \
  --repl-max-steps 4 \
  --rebuild-index

python nq_rlm_repl8_rag_eval.py \
  --max-rows 500 \
  --rebuild-index

python nq_rlm_repl_focused_eval.py \
  --max-rows 500 \
  --rebuild-index

python plot_six_rag_models.py


Optional faster run without LLM judge
=====================================

If you only want automatic F1/BLEU/ROUGE and want to save time, add:

--skip-llm-judge

Example:

python nq_rlm_repl_rag_eval.py \
  --max-rows 10 \
  --repl-max-steps 4 \
  --skip-llm-judge \
  --rebuild-index


Five-model plot outputs
=======================

After running:

python /Users/htet/Desktop/Projects/X-RLM/plot_six_rag_models.py

The outputs are written to:

/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports

Main outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_comparison_summary.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_chart_ready_long.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_metric_correlations.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_quality_runtime_scatter_data.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_short_long_comparison.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_quality_metrics.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_short_long_metric_comparison.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_short_vs_long_f1_scatter.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_runtime.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_llm_judge_scores.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_quality_vs_runtime_scatter.png


Five-model trace comparison outputs
===================================

After all five detailed trace CSVs exist, run:

python /Users/htet/Desktop/Projects/X-RLM/compare_five_trace_outputs.py

This reads question-level trace CSVs, including:

/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_rag_llama_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_genetic_rag_llama_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_outputs/nq_rlm_repl_rag_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_outputs/nq_rlm_repl_focused_assessment_traces.csv

Outputs are written to:

/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models

Main trace comparison outputs:

/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_model_question_comparison_long.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_metric_summary_by_model.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_best_model_by_question.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_question_difficulty.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_quality_metrics_by_model.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_llm_judge_score_by_model.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/trace_compare_five_models/trace_best_model_counts.png


How to compare fairly
=====================

Use the same --max-rows value for all five models.

For 10-question comparison, all five summaries should show:

rows_evaluated = 10

For 500-question comparison, all five summaries should show:

rows_evaluated = 500

Do not compare:

- dry-run output with real Llama output
- 10-row output with 500-row output
- outputs with LLM judge against outputs using --skip-llm-judge


Metrics in the CSV
==================

short_exact_match:
1 if the predicted short answer exactly matches one reference short answer.

rag_short_token_f1:
Token-level F1 for the short answer.

rag_long_token_f1:
Token-level F1 for the long answer.

rag_short_bleu / rag_long_bleu:
BLEU score for short and long answers.

rag_short_rouge1 / rag_long_rouge1:
ROUGE-1 score for short and long answers.

llm_judge_score_0_to_5:
Local Llama-as-judge score from 0 to 5.

faithfulness_fused_score:
Combined score using available short/long F1, short/long ROUGE-1, and normalized
LLM judge score.

estimated_lm_calls:
Approximate number of LLM calls used by that model.

recursive_call_count:
Number of sub_RLM evidence calls used by RLM REPL.

repl_step_count:
Number of Algorithm-1 REPL iterations executed.

repl_code_call_count:
Number of LLM calls used to generate REPL actions/code.

elapsed_seconds:
Mean seconds per question in the summary CSV, or per-question seconds in the
trace CSV.

elapsed_display:
Readable timing like elapsed=156.542s.


Trace CSV guide
===============

Open the summary CSV first:

/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/five_model_comparison_summary.csv

Then open each detailed trace CSV if you need question-level details.

Useful readable columns:

- experiment_name
- row_explanation
- question
- answer_summary
- retrieval_summary
- trace_summary
- judge_summary
- elapsed_display
- paper_method_mapping
- estimated_lm_calls
- recursive_call_count
- repl_step_count
- repl_code_call_count
- faithfulness_fused_score
- compound_call_warning

For RLM REPL + RAG and Focused RLM REPL + RAG, inspect these P-value columns:

- p_question
- p_instruction
- p_handle
- p_snippet_count
- p_total_snippet_chars
- p_available_handles
- p_summary
- recursive_call_count
- repl_step_count
- repl_code_call_count
- repl_history_json
- repl_final_state_json
- repl_allowed_functions
- repl_algorithm_mapping
- rlm_environment_json
- rlm_root_prompt
- rlm_sub_results_json
- rlm_trace_json

For Focused RLM REPL + RAG, also inspect:

- focused_repl_job_sequence
- question_understanding_json
- question_understanding_prompt
- selected_evidence_handles
- snippet_classifications_json
- question_understanding_call_count
- final_generation_call_count


Troubleshooting
===============

Problem:
plot_six_rag_models.py shows missing files.

Fix:
Run the five evaluator scripts first. The plotter only reads existing summary
CSVs; it does not generate model answers.


Problem:
A command says "unrecognized arguments".

Fix:
Make sure you are running one of these:

/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_eval.py
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_genetic_prompt_eval.py
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_rag_eval.py
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_rag_eval.py
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl_focused_eval.py


Problem:
The chart says mixed sample sizes.

Fix:
Rerun all five models with the same --max-rows value.


Problem:
GRLM or an RLM REPL run is too slow.

Fix:
Test with --max-rows 1 first, then --max-rows 10, then scale to 500.
You can also add --skip-llm-judge to save time.


Recommended workflow
====================

1. Run all five with --max-rows 1.
2. Run plot_six_rag_models.py and check that plots are created.
3. Run all five with --max-rows 10.
4. Open five_model_comparison_summary.csv.
5. If everything looks correct, run all five with --max-rows 500.


GA vs Evolution Strategy Prompt Search
======================================

This section compares two prompt-search methods using the same Natural Questions
RAG evaluator and the first 100 questions.

1. GA + RAG
-----------

Existing genetic algorithm prompt search.

Strategy:
- prompt population
- scoring by local Llama prompt judge
- elitism with keep_top
- crossover between surviving prompts
- mutation after crossover

Source code:
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_genetic_prompt_eval.py

Trace CSV:
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_rag_outputs/nq_genetic_rag_llama_assessment_traces.csv

2. ES + RAG
-----------

New evolutionary strategy prompt search.

Strategy:
- prompt population
- scoring by local Llama prompt judge
- tournament selection
- elitism with keep_top
- mutation only
- no crossover

Source code:
/Users/htet/Desktop/Projects/X-RLM/nq_langgraph_evolution_strategy_eval.py

Output folder:
/Users/htet/Desktop/Projects/X-RLM/nq_evolution_strategy_rag_outputs

Trace CSV:
/Users/htet/Desktop/Projects/X-RLM/nq_evolution_strategy_rag_outputs/nq_evolution_strategy_rag_llama_assessment_traces.csv

Summary CSV:
/Users/htet/Desktop/Projects/X-RLM/nq_evolution_strategy_rag_outputs/nq_evolution_strategy_rag_llama_summary.csv

Run ES for 100 questions
------------------------

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate

python nq_langgraph_evolution_strategy_eval.py \
  --max-rows 100 \
  --use-evolution-strategy \
  --population-size 6 \
  --generations 3 \
  --keep-top 2 \
  --tournament-size 3 \
  --rebuild-index

Compare GA vs ES for 100 questions
-----------------------------------

After the ES trace CSV exists, run:

python compare_ga_es_prompt_strategies.py

Comparison script:
/Users/htet/Desktop/Projects/X-RLM/compare_ga_es_prompt_strategies.py

Output folder:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100

Output CSVs:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_summary.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_trace_rows_long.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_chart_ready_long.csv

Output plots:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_quality_metrics.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_runtime.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/ga_vs_es_first100/ga_vs_es_first100_fused_components.png

Important note:
If ES has not been run yet, compare_ga_es_prompt_strategies.py will still create
plots and CSVs from the GA trace, but the ES row will be marked missing or empty.
For a fair comparison, both GA and ES should have 100 completed rows.

Compare all available model outputs for first 100 questions
-----------------------------------------------------------

Use this when you want one CSV and one plot set across every completed model
trace currently available in the project.

Comparison script:
/Users/htet/Desktop/Projects/X-RLM/compare_all_models_first100.py

Run:

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate
python compare_all_models_first100.py --max-rows 100

Models read by this script when their trace CSV exists:
- Llama 3 8B + RAG
- GA + RAG
- ES + RAG
- RLM REPL + RAG
- RLM REPL 8-step + RAG
- RLM REPL 12-step + RAG
- Focused RLM REPL + RAG

Output folder:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100

Output CSVs:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_summary.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_trace_rows_long.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_chart_ready_long.csv

Output plots:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_quality_metrics.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_runtime.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_balanced_fused_score.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_first100/all_models_first100_short_vs_long_f1.png

Metrics included:
- short and long answer F1
- short and long BLEU
- short and long ROUGE-1
- normalized LLM-as-judge score
- retrieval self-hit
- elapsed seconds per question
- estimated LM calls when available
- balanced fused score

Balanced fused score:

fused_score =
  0.15 * short_f1
+ 0.20 * long_f1
+ 0.10 * short_bleu
+ 0.10 * long_bleu
+ 0.10 * short_rouge1
+ 0.10 * long_rouge1
+ 0.20 * llm_judge_score_norm
+ 0.05 * retrieved_self_hit
- runtime_penalty
- call_penalty

Important note:
The summary CSV includes rows_used and is_full_100. If a model only has 50
completed rows, it is still plotted, but is_full_100 is False. Treat those
partial-model scores as early signals, not a direct fair 100-question result.

Run RLM REPL 8-step as a separate full 100-question experiment
--------------------------------------------------------------

Use this new file when you want the 8-step RLM REPL result to be directly
comparable with the 100-question Llama, GA, ES, RLM REPL, and Focused RLM
outputs. This does not overwrite the earlier 50-question 8-step output.

Evaluator script:
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_full100_rag_eval.py

Run full 100:

cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate

python nq_rlm_repl8_full100_rag_eval.py \
  --max-rows 100 \
  --repl-max-steps 8

New output folder:
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_full100_rag_outputs

New evaluator CSV outputs:
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_full100_rag_outputs/nq_rlm_repl8_full100_rag_assessment_traces.csv
/Users/htet/Desktop/Projects/X-RLM/nq_rlm_repl8_full100_rag_outputs/nq_rlm_repl8_full100_rag_summary.csv

After the full-100 run finishes, regenerate the fair comparison plots:

python compare_all_models_full100_with_repl8.py --max-rows 100

Comparison script:
/Users/htet/Desktop/Projects/X-RLM/compare_all_models_full100_with_repl8.py

Comparison output folder:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8

Comparison CSV outputs:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_summary.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_trace_rows_long.csv
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_chart_ready_long.csv

Comparison plots:
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_quality_metrics.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_runtime.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_balanced_fused_score.png
/Users/htet/Desktop/Projects/X-RLM/nq_clean_monitor_exports/all_models_full100_with_repl8/all_models_full100_with_repl8_short_vs_long_f1.png

Important note:
The full-100 8-step run is slow. The previous 50-question run took about
4961.74 seconds, so 100 questions may take roughly 2.5 to 3 hours on the
MacBook. Keep the terminal open until the final [done] lines are printed.

Plot
cd /Users/htet/Desktop/Projects/X-RLM
source venv/bin/activate
python compare_seven_500_metrics.py