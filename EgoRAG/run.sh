#!/bin/bash
#SBATCH --job-name=EgoRAGBuild7d_JAKE_BUILD_ans
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --partition=gpu_h100#gpu_l40#gpupart24hr#gpupart24hr#gpupart12hr  #gpupart12hr
#SBATCH --cpus-per-task=1         # Use enough CPU for faster compilation
#SBATCH --qos=limited#limited q
#set -x
#set -e

echo "Running on $(hostname)"
nvidia-smi

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt
# 👇 EXPERIMENT DESCRIPTION (CHANGE THIS ONLY)
export EXP_DESC="DB_BUILD_TASHA"
export SLURM_STDOUT=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
export SLURM_STDERR=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err



#python3 main.py --name "A1_JAKE" --db_name "Jake_All_Day_Qwen_Question" \
#--query_output_dir "query_results_All_Day_Qwen_Question" \
#--answer_output_dir "answer_results_All_Day_Qwen_Question" \
#--diary_dir "events_diary_All_Day_Qwen_Question" \
#--stage create query answer #answer #create query answer

python3 create_database.py --db_name "DB_EGO_life_JAKE_2306" --diary_dir "event_DB_EGO_life_JAKE_2306" --json_path "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/meta_q_guided_7days_jake_05-02-2026-1.json"

# # create database from json
# python3 main.py \
#     --name "A1_JAKE" \A3_TASHA
#     --db_name "DB_EGO_life_WITHOUT_meta_7d_15042026" \
#     --stage query \answer
#     --diary_dir "event_DB_EGO_life_WITHOUT_meta_7d_15042026" \
#     --video_dir "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/data/EgoLife"
# #    --query_result_json "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/query_results/20260218_045113_DB_EGO_life_meta_7d_170226_EgoLifeQA_A1_JAKE_results.json"
# #   --query_json "path/to/QA.json" #already added in the code


python main.py \
  --stage query answer \
  --name "A1_JAKE" \
  --db_name "DB_EGO_life_JAKE_2306" \
  --diary_dir "event_DB_EGO_life_JAKE_2306" \
  --query_json "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoLifeQA_A1_JAKE.json"\
  --video_dir "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/data/EgoLife"
#   --query_result_json "/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG/query_results/20260623_081129_DB_EGO_life_TASHA_2306_EgoLifeQA_A3_TASHA_results.json"


# cd /home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG
# rm -rf event_DB_EGO_life_TASHA_1706
# python -c "
# from egorag.utils.gen_event import gen_event
# gen_event('DB_EGO_life_TASHA_1706', 'event_DB_EGO_life_TASHA_1706',
#           db_dir='/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoGPT/chroma_db', staging=True)
# "