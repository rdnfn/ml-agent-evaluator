# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Experiment: Rewardbench non-code, non-maths
# gpt4o baselines
DATANAMES="alpacaeval-easy, alpacaeval-hard, alpacaeval-length, donotanswer, llmbar-adver-GPTInst, llmbar-adver-GPTOut, llmbar-adver-manual, llmbar-adver-neighbor, llmbar-natural, mt-bench-easy, mt-bench-hard, mt-bench-med, refusals-dangerous, refusals-offensive, xstest-should-refuse, xstest-should-respond"
DUMMY="1,2,3"
ageval -m -cd exp/configs/2024_08_22_paper/001_longfact/001_gpt4o_baselines data_name="${DATANAMES}" dummy="${DUMMY}" data_path=null
# alpacaeval baseline
ageval -m -cd exp/configs/2024_08_22_paper/001_longfact/002_ae_baseline data_name="${DATANAMES}"  dummy="${DUMMY}" data_path=null
# gpt35 baseline and agent
ageval -m -cd exp/configs/2024_08_22_paper/001_longfact/004_gpt35t_baseline_and_agent data_name="${DATANAMES}"  dummy="${DUMMY}" data_path=null
# gpt4o agent
ageval -m -cd exp/configs/2024_08_22_paper/001_longfact/005_gpt4o_agent data_name="${DATANAMES}"  dummy="${DUMMY}" data_path=null
# gemini baseline (currently very rate limited, need other key to run these experiments)
# ageval -m -cd exp/configs/2024_08_22_paper/001_longfact/003_gemini_baseline data_name="${DATANAMES}" data_path=null