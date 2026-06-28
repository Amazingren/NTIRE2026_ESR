# --- Evaluation on LSDIR_DIV2K_valid datasets for One Method: ---
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir [path to your data dir] \
    --save_dir [path to your save dir] \
    --model_id 0


# --- When only LSDIR_DIV2K_test datasets are included (For Organizer) ---
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir ../ \
#     --save_dir ../results \
#     --include_test \
#     --model_id 0

# --- Test all the methods (For Organizer) ---
#!/bin/bash
# DATA_DIR="/Your/Validate/Datasets/Path"
# SAVE_DIR="./results"
# MODEL_IDS=(
#     0 1 4 5 6
#     9 10 11 12 15
#     16 17 18 19 20
#     21 22
# )

# for model_id in "${MODEL_IDS[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --include_test --model_id "$model_id"
# done
