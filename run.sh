PYTHONPATH=./ python flux_fill_lora_finetune_split.py \
    --outdir lora-sphere-coords-testing-new-shape \
    --dataset datasets/house \
    --epochs 12000 --save-every 4000 --sphere-coords 1
