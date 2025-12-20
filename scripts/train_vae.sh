python -W ignore train_vae.py \
    --htr_path "pretrained_models/emuru_vae_htr" \
    --writer_id_path "pretrained_models/emuru_vae_writer_id" \
    --train_batch_size 64 \
    --epochs 2 \
    --lr 5e-5 \
    --mixed_precision fp16 \
    --report_to "wandb" \
    --disable_writer_id_loss \