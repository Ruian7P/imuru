python train_writer_id.py \
    --train_batch_size 256 \
    --epochs 30 \
    --lr 1e-3 \
    --writer_id_config "./configs/writer_id/WriterID_ours.json" \
    --report_to "wandb" \
    --wandb_entity "emuru" \
    --wandb_project_name "ours_wid" \
    --wandb_log_interval_steps 100