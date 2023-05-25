python train.py \
    --gpu_id 0 \
    --save_epoch_freq 5 \
    --data_root dataset/ffhq \
    --dataset_mode refmm \
    --batch_size 16 \
    --name pretrain \
    --camera_rig_path RefMM/camera_rig.pkl \
    --update_model False

python train.py \
    --gpu_id 0 \
    --n_epochs 5 \
    --evaluation_freq 1000 \
    --vis_batch_nums 4 \
    --display_freq 250 \
    --data_root dataset/ffhq \
    --dataset_mode refmm \
    --batch_size 8 \
    --name finetune \
    --camera_rig_path RefMM/camera_rig.pkl \
    --update_model True \
    --load_prefit_net True \
    --prefit_net_path checkpoints/pretrain/epoch_latest.pth

python ckpt_to_model.py \
    --ckpt_path checkpoints/finetune/epoch_latest.pth \
    --refmm_save_path RefMM/finetuned_refmm_model.pkl
