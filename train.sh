python train.py \
    --run_name run/vae_Adam_1e-3_RELU_init \
    --epoch 270 \
    --batch_size 256 \
    --train_steps 40000 \
    --log_steps 10 \
    --log_image_steps 101 \
    --test_steps 100 \
    --save_steps 5000 \
    --save_dir ./save \
    --gpu 5 \
    --lr 0.001 \
    --stage 2  \
    --ckpt ./save/run/vae_Adam_1e-3_RELU_init/model_40000.pth
    