python train_gan.py \
    --run_name "run/dcgan_k=1,lr_g=1.5x,no_bn" \
    --epoch 270 \
    --batch_size 256 \
    --train_steps 40000 \
    --log_steps 10 \
    --log_image_steps 101 \
    --test_steps 100 \
    --save_steps 5000 \
    --save_dir ./save \
    --gpu 4 \
    --lr 0.0002