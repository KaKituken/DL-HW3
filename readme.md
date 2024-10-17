# CS 5787 HW 3

## Semi-supervision learning with VAE

### Train VAE

The model of vae is implemented in `model.py`, with a encoder and decoder.

To train the VAE, simply run

```bash
$ ./train.sh
```

Note that the param `--stage` should `1.`

The model will be saved at `save/model.pth`

### Train SVM

To train the svm on VAE's feature, also run

```bash
$ ./train.sh
```

This time the param `--stage` should be `2`, and specify the VAE checkpoint use `--ckpt`.

## GANs

### DCGAN

The model's architecture is in `models/DCGAN.py`.

To train the model, run


### DCGAN

The model's architecture is in `models/DCGAN.py`.

To train the model, run

```bash
./train_gan.sh
```

### WGAN

The model's architecture is in `models/WGAN.py`.

To train the model, run

```bash
./train_wgan.sh
```

### WGAN-GP

The model's architecture is also in `models/WGAN.py`. They share the same architecture.

To train the model, run

```bash
./train_wgan_gp.sh
```
