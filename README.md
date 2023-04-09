## Wavelet PhotoRealistic Style Transfer

### Train
```.bash
nsml run -d COCO2014 -a "--feature_weight 10 --recon_weight 1000 --img_size 256 --batch_size 32 --block 4(or 3 or 2)"
```

block selects which conv{}_1 to train
e.g) if block == 4, conv4_1 will be trained
therefore, if you need to train whole vgg, separately run block 4, 3, 2

### Transfer

get models from works drive.

and locate it in "model" directory. e.g) model/conv{}_1 and model/dec{}_1

```.bash
nsml run -e transfer.py --content_path --content_seg_path --style_path --style_seg_path
--alpha --conv_level(how many conv{}_1 to use) --transform_index(which component to transfer)
```