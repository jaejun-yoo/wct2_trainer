## Wavelet PhotoRealistic Style Transfer

## Note from the author
This is for providing the training code of wavelet encoder and decoder. I found this version from my old disk so I am not sure this is the final version that I used for the paper. However, I think the `trainer_jjy.py` would be helpful for those who want to train the wave encoder and decoder. (You should modify the main.py if you want to use the full framework since there are some auxiliary headers and codes that I had to include when I was using the NSML GPU resource scheduler of NAVER-this code won't work as it is if you try to run it without modification.

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
