# SinGAN_colab
SinGAN([arxiv](https://arxiv.org/abs/1905.01164))'s implementation. 

This code can execute by colab(ipynb file) and your local VScode

----
# In your local

At first, git clone this repository.

1. Training

Just type this on terminal

```
python ./code/main.py
```

2. Validation

You no need to do verify your model because all of the elapsed data is stored at the time of learning.

But if you want validate your trained model, type like this on terminal

```
python ./code/main.py --validation 1 --img_size_max 1025 --load_model (folder name in your log folder, like "SinGAN_2021-07-21_11-02-24_zerogp" that I pretrained model)
```

----
# In colab

1. Create a new folder named "SinGAN_data" on your Google drive and upload the Train.jpg in this repository's "data" folder. (If you have other image that you want to learn, you can upload that)

![코랩에서 열기](https://i.ibb.co/4MVgmc9/image.png) <br>

2. Open "SinGAN.ipynb" in colab (Please refer to the picture above.)
3. Just enjoy!


reference : https://github.com/tamarott/SinGAN (ofiicial repository) & https://github.com/FriedRonaldo/SinGAN
