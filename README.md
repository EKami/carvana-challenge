# Carvana challenge

This repository holds the code for the [carnava image making challenge](https://www.kaggle.com/c/carvana-image-masking-challenge). It's meant to show how to construct Unets with Pytorch in a concise and straightforward way.

# Dependencies

 - Pytorch 0.2.0
 - Sklearn
 - Numpy
 - [OpenCV-Python](https://pypi.python.org/pypi/opencv-python)
 - [Pillow SIMD](https://github.com/uploadcare/pillow-simd) (for speedup) or the original [Pillow](https://github.com/python-pillow/Pillow)
 - [Kaggle data downloader](https://github.com/EKami/kaggle-data-downloader)
 - [Tensorboard Pytorch](https://github.com/lanpa/tensorboard-pytorch)

# Usage

In the `notebook/` folder you'll find a jupyter notebook containing a little exploratory data analysis.
To run the script you'll need to set 2 variables in your environment, `KAGGLE_USER` and `KAGGLE_PASSWD`:
```
export KAGGLE_USER="your_kaggle_username"
export KAGGLE_PASSWD="your_kaggle_password"
```

This will allow you to automatically check and download the required dataset from Kaggle.
When it's done simply execute the main file with:
```
python src/main.py
```

If you want to take a look at the prediction at each epochs you can use tensorboard with:
```
tensorboard --logdir=./logs
```
