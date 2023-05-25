# Prepare Dataset
our dataset is organized as follow:
```
ReflectanceMM
|-dataset
    |- ffhq
        |- train
            |- landmarks
                |- xxxxx.txt
                |- ...
            |- mask
                |- xxxxx.png
                |- ...
            |- xxxxx.png
            |- ...
        |- val
            |- landmarks
                |- yyyyy.txt
                |- ...
            |- mask
                |- yyyyy.png
                |- ...
            |- yyyyy.png
            |- ...
        |- ffhq-coeffs-train-refine
            |- xxxxx.pkl
            |- ...
        |- ffhq-coeffs-val-refine
            |- yyyyy.pkl
            |- ...
```

We provide the coeffs, landmarks, and masks at [here](https://cloud.tsinghua.edu.cn/f/837a65075f804bd7a988/).
After downloading the provided dataset, you need to request the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) and move all the 70K images to either the val or the train directories, according to the index. 

For example, if `12345.pkl` in `ffhq-coeffs-train-refine`, then `12345.png` should be placed into `train/12345.png`; if `02468.pkl` in `ffhq-coeffs-val-refine`, then `02468.png` should be placed into `val/12345.png`.
