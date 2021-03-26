# DeepSRQ
Blind quality assessment for image superresolution using deep two-stream convolutional networks, published in Information Sciences 2020.

## Download Databases
Three databases are involved in our experiments, inluding SRID, CVIU, and QADS. In this demo, we use the latest [QADS](http://www.vista.ac.cn/super-resolution/) as an example.

## Extract Structure and Texture Information
```
matlab extract_structure.m
```
```
python extract_texture.py
```

## Divide Images into Patches
```
matlab convert_patch.m
```

## Read Data
```
train_structure.txt: /data/zhouw/QADS/structure_patch32/img05_3_08/img05_3_08-1.bmp 0.62647

train_lbp.txt: /data/zhouw/QADS/lbp_patch32/img05_3_08/img05_3_08-1.bmp 0.62647
```

## Training
```
python train_model.py
```

## Testing
```
python test_model.py
```

## Citation
You may cite it in your paper. Thanks a lot.

```
@article{zhou2020blind,
  title={Blind quality assessment for image superresolution using deep two-stream convolutional networks},
  author={Zhou, Wei and Jiang, Qiuping and Wang, Yuwang and Chen, Zhibo and Li, Weiping},
  journal={Information Sciences},
  year={2020},
  publisher={Elsevier}
}
```


