## YOLO Pytorch (A Real Time Deep-Learning-based Object Detection)
In this repository, we are implementing the article [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) with [PYTORCH](http://pytorch.org/) framework. This project is under the progress. Pretrianed network will be released after final result.

For training, at root of the master directory, run below command in Termianl:
```
curl -O http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
curl -O http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
curl -O http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```
