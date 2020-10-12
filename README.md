## Introduction
This is Graph Structured Network for Image-Text Matching, source code of [GSMN](https://arxiv.org/abs/2004.00277) ([project page](https://github.com/CrossmodalGroup/GSMN)). The paper is accepted by CVPR2020. It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Pretrained model
If you don't want to train from scratch, you can download the pretrained GSMN model from [here](https://drive.google.com/file/d/1kEi92w49Et5D2WVOv-Lc52HcpF2SPNNF/view?usp=sharing)(for Flickr30K dense model) and [here](https://drive.google.com/file/d/1vTPDToCJNLPU80K0ISXRmpSzys6-MjBT/view?usp=sharing)(for Flickr30K sparse model). The performance of this pretrained single model is as follows, in which some Recall@1 values are even better than results produced by our paper:
```bash
GSMN-dense:
rsum: 481.4
Average i2t Recall: 87.0
Image to text: 74.4 91.1 95.4 1.0 3.4
Average t2i Recall: 73.5
Text to image: 54.1 79.9 86.5 1.0 9.4

GSMN-sparse:
rsum: 476.8
Average i2t Recall: 86.5
Image to text: 72.8 91.0 95.8 1.0 4.0
Average t2i Recall: 72.4
Text to image: 52.8 78.8 85.6 1.0 10.1
```


## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). The text feature, image bounding box and semantic dependency are precomputed, and can be downloaded from [here](https://drive.google.com/file/d/1ZVLIN7uSh3dqYAEldelyYF2ei9vicJvZ/view?usp=sharing) (for Flickr30K and MSCOCO) 

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --model_name "$MODEL_PATH" --bi_gru
```

Arguments used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=20 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=20 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 --is_sparse `| 

For MSCOCO:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=10 --num_epochs=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=10 --num_epochs=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 --is_sparse `|

## Evaluation

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```

To ensemble sparse model and dense model, specify the model_path in test_stack.py, and run
```bash
python test_stack.py
```

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{liu2020graph,
  title={Graph Structured Network for Image-Text Matching},
  author={Liu, Chunxiao and Mao, Zhendong and Zhang, Tianzhu and Xie, Hongtao and Wang, Bin and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10921--10930},
  year={2020}
}
```

