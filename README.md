The text feature and pretrained model will be uploaded later.
## Introduction
This is Graph Structured Network for Image-Text Matching, source code of [GSMN](https://arxiv.org/abs/2004.00277) ([project page](https://github.com/CrossmodalGroup/GSMN)). The paper is accepted by CVPR2020. It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)


## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The word ids for each sentence is precomputed, and can be downloaded from [here](https://drive.google.com/open?id=1IoL1eJDQlaLDCub6zsmjDpAJDz7LjW59) (for Flickr30K and MSCOCO) 

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --model_name "$MODEL_PATH"
```

Arguments used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=20 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=20 --num_epoches=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 `|

For MSCOCO:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=10 --num_epoches=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=10 --num_epoches=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 `|

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

## Reference

If you found this code useful, please cite the following paper:
```
@article{liu2020graph,
  title={Graph Structured Network for Image-Text Matching},
  author={Liu, Chunxiao and Mao, Zhendong and Zhang, Tianzhu and Xie, Hongtao and Wang, Bin and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2004.00277},
  year={2020}
}
```

