from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RUN_PATH = "model_best.pth.tar"
DATA_PATH = "/media/ubuntu/data/chunxiao/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall",fold5=True)
