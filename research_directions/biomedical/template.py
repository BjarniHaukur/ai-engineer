from datasets import load_dataset
import pytorch_lightning as pl
dataset = load_dataset('scikit-learn/breast-cancer-wisconsin')
data_train = dataset['train']