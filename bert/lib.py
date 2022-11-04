"""------------Start Use predictText.py file--------------------"""
# import transformers
# from transformers import BertModel, BertJapaneseTokenizer, AdamW, get_linear_schedule_with_warmup
# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# from collections import defaultdict
# from textwrap import wrap
"""------------End Use predictText.py file--------------------"""

"""------------Start Use classification.py file--------------------"""
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import torch
import spacy
import ja_core_news_sm
"""------------End Use classification.py file--------------------"""

import numpy as np
import pandas as pd