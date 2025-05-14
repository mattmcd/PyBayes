import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
# For JAX default to CPU to allow use of linalg
# os.environ['JAX_PLATFORMS'] = 'cpu,METAL'   # Actually, do this in PyCharm console settings