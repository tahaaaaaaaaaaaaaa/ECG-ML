import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import wfdb

print("Environment successfully loaded.\n")

print(f"Python      : {sys.version.split()[0]}")
print(f"Interpreter : {sys.executable}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"SciPy       : {scipy.__version__}")
print(f"WFDB        : {wfdb.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"PyTorch     : {torch.__version__}")
print(f"Joblib      : {joblib.__version__}")

print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CPU threads   : {torch.get_num_threads()}")
