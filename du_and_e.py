import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def explore_dataset(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    