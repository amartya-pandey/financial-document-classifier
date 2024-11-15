# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn

class EDA:
    def __init__(self, data):
        self.data = data

    def visualize_data_distribution(self):
        seaborn.countplot(y='category', data=self.data)
        plt.title('Distribution of Article Categories')
        plt.show()

    def calculate_statistics(self):
        lengths = self.data['text'].apply(lambda x: len(x.split()))
        print(f'Mean length of articles: {lengths.mean()}')
        print(f'Median length of articles: {lengths.median()}')