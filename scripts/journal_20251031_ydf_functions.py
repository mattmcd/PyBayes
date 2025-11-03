import ydf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class ModelBasedAnalysis:
    def __init__(self, df_train, df_test, target):
        self.df_train = df_train
        self.df_test = df_test
        self.target = target
        self.model =  ydf.RandomForestLearner(label=target).train(df_train)
        self.evaluation = self.model.evaluate(df_test)

    def variable_importance(self):
        return pd.DataFrame(
            self.model.variable_importances()['INV_MEAN_MIN_DEPTH']
        ).rename(
            columns={0: 'importance', 1: 'feature'}
        ).set_index('feature')['importance']

    def __repr__(self):
        out_str = f'ROC AUC: {self.evaluation.characteristics[0].roc_auc:0.2f}'
        return out_str

    def plot(self):
        df_p = self.df_test.sample(frac=0.2)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), width_ratios=[0.4, 0.6])
        manifold = TSNE(n_components=2).fit_transform(self.model.distance(df_p, df_p))
        ax = axs[1]
        sns.scatterplot(x=manifold[:, 0], y=manifold[:, 1], hue=df_p[self.target], alpha=0.2, ax=ax)
        ax.set_title(
            f'ROC AUC: {self.evaluation.characteristics[0].roc_auc:0.2f}'
        )
        df = self.variable_importance()
        ax = axs[0]
        sns.barplot(
            df.sort_values(ascending=False),
            orient='h', ax=ax
        )
        ax.set_title('Variable Importance')
        plt.tight_layout()
        return fig
