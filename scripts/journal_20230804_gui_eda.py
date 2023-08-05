from startup import np, pd, plt, sns
import ipywidgets as widgets
from ipywidgets import interact, interactive
import os
from IPython.display import display


def get_csv_reader(source_dir):
    return {
        n.replace('.csv', ''):
            pd.read_csv(os.path.join(source_dir, n)) for n in os.listdir(source_dir)
    }


def interact_filter(df, x='length_cm', y='mass_g', hue='species'):
    plot_funs = ['describe', 'display', 'scatter', 'regplot', 'displot_x', 'displot_y', 'displot_xy']

    def plot(df_f, plot_fun):
        if plot_fun == 'describe':
            display(df_f.describe())
        elif plot_fun == 'display':
            display(df_f)
        elif plot_fun == 'scatter':
            sns.scatterplot(df_f, x=x, y=y, hue=hue)
        elif plot_fun == 'regplot':
            g = sns.FacetGrid(data=df_f, hue=hue)
            g.map(sns.regplot, x, y)
        elif plot_fun == 'displot_x':
            sns.displot(df_f, x=x, hue=hue)
        elif plot_fun == 'displot_y':
            sns.displot(df_f, y=y, hue=hue)
        elif plot_fun == 'displot_xy':
            sns.displot(df_f, x=x, y=y, hue=hue)

    def inner(hue_val, plot_fun):
        df_f = df.query(f'{hue} == "{hue_val}"') if hue_val != '' else df
        plot(df_f, plot_fun)

    hue_vals = [''] + df[hue].unique().tolist()
    return interact(inner, hue_val=hue_vals, plot_fun=plot_funs)

