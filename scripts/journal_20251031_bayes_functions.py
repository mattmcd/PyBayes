from collections import defaultdict
import itertools
import pandas as pd
from typing import Callable


def aggregate_data(df: pd.DataFrame, features: list[str], target: str, label_fun: Callable) -> pd.DataFrame:
    """
    Aggregates the provided dataframe by grouping based on a subset of
    features, and generates trial and success statistics for each group.
    The method applies a label function on the data to determine
    success conditions and subsequently aggregates data based on specified
    grouping criteria.

    :param df: The input DataFrame containing the data to be aggregated.
        It must include the columns specified in the features parameter.
    :param features: A list of column names in the DataFrame that will be
        used for grouping the data. The target column must also be included
        in this list.
    :param target: The primary column in `features` that identifies the
        specific feature column for evaluation of success conditions.
    :param label_fun: A callable function that determines the success condition
        for each row in the provided DataFrame. Returns a boolean-like result
        indicating success or failure for a given row.
    :return: A DataFrame with aggregated data, containing columns for the
        number of trials (`trials`) and the number of successes (`successes`)
        based on the provided grouping criteria and label function.
    """
    return df.loc[:, features].assign(
        label=label_fun
    ).groupby(
        [c for c in features if c != target]
    ).agg(
        trials=pd.NamedAgg(column='label', aggfunc='count'),
        successes=pd.NamedAgg(column='label', aggfunc='sum')
    )


def extract_sub_population(df, sub_pop):
    """Extract sub-population from multi-indexed dataframe
    Basically a generalization of df.xs(key, level) that allows for multiple levels

    :param df: multi-indexed dataframe with columns 'trials' and 'successes'
    :param sub_pop: dict of level names and values to extract
    :return: multi-indexed dataframe of sub-population with columns 'trials' and 'successes'
    """
    level_names = df.index.names
    index_factory = defaultdict(lambda: slice(None))

    for k, v in sub_pop.items():
        index_factory[k] = v

    pop_indexer = tuple([index_factory[c] for c in level_names])

    try:
        return df.loc[pop_indexer, :]
    except KeyError:
        # Not multi-indexed so not suitable for further sub-setting but
        # prevents bayes_rule from failing
        return pd.DataFrame({'trials': [0], 'successes': [0]}, )


def bayes_rule(df, df_s):
    """Apply Bayes rule to estimate probability of target given sub population

    :param df: dataframe of full population with columns 'trials' and 'successes'
    :param df_s: dataframe of sub population
    :return: tuple of probabilities
    """
    # Alternatively, could use beta distributions for these to get confidence intervals
    p_sub_pop = df_s.trials.sum() / df.trials.sum()
    p_target = df.successes.sum() / df.trials.sum()
    p_target_given_sub_pop = df_s.successes.sum() / df_s.trials.sum()
    p_sub_pop_given_target = p_target_given_sub_pop * p_sub_pop / p_target
    # Or more simply: p_sub_pop_given_target = df_s.successes.sum()/df.successes.sum()
    # i.e. fraction of successes in sub-population as a fraction of successes in full population
    return p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target


def long_summary(pop_slice, target, p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target):
    """Summary of sub-population probabilities including level values in description"""
    slice_str = ', '.join({k: f'{k}={" or ".join(v)}' for k, v in pop_slice.items()}.values())
    return f'{pop_slice}\n' \
           f'P({target}) = {p_target:.3f}\n' \
           f'P({slice_str}) = {p_sub_pop:.3f}\n' \
           f'P({target} | {slice_str}) = {p_target_given_sub_pop:.3f}\n' \
           f'P({slice_str} | {target}) = {p_sub_pop_given_target:.3f}\n' \
           f'Odds ratio = {p_sub_pop_given_target / (1 - p_sub_pop_given_target):.3f}:1'


def brief_summary(pop_slice, target, p_target, p_sub_pop, p_target_given_sub_pop, p_sub_pop_given_target):
    """Summary of sub-population probabilities with level names in description"""
    return f'{pop_slice}\n' \
           f'P({target}) = {p_target:.3f}\n' \
           f'P({", ".join(pop_slice.keys())}) = {p_sub_pop:.3f}\n' \
           f'P({target} | {", ".join(pop_slice.keys())}) = {p_target_given_sub_pop:.3f}\n' \
           f'P({", ".join(pop_slice.keys())} | {target}) = {p_sub_pop_given_target:.3f}\n' \
           f'Odds ratio = {p_sub_pop_given_target / (1 - p_sub_pop_given_target):.3f}:1'


class BayesRule:
    def __init__(self, df, sub_pop, target_label='target', summary_type='long'):
        self.df = df
        self.sub_pop = sub_pop
        self.target_label = target_label
        self.df_s = extract_sub_population(self.df, self.sub_pop)
        self.p_target, self.p_sub_pop, self.p_target_given_sub_pop, self.p_sub_pop_given_target = \
            bayes_rule(self.df, self.df_s)
        self.summary_fun = long_summary if summary_type == 'long' else brief_summary

    def __repr__(self):
        return self.summary_fun(
            self.sub_pop, self.target_label,
            self.p_target, self.p_sub_pop,
            self.p_target_given_sub_pop, self.p_sub_pop_given_target
        )


def marginalize(df, groups):
    """Marginalize over all levels except those in groups

    :param df: multi-indexed dataframe with columns 'trials' and 'successes'
    :param groups: list of levels to keep
    :return: dataframe of marginalized probabilities
    """

    #
    sub_pops = [
        dict(zip(groups, g)) for g in
        itertools.product(*[df.index.unique(level=g).tolist() for g in groups])
    ]
    df_marginal_probs = pd.DataFrame(
        [sp | {'prob': BayesRule(df, sp).p_sub_pop_given_target} for sp in sub_pops]
    )
    return df_marginal_probs  # .fillna(0)  # leaving in NaN makes excluded groups more obvious
