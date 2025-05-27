import pandas as pd
import numpy as np
from dataclasses import dataclass
import numbers
import itertools
import functools
from abc import ABC, abstractmethod

from experiment_config import sanity_check_result_level_names, metric_types


class AbstractCriterion(ABC):
    @abstractmethod
    def check_criterion(self, selected_df, metric_name):
        pass


class SimpleCriterion(AbstractCriterion):
    """Checks if the value of a metric at a specific point fulfills a predicate.

    The point is defined by variable_to_check having the value value_to_check.
    value_to_check can be "min" or "max", which check at the minimum or 
    maximum value of variable_to_check.

    Input to check is given as a dataframe containing the following columns:
    - variable_to_check: Value describing the experiment setting that is checked, such as difference in real and synthetic means.
    - col_to_check: The metric value that is checked.

    For example, to check if the metric is close to zero at the right extreme
    of "syn_mean", use:
    SimpleCriterion("syn_mean", "max", check_close_to_zero, col_to_check="mean")
    """
    def __init__(self, variable_to_check, value_to_check, predicate, col_to_check="mean"):
        """Constructor of SimpleCriterion.

        Args:
            variable_to_check (str): Variable defining the point that is checked.
            value_to_check (real number or str): Value defining the point that is checked. Can be "min" or "max".
            predicate: Predicate to check.
            col_to_check (str, optional): Column of input dataframe containing the metric value. Defaults to "mean".

        Raises:
            ValueError: _description_
        """
        self.variable_to_check = variable_to_check
        self.value_to_check = value_to_check
        if not isinstance(value_to_check, numbers.Real):
            if value_to_check not in ["min", "max"]:
                raise ValueError("value_to_check must be a real number or 'min' or 'max', got {}".format(value_to_check))

        self.predicate = predicate
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        if self.value_to_check == "min":
            value_to_check = selected_df[self.variable_to_check].min()
        elif self.value_to_check == "max":
            value_to_check = selected_df[self.variable_to_check].max()
        else:
            value_to_check = self.value_to_check

        selected_value = selected_df[selected_df[self.variable_to_check] == value_to_check][self.col_to_check].item()
        return self.predicate(selected_value)


class CriterionSeparatorByMetricType(AbstractCriterion):
    """Meta-criterion that chooses the criterion to check based on the metric type.

    Metric types are determined by metric_name, and defined in experiment_config.py.
    """
    def __init__(self, criteria_by_metric_type):
        """Constructor of CriterionSeparatorByMetricType.

        Args:
            criteria_by_metric_type (dict): Dictionary mapping metric types to criteria.
        """
        self.criteria_by_metric_type = criteria_by_metric_type

    def check_criterion(self, selected_df, metric_name):
        metric_type = metric_types[metric_name]
        if metric_type not in self.criteria_by_metric_type.keys():
            return np.nan
        else:
            return self.criteria_by_metric_type[metric_type].check_criterion(selected_df, metric_name)


class CriterionSeparatorByIndexValue(AbstractCriterion):
    """Meta-criterion that chooses the criterion to check based on a value in the index of the dataframe to check.
    """
    def __init__(self, separator_variable, criteria_by_index_value):
        """Constructor of CriterionSeparatorByIndexValue.

        Args:
            separator_variable (str): Index variable of the dataframe defining the check.
            criteria_by_index_value (dict): Dictionary mapping values of the index variable to criteria.
        """
        self.separator_variable = separator_variable
        self.criteria_by_index_value = criteria_by_index_value

    def check_criterion(self, selected_df, metric_name):
        separator_values = selected_df[self.separator_variable]
        separator_value = separator_values.unique().item()
        return self.criteria_by_index_value[separator_value].check_criterion(selected_df, metric_name)


class BellShapeCriterion(AbstractCriterion):
    """Criterion that checks if the metric value is bell-shaped around a midpoint.

    variable_to_check is the x-axis of the bell, and col_to_check is the y-axis.
    """
    def __init__(self, variable_to_check, bell_midpoint, 
        midpoint_to_extreme_min_difference=0.2, midpoint_to_max_max_difference=0.1, 
        extreme_to_min_max_difference=0.1, col_to_check="mean"):
        """Constructor of BellShapeCriterion.

        Args:
            variable_to_check (str): x-axis of the bell.
            bell_midpoint (float): Midpoint of the bell shape.
            midpoint_to_extreme_min_difference (float, optional): Defaults to 0.2.
            midpoint_to_max_max_difference (float, optional): Defaults to 0.1.
            extreme_to_min_max_difference (float, optional): Defaults to 0.1.
            col_to_check (str, optional): y-axis of the bell. Defaults to "mean".
        """

        self.variable_to_check = variable_to_check
        self.bell_midpoint = bell_midpoint
        self.midpoint_to_extreme_min_difference = midpoint_to_extreme_min_difference
        self.midpoint_to_max_max_difference = midpoint_to_max_max_difference
        self.extreme_to_min_max_difference = extreme_to_min_max_difference
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        left_extreme_point = selected_df[self.variable_to_check].min()
        left_extreme_value = selected_df[selected_df[self.variable_to_check] == left_extreme_point][self.col_to_check].item()

        right_extreme_point = selected_df[self.variable_to_check].max()
        right_extreme_value = selected_df[selected_df[self.variable_to_check] == right_extreme_point][self.col_to_check].item()

        midpoint_value = selected_df[selected_df[self.variable_to_check] == self.bell_midpoint][self.col_to_check].item()
        max_value = selected_df[self.col_to_check].max()
        min_value = selected_df[self.col_to_check].min()

        midpoint_to_left_extreme_difference = midpoint_value - left_extreme_value
        midpoint_to_right_extreme_difference = midpoint_value - right_extreme_value
        midpoint_to_max_difference = max_value - midpoint_value
        left_extreme_to_min_difference = left_extreme_value - min_value
        right_extreme_to_min_difference = right_extreme_value - min_value

        return midpoint_to_left_extreme_difference >= self.midpoint_to_extreme_min_difference \
            and midpoint_to_right_extreme_difference >= self.midpoint_to_extreme_min_difference \
            and midpoint_to_max_difference <= self.midpoint_to_max_max_difference \
            and (left_extreme_to_min_difference <= self.extreme_to_min_max_difference \
                or right_extreme_to_min_difference <= self.extreme_to_min_max_difference)


class LowToHighShapeCriterion(AbstractCriterion):
    """Criterion that checks if the metric value is low at the left extreme and high at the right extreme.
    """
    def __init__(
        self, variable_to_check, left_extreme_to_right_extreme_min_difference=0.2,
        left_extreme_to_min_max_difference=0.1,
        right_extreme_to_max_max_difference=0.1,
        col_to_check="mean",
        ):
        """Constructor of LowToHighShapeCriterion.

        Args:
            variable_to_check (str): Variable defining the extremes.
            left_extreme_to_right_extreme_min_difference (float, optional): Defaults to 0.2.
            left_extreme_to_min_max_difference (float, optional): Defaults to 0.1.
            right_extreme_to_max_max_difference (float, optional): Defaults to 0.1.
            col_to_check (str, optional): Column containing values that are checked. Defaults to "mean".
        """
        
        self.variable_to_check = variable_to_check
        self.left_extreme_to_right_extreme_min_difference = left_extreme_to_right_extreme_min_difference
        self.left_extreme_to_min_max_difference = left_extreme_to_min_max_difference
        self.right_extreme_to_max_max_difference = right_extreme_to_max_max_difference
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        left_extreme_point = selected_df[self.variable_to_check].min()
        left_extreme_value = selected_df[selected_df[self.variable_to_check] == left_extreme_point][self.col_to_check].item()

        right_extreme_point = selected_df[self.variable_to_check].max()
        right_extreme_value = selected_df[selected_df[self.variable_to_check] == right_extreme_point][self.col_to_check].item()

        max_value = selected_df[self.col_to_check].max()
        min_value = selected_df[self.col_to_check].min()

        left_extreme_to_right_extreme_difference = right_extreme_value - left_extreme_value
        left_extreme_to_min_difference = left_extreme_value - min_value
        right_extreme_to_max_difference = max_value - right_extreme_value

        return left_extreme_to_right_extreme_difference >= self.left_extreme_to_right_extreme_min_difference \
            and left_extreme_to_min_difference <= self.left_extreme_to_min_max_difference \
            and right_extreme_to_max_difference <= self.right_extreme_to_max_max_difference \


class HighToLowShapeCriterion(AbstractCriterion):
    """Criterion that checks if the metric value is high at the left extreme and low at the right extreme.
    """
    def __init__(
        self, variable_to_check, left_extreme_to_right_extreme_min_difference=0.2,
        right_extreme_to_min_max_difference=0.1,
        left_extreme_to_max_max_difference=0.1,
        col_to_check="mean",
        ):
        """Constructor of HighToLowShapeCriterion.

        Args:
            variable_to_check (str): Variable defining the extremes.
            left_extreme_to_right_extreme_min_difference (float, optional): _description_. Defaults to 0.2.
            right_extreme_to_min_max_difference (float, optional): _description_. Defaults to 0.1.
            left_extreme_to_max_max_difference (float, optional): _description_. Defaults to 0.1.
            col_to_check (str, optional): Column containing values that are checked. Defaults to "mean".
        """
        
        self.variable_to_check = variable_to_check
        self.left_extreme_to_right_extreme_min_difference = left_extreme_to_right_extreme_min_difference
        self.right_extreme_to_min_max_difference = right_extreme_to_min_max_difference
        self.left_extreme_to_max_max_difference = left_extreme_to_max_max_difference
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        left_extreme_point = selected_df[self.variable_to_check].min()
        left_extreme_value = selected_df[selected_df[self.variable_to_check] == left_extreme_point][self.col_to_check].item()

        right_extreme_point = selected_df[self.variable_to_check].max()
        right_extreme_value = selected_df[selected_df[self.variable_to_check] == right_extreme_point][self.col_to_check].item()

        max_value = selected_df[self.col_to_check].max()
        min_value = selected_df[self.col_to_check].min()

        left_extreme_to_right_extreme_difference = left_extreme_value - right_extreme_value
        left_extreme_to_max_difference = max_value - left_extreme_value
        right_extreme_to_min_difference = right_extreme_value - min_value

        return left_extreme_to_right_extreme_difference >= self.left_extreme_to_right_extreme_min_difference \
            and left_extreme_to_max_difference <= self.left_extreme_to_max_max_difference \
            and right_extreme_to_min_difference <= self.right_extreme_to_min_max_difference \


class HighToLowShapeWithMidDropCriterion(AbstractCriterion):
    """Criterion that checks if the metric value is high at the left extreme and low at the right extreme,
    including a sufficient drop in the middle.
    """
    def __init__(
        self, variable_to_check, midpoint_quantile=0.5, left_extreme_to_midpoint_min_difference=0.1,
        left_extreme_to_right_extreme_min_difference=0.2,
        right_extreme_to_min_max_difference=0.1,
        left_extreme_to_max_max_difference=0.1,
        col_to_check="mean",
        ):
        """Constructor of HighToLowShapeWithMidDropCriterion.

        Args:
            variable_to_check (str): Variable defining the extremes.
            midpoint_quantile (float, optional): Defaults to 0.5.
            left_extreme_to_midpoint_min_difference (float, optional): Defaults to 0.1.
            left_extreme_to_right_extreme_min_difference (float, optional): Defaults to 0.2.
            right_extreme_to_min_max_difference (float, optional): Defaults to 0.1.
            left_extreme_to_max_max_difference (float, optional): Defaults to 0.1.
            col_to_check (str, optional): Column containing values that are checked. Defaults to "mean".
        """
        self.variable_to_check = variable_to_check
        self.midpoint_quantile = midpoint_quantile
        self.left_extreme_to_midpoint_min_difference = left_extreme_to_midpoint_min_difference
        self.high_to_low_shape = HighToLowShapeCriterion(
            variable_to_check, left_extreme_to_right_extreme_min_difference,
            right_extreme_to_min_max_difference, left_extreme_to_max_max_difference,
            col_to_check
        )
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        pass_high_to_low_shape = self.high_to_low_shape.check_criterion(selected_df, metric_name)
        midpoint = selected_df[self.variable_to_check].quantile(self.midpoint_quantile, interpolation="nearest")
        midpoint_value = selected_df[selected_df[self.variable_to_check] == midpoint][self.col_to_check].item()

        left_extreme_point = selected_df[self.variable_to_check].min()
        left_extreme_value = selected_df[selected_df[self.variable_to_check] == left_extreme_point][self.col_to_check].item()

        left_extreme_to_midpoint_difference = left_extreme_value - midpoint_value
        return pass_high_to_low_shape and left_extreme_to_midpoint_difference >= self.left_extreme_to_midpoint_min_difference


class HorizontalLineShapeCriterion(AbstractCriterion):
    """Criterion that checks if the metric value is close to a horizontal line.

    Horizontal line is defined by the maximum and minimum value of the metric
    being sufficiently close to each other.
    """
    def __init__(self, variable_to_check, max_difference=0.05, col_to_check="mean"):
        """Constructor of HorizontalLineShapeCriterion.

        Args:
            variable_to_check (str): x-axis of the horizontal line.
            max_difference (float, optional): Max difference between min and max values. Defaults to 0.05.
            col_to_check (str, optional): y-axis of the line. Defaults to "mean".
        """
        self.variable_to_check = variable_to_check
        self.max_difference = max_difference
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        max_value = selected_df[self.col_to_check].max()
        min_value = selected_df[self.col_to_check].min()

        difference = max_value - min_value 
        return difference <= self.max_difference


class ConvergingLineShapeCriterion(AbstractCriterion):
    """Criterion that checks if the metric value converges to a value, and is constant after that.
    """
    def __init__(self, variable_to_check, convergence_point_quantile, convergence_point_after_max_difference=0.05, col_to_check="mean"):
        """Contructor of ConvergingLineShapeCriterion.

        Args:
            variable_to_check (str): x-axis of the line.
            convergence_point_quantile (float): Quantile the line must converge by.
            convergence_point_after_max_difference (float, optional): Maximum difference after convergence. Defaults to 0.05.
            col_to_check (str, optional): y-axis of the line. Defaults to "mean".
        """
        self.variable_to_check = variable_to_check
        self.convergence_point_quantile = convergence_point_quantile
        self.convergence_point_after_max_difference = convergence_point_after_max_difference
        self.col_to_check = col_to_check

    def check_criterion(self, selected_df, metric_name):
        convergence_point_value = selected_df[self.variable_to_check].quantile(self.convergence_point_quantile)
        subset_after_convergence_point_df = selected_df[selected_df[self.variable_to_check] >= convergence_point_value]

        max_value_after_convergence_point = subset_after_convergence_point_df[self.col_to_check].max()
        min_value_after_convergence_point = subset_after_convergence_point_df[self.col_to_check].min()
        difference = max_value_after_convergence_point - min_value_after_convergence_point
        return difference <= self.convergence_point_after_max_difference


class DiversityHighLowShapeCriterion(AbstractCriterion):
    """Criterion that checks whether a diversity metric is a low metric of a high metric.
    """
    def __init__(self, variable_to_check, low_shape=None, low_shape_bell_midpoint=None, high_shape=None, col_to_check="mean"):
        """Constructor of DiversityHighLowShapeCriterion.

        low_shape is a BellShapeCriterion by default, and high_shape is a LowToHighShapeCriterion by default.
        Args:
            variable_to_check (str): x-axis of the shape
            low_shape (AbstractCriterion, optional): Criterion for low shapes. Defaults to None.
            low_shape_bell_midpoint (_type_, optional): Midpoint for default low shape. Defaults to None.
            high_shape (AbstractCriterion, optional): Criterion for high shapes. Defaults to None.
            col_to_check (str, optional): y-axis of the shape. Defaults to "mean".

        Raises:
            ValueError: _description_
        """
        self.variable_to_check = variable_to_check
        if low_shape is None:
            if low_shape_bell_midpoint is None:
                raise ValueError("low_shape_bell_midpoint is required if low shape is not given.")
            self.low_shape = BellShapeCriterion(variable_to_check, low_shape_bell_midpoint, col_to_check=col_to_check)
        else:
            self.low_shape = low_shape
        
        if high_shape is None:
            self.high_shape = LowToHighShapeCriterion(variable_to_check, col_to_check=col_to_check)
        else:
            self.high_shape = high_shape

    def check_criterion(self, selected_df, metric_name):
        pass_low_shape = self.low_shape.check_criterion(selected_df, metric_name)
        pass_high_shape = self.high_shape.check_criterion(selected_df, metric_name)

        if pass_low_shape and pass_high_shape:
            raise Exception("Passed both low and high criteria.") # Should not happen
        elif pass_low_shape:
            return "low"
        elif pass_high_shape:
            return "high"
        else:
            return False


@dataclass
class SanityCheckCriteria:
    index_variables: list
    criteria: dict

    def get_index_tuples_with_criteria(self, group_df):
        index_tuples = self.get_index_tuples(group_df)
        return [
            index_tuple + (criterion_type, criterion_name)
            for index_tuple in index_tuples
            for criterion_type, criteria in self.criteria.items()
            for criterion_name in criteria.keys() 
        ]

    def get_index_tuples(self, group_df):
        items_in_product = [
            group_df[index_variable].unique()
            for index_variable in self.index_variables
        ]
        return list(itertools.product(*items_in_product))

    def get_empty_pass_fail_df(self, group_df):
        metric_names = group_df.Metric.unique()
        index_tuples = self.get_index_tuples_with_criteria(group_df)
        index = pd.MultiIndex.from_tuples(index_tuples, names=self.index_variables + ["desiderata", "criterion"])
        return pd.DataFrame(index=index, columns=metric_names)

    def get_pass_fail_df(self, group_df):
        pass_fail_df = self.get_empty_pass_fail_df(group_df)
        metric_names = group_df.Metric.unique()

        for metric_name in metric_names:
            for index_tuple in self.get_index_tuples(group_df):
                selectors = [group_df.Metric == metric_name] + [
                    group_df[index_variable] == index_tuple[i]
                    for i, index_variable in enumerate(self.index_variables)
                ]
                and_selector = functools.reduce(lambda s1, s2: s1 & s2, selectors)
                selected_df = group_df[and_selector]

                for criterion_type, criteria_type in self.criteria.items():
                    for criterion_name, criterion in criteria_type.items():
                        did_pass = criterion.check_criterion(selected_df, metric_name)
                        pass_fail_df.loc[index_tuple + (criterion_type, criterion_name), metric_name] = did_pass

        return pass_fail_df


def check_close_to(target, max_difference=0.05):
    """Returns a predicate that checks if a value is close to a target value.

    Args:
        target (float): Target value.
        max_difference (float, optional): Max difference to target value. Defaults to 0.05.
    """
    def fun(v):
        return np.abs(v - target) <= max_difference

    return fun

def check_close_to_one(v):
    return 0.95 <= v <= 1.05

def check_close_to_zero(v):
    return -0.05 <= v <= 0.05

def check_wide_syn_distribution_diversity(v):
    """Checks for correct diversity metric value for wide synthetic distributions.

    High metrics should be close to 1, and low metrics should be close to 0.
    Args:
        v (float): Value to check.

    Returns:
        bool or str: "high" if high metric, "low" if low metric, False otherwise.
    """
    if check_close_to_one(v):
        return "high"
    elif check_close_to_zero(v):
        return "low"
    else:
        return False