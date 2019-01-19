import functools
import io
import logging
import time
import warnings

import numpy
import pandas
from sqlalchemy.orm import sessionmaker

from . import metrics
from .utils import db_retry, sort_predictions_and_labels


def generate_binary_at_x(test_predictions, x_value, unit="top_n"):
    """Generate subset of predictions based on top% or absolute

    Args:
        test_predictions (list) A list of predictions, sorted by risk desc
        x_value (int) The percentile or absolute value desired
        unit (string, default 'top_n') The subsetting method desired,
            either percentile or top_n

    Returns: (list) The predictions subset
    """
    if unit == "percentile":
        cutoff_index = int(len(test_predictions) * (x_value / 100.00))
    else:
        cutoff_index = x_value
    test_predictions_binary = [
        1 if x < cutoff_index else 0 for x in range(len(test_predictions))
    ]
    return test_predictions_binary


class ModelEvaluator(object):
    """An object that can score models based on its known metrics"""

    """Available metric calculation functions

    Each value is expected to be a function that takes in the following params
    (predictions_proba, predictions_binary, labels, parameters)
    and return a numeric score
    """
    available_metrics = {
        "precision@": metrics.precision,
        "recall@": metrics.recall,
        "fbeta@": metrics.fbeta,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "roc_auc": metrics.roc_auc,
        "average precision score": metrics.avg_precision,
        "true positives@": metrics.true_positives,
        "true negatives@": metrics.true_negatives,
        "false positives@": metrics.false_positives,
        "false negatives@": metrics.false_negatives,
        "fpr@": metrics.fpr,
    }

    def __init__(
        self,
        testing_metric_groups,
        training_metric_groups,
        db_engine,
        sort_seed=None,
        custom_metrics=None,
    ):
        """
        Args:
            testing_metric_groups (list) A list of groups of metric/configurations
                to use for evaluating all given models

                Each entry is a dict, with a list of metrics, and potentially
                    thresholds and parameter lists. Each metric is expected to
                    be a key in self.available_metrics

                Examples:

                testing_metric_groups = [{
                    'metrics': ['precision@', 'recall@'],
                    'thresholds': {
                        'percentiles': [5.0, 10.0],
                        'top_n': [5, 10]
                    }
                }, {
                    'metrics': ['f1'],
                }, {
                    'metrics': ['fbeta@'],
                    'parameters': [{'beta': 0.75}, {'beta': 1.25}]
                }]
            training_metric_groups (list) metrics to be calculated on training set,
                in the same form as testing_metric_groups
            db_engine (sqlalchemy.engine)
            sort_seed (int) the seed to set in random.set_seed() to break ties
                when sorting predictions
            custom_metrics (dict) Functions to generate metrics
                not available by default
                Each function is expected take in the following params:
                (predictions_proba, predictions_binary, labels, parameters)
                and return a numeric score
        """
        self.testing_metric_groups = testing_metric_groups
        self.training_metric_groups = training_metric_groups
        self.db_engine = db_engine
        self.sort_seed = sort_seed or int(time.time())
        if custom_metrics:
            self._validate_metrics(custom_metrics)
            self.available_metrics.update(custom_metrics)

    @property
    def sessionmaker(self):
        return sessionmaker(bind=self.db_engine)

    def _subset_labels_and_predictions(
        self,
        matrix_store,
        predictions_proba,
        subset_query,
        model_id,
    ):
        """Runs the subset query and returns only the predictions and labels
        relevant for the subset.

        Args:
            matrix_store (catwalk.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata
            predictions_proba (list) Probability predictions
            subset_query (str) A query against the predictions table that
                produces two columns, entity_id and as_of_date, with
                placeholders for the results_schema name, the list of
                as_of_dates in the matrix, and the model_id
            model_id (int) the id of the model being evaluated

        Returns: the labels (pandas.Series) and predictions (numpy.array) 
        """
        logging.info("Subsetting labels and predictions")
        labels = matrix_store.labels()
        logging.info(labels)
        indexed_predictions = pandas.Series(predictions_proba, index=labels.index)
        
        index_subset = self._query_to_pandas_index(
            subset_query.format(
                model_id=model_id,
                as_of_dates=[d.strftime("%Y-%m-%d %H:%M:%S.%f") for d in matrix_store.as_of_dates],
                results_schema=matrix_store.matrix_type.prediction_obj.__table_args__["schema"],
            ),
            index_columns=["entity_id", "as_of_date"],
        )

        labels_subset = labels.loc[index_subset]
        predictions_subset = indexed_predictions.loc[index_subset].values

        logging.debug(f"len(labels_subset) entities in subset out of len(labels) in matrix.")

        return (labels_subset, predictions_subset)
    
    def _query_to_pandas_index(self, query_string, index_columns):
        """Given a query, create a pandas.Index or MultiIndex object containing
        the given columns.

        Args:
            query_string (str) query to send
            index_columns (list) list of columns to create the index from

        Returns: (pandas.Index or pandas.MultiIndex) the requested index
        """
        copy_sql = f"COPY ({query_string}) TO STDOUT WITH CSV HEADER"
        conn = self.db_engine.raw_connection()
        cur = conn.cursor()
        out = io.StringIO()
        logging.debug(f"Running query {copy_sql} to get subset")
        cur.copy_expert(copy_sql, out)
        out.seek(0)
        df = pandas.read_csv(out, parse_dates=["as_of_date"], index_col=index_columns)
        
        return df.index

    def _validate_metrics(self, custom_metrics):
        for name, met in custom_metrics.items():
            if not hasattr(met, "greater_is_better"):
                raise ValueError(
                    "Custom metric {} missing greater_is_better "
                    "attribute".format(name)
                )
            elif met.greater_is_better not in (True, False):
                raise ValueError(
                    "For custom metric {} greater_is_better must be "
                    "boolean True or False".format(name)
                )

    def _build_parameter_string(
        self,
        threshold_unit,
        threshold_value,
        parameter_combination,
        threshold_specified_by_user,
    ):
        """Encode the metric parameters and threshold into a short, human-parseable string

        Examples are: '100_abs', '5_pct'

        Args:
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            parameter_combination (dict) The non-threshold parameter keys and values used
                Usually this will be empty, but an example would be {'beta': 0.25}

        Returns: (string) A short, human-parseable string
        """
        full_params = parameter_combination.copy()
        if threshold_specified_by_user:
            short_threshold_unit = "pct" if threshold_unit == "percentile" else "abs"
            full_params[short_threshold_unit] = threshold_value
        parameter_string = "/".join(
            ["{}_{}".format(val, key) for key, val in full_params.items()]
        )
        return parameter_string

    def _filter_nan_labels(self, predicted_classes, labels):
        """Filter missing labels and their corresponding predictions

        Args:
            predicted_classes (list) Predicted binary classes, of same length as labels
            labels (list) Labels, maybe containing NaNs

        Returns: (tuple) Copies of the input lists, with NaN labels removed
        """
        labels = numpy.array(labels)
        predicted_classes = numpy.array(predicted_classes)
        nan_mask = numpy.isfinite(labels)
        return ((predicted_classes[nan_mask]).tolist(), (labels[nan_mask]).tolist())

    def _evaluations_for_threshold(
        self,
        metrics,
        parameters,
        predictions_proba,
        labels,
        evaluation_table_obj,
        threshold_unit,
        threshold_value,
        threshold_specified_by_user=True,
    ):
        """Generate evaluations for a given threshold in a metric group,
        and create ORM objects to hold them

        Args:
            metrics (list) names of metric to compute
            parameters (list) dicts holding parameters to pass to metrics
            predictions_proba (list) Probability predictions
            labels (list) True labels (may have NaNs)
            threshold_unit (string) the type of threshold, either 'percentile' or 'top_n'
            threshold_value (int) the numeric threshold,
            threshold_specified_by_user (bool) Whether or not there was any threshold
                specified by the user. Defaults to True
            evaluation_table_obj (schema.TestEvaluation, TrainEvaluation,
                TestSubsetEvaluation, or TrainSubsetEvaluation) specifies to
                which table to add the evaluations

        Returns: (list) results_schema.TrainEvaluation or TestEvaluation objects
        Raises: UnknownMetricError if a given metric is not present in
            self.available_metrics
        """

        # using threshold configuration, convert probabilities to predicted classes
        predicted_classes = generate_binary_at_x(
            predictions_proba, threshold_value, unit=threshold_unit
        )
        # filter out null labels
        predicted_classes_with_labels, present_labels = self._filter_nan_labels(
            predicted_classes, labels
        )
        num_labeled_examples = len(present_labels)
        num_labeled_above_threshold = predicted_classes_with_labels.count(1)
        num_positive_labels = present_labels.count(1)
        evaluations = []
        for metric in metrics:
            if metric not in self.available_metrics:
                raise metrics.UnknownMetricError()

            for parameter_combination in parameters:
                try:
                    value = self.available_metrics[metric](
                        predictions_proba,
                        predicted_classes_with_labels,
                        present_labels,
                        parameter_combination,
                    )
                except:
                    warnings.warn(
                        f"{metric} not defined for parameter "
                        "{parameter_combination}. Inserting NULL for value.",
                        RuntimeWarning
                    )
                    value = None

                # convert the thresholds/parameters into something
                # more readable
                parameter_string = self._build_parameter_string(
                    threshold_unit=threshold_unit,
                    threshold_value=threshold_value,
                    parameter_combination=parameter_combination,
                    threshold_specified_by_user=threshold_specified_by_user,
                )

                logging.info(
                    "%s for %s%s, labeled examples %s "
                    "above threshold %s, positive labels %s, value %s",
                    evaluation_table_obj,
                    metric,
                    parameter_string,
                    num_labeled_examples,
                    num_labeled_above_threshold,
                    num_positive_labels,
                    value,
                )
                evaluations.append(
                    evaluation_table_obj(
                        metric=metric,
                        parameter=parameter_string,
                        value=value,
                        num_labeled_examples=num_labeled_examples,
                        num_labeled_above_threshold=num_labeled_above_threshold,
                        num_positive_labels=num_positive_labels,
                        sort_seed=self.sort_seed,
                    )
                )
        return evaluations

    def _evaluations_for_group(
        self,
        group,
        predictions_proba_sorted,
        labels_sorted,
        evaluation_table_obj,
    ):
        """Generate evaluations for a given metric group, and create ORM objects to hold them

        Args:
            group (dict) A configuration dictionary for the group.
                Should contain the key 'metrics', and optionally 'parameters' or 'thresholds'
            predictions_proba (list) Probability predictions
            labels (list) True labels (may have NaNs)
            evaluation_table_obj (schema.TestEvaluation, TrainEvaluation,
                TestSubsetEvaluation, or TrainSubsetEvaluation) specifies to
                which table to add the evaluations
        
        Returns: (list) results_schema.Evaluation objects
        """
        logging.info("Creating evaluations for metric group %s", group)
        parameters = group.get("parameters", [{}])
        generate_evaluations = functools.partial(
            self._evaluations_for_threshold,
            metrics=group["metrics"],
            parameters=parameters,
            predictions_proba=predictions_proba_sorted,
            labels=labels_sorted,
            evaluation_table_obj=evaluation_table_obj,
        )
        evaluations = []
        if "thresholds" not in group:
            logging.info(
                "Not a thresholded group, generating evaluation based on all predictions"
            )
            evaluations = evaluations + generate_evaluations(
                threshold_unit="percentile",
                threshold_value=100,
                threshold_specified_by_user=False,
            )

        for pct_thresh in group.get("thresholds", {}).get("percentiles", []):
            logging.info("Processing percent threshold %s", pct_thresh)
            evaluations = evaluations + generate_evaluations(
                threshold_unit="percentile", threshold_value=pct_thresh
            )

        for abs_thresh in group.get("thresholds", {}).get("top_n", []):
            logging.info("Processing absolute threshold %s", abs_thresh)
            evaluations = evaluations + generate_evaluations(
                threshold_unit="top_n", threshold_value=abs_thresh
            )
        return evaluations

    def needs_evaluations(self, matrix_store, model_id, subset_hash=''):
        """Returns whether or not all the configured metrics are present in the
        database for the given matrix and model.

        Args:
            matrix_store (triage.component.catwalk.storage.MatrixStore)
            model_id (int) A model id
            subset_hash (str) An identifier for the subset to be evaluated

        Returns:
            (bool) whether or not this matrix and model are missing any evaluations in the db
        """

        # assemble a list of evaluation objects from the config
        # by running the evaluation code with an empty list of predictions and labels
        eval_obj = matrix_store.matrix_type.evaluation_obj
        matrix_type = matrix_store.matrix_type
        if matrix_type.is_test:
            metric_groups_to_compute = self.testing_metric_groups
        else:
            metric_groups_to_compute = self.training_metric_groups
        evaluation_objects_from_config = [
            item
            for group in metric_groups_to_compute
            for item in self._evaluations_for_group(group, [], [], eval_obj)
        ]

        # assemble a list of evaluation objects from the database
        # by querying the unique metrics and parameters relevant to the passed-in matrix
        session = self.sessionmaker()
        evaluation_objects_in_db = session.query(eval_obj).filter_by(
            model_id=model_id,
            evaluation_start_time=matrix_store.as_of_dates[0],
            evaluation_end_time=matrix_store.as_of_dates[-1],
            as_of_date_frequency=matrix_store.metadata["as_of_date_frequency"],
            subset_hash=subset_hash,
        ).distinct(eval_obj.metric, eval_obj.parameter).all()

        # The list of needed metrics and parameters are all the unique metric/params from the config
        # not present in the unique metric/params from the db
        needed = bool(
            {(obj.metric, obj.parameter) for obj in evaluation_objects_from_config} -
            {(obj.metric, obj.parameter) for obj in evaluation_objects_in_db}
        )
        session.close()
        return needed

    def evaluate(
        self,
        predictions_proba,
        matrix_store,
        model_id,
        subset=None,
    ):
        """Evaluate a model based on predictions, and save the results

        Args:
            predictions_proba (numpy.array) List of prediction probabilities
            matrix_store (catwalk.storage.MatrixStore) a wrapper for the
                prediction matrix and metadata
            model_id (int) The database identifier of the model
            subset (dict) A dictionary containing a predictions query and a
                name for the subset to evaluate on, if any
        """
        evaluation_table_obj = matrix_store.matrix_type.evaluation_obj
        
        # If we are evaluating on a subset, we want to get just the labels and
        # predictions for the included entity-date pairs
        if subset:
            labels, predictions_proba = self._subset_labels_and_predictions(
                    matrix_store,
                    predictions_proba,
                    subset["query"],
                    model_id,
            )
            subset_hash = subset["hash"]
        else:
            labels = matrix_store.labels()
            subset_hash = '' 
        
        matrix_type = matrix_store.matrix_type.string_name
        evaluation_start_time = matrix_store.as_of_dates[0]
        evaluation_end_time = matrix_store.as_of_dates[-1]
        as_of_date_frequency = matrix_store.metadata["as_of_date_frequency"]

        logging.info(
            "Generating evaluations for model id %s, evaluation range %s-%s, "
            "as_of_date frequency %s, subset %s",
            model_id,
            evaluation_start_time,
            evaluation_end_time,
            as_of_date_frequency,
            subset
        )

        predictions_proba_sorted, labels_sorted = sort_predictions_and_labels(
            predictions_proba, labels, self.sort_seed
        )
        evaluations = []
        matrix_type = matrix_store.matrix_type
        if matrix_type.is_test:
            metric_groups_to_compute = self.testing_metric_groups
        else:
            metric_groups_to_compute = self.training_metric_groups
        for group in metric_groups_to_compute:
            evaluations = evaluations + self._evaluations_for_group(
                group,
                predictions_proba_sorted,
                labels_sorted,
                evaluation_table_obj
            )

        logging.info(
            "Writing metrics to db: %s table for subset %s",
            matrix_type,
            subset
        )
        self._write_to_db(
            model_id,
            subset_hash,
            evaluation_start_time,
            evaluation_end_time,
            as_of_date_frequency,
            evaluations,
            evaluation_table_obj,
            subset_hash,
        )
        logging.info(
            "Done writing metrics to db: %s table for subset %s",
            matrix_type,
            subset
        )
        logging.info(
            "Done writing metrics to db: %s table for subset %s",
            matrix_type,
            subset
        )

    @db_retry
    def _write_to_db(
        self,
        model_id,
        subset_hash,
        evaluation_start_time,
        evaluation_end_time,
        as_of_date_frequency,
        evaluations,
        evaluation_table_obj,
        subset_hash=None,
    ):
        """Write evaluation objects to the database

        Binds the model_id as as_of_date to the given ORM objects
        and writes them to the database

        Args:
            model_id (int) primary key of the model
            evaluation_start_time (pandas._libs.tslibs.timestamps.Timestamp)
                first as_of_date included in the evaluation period
            evaluation_end_time (pandas._libs.tslibs.timestamps.Timestamp) last
                as_of_date included in the evaluation period
            as_of_date_frequency (str) the frequency with which as_of_dates
                occur between the evaluation_start_time and evaluation_end_time
            evaluations (list) results_schema.TestEvaluation, TrainEvaluation,
                TestSubsetEvaluation, or TrainSubsetEvaluation objects
            evaluation_table_obj (schema.TestEvaluation, TrainEvaluation,
                TestSubsetEvaluation, or TrainSubsetEvaluation) specifies to
                which table to add the evaluations
            subset_hash (str) the hash of the subset, if any, that the
                evaluation is made on
        """
        session = self.sessionmaker()

        session.query(evaluation_table_obj).filter_by(
            model_id=model_id,
            evaluation_start_time=evaluation_start_time,
            evaluation_end_time=evaluation_end_time,
            as_of_date_frequency=as_of_date_frequency,
            subset_hash=subset_hash
        ).delete()

        for evaluation in evaluations:
            evaluation.model_id = model_id
            evaluation.as_of_date_frequency = as_of_date_frequency
            evaluation.subset_hash = subset_hash
            evaluation.evaluation_start_time = evaluation_start_time
            evaluation.evaluation_end_time = evaluation_end_time
            evaluation.as_of_date_frequency = as_of_date_frequency
            if subset_hash is not None:
                evaluation.subset_hash = subset_hash
            session.add(evaluation)
        session.commit()
        session.close()
