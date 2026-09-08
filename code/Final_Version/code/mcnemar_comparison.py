"""Exact McNemar tests for fixed classifiers on independent held-out cases."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def compare_predictions(y_true, predictions, alpha=0.05):
    """Compare paired labels in identical case order, with Holm adjustment.

    Tests error rates, not F1/AUC; does not correct CV dependence.
    """
    truth = np.asarray(y_true)
    if truth.ndim != 1 or not len(truth) or pd.isna(truth).any():
        raise ValueError('Expected nonempty 1-D true labels without missing values.')
    if len(predictions) < 2 or not 0 < alpha < 1:
        raise ValueError('Require at least two models and 0 < alpha < 1.')
    correct = {}
    for name, labels in predictions.items():
        if isinstance(labels, pd.Series) and isinstance(y_true, pd.Series):
            if not labels.index.equals(y_true.index):
                raise ValueError('Prediction and label indexes must match.')
        labels = np.asarray(labels)
        if labels.shape != truth.shape or pd.isna(labels).any():
            raise ValueError('Require one nonmissing prediction per test case.')
        correct[name] = labels == truth
    rows = []
    for name_a, name_b in combinations(correct, 2):
        a, b = correct[name_a], correct[name_b]
        ab, ba = int(np.sum(a & ~b)), int(np.sum(~a & b))
        p = binomtest(ab, ab + ba, p=0.5, alternative='two-sided').pvalue if ab + ba else 1.0
        rows.append(dict(Model_A=name_a, Model_B=name_b, N_test=len(truth),
                         Both_correct=int(np.sum(a & b)), A_correct_B_wrong=ab,
                         A_wrong_B_correct=ba, Both_wrong=int(np.sum(~a & ~b)),
                         Discordant_pairs=ab + ba, Accuracy_A=float(a.mean()),
                         Accuracy_B=float(b.mean()),
                         Accuracy_difference_A_minus_B=(ab - ba) / len(truth), P_exact=p))
    result = pd.DataFrame(rows)
    pvalues = result['P_exact'].to_numpy()
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    result['P_Holm'] = adjusted
    result['Alpha'] = alpha
    result['Significant_Holm'] = adjusted <= alpha
    return result


def evaluate_mcnemar(trained_models, y_test, output_dir='.'):
    """Save all pairwise tests and the underlying paired predictions."""
    predictions = {}
    for name, (model, features) in trained_models.items():
        if not features.index.equals(y_test.index):
            raise ValueError('Test features and labels must have matching indexes.')
        predictions[name] = np.asarray(model.predict(features)).reshape(-1)
    result = compare_predictions(y_test, predictions)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y_true': y_test.to_numpy(), **predictions}, index=y_test.index).to_csv(
        destination / 'mcnemar_test_predictions.csv', index_label='Dataset_row_index')
    result.to_csv(destination / 'mcnemar_results.csv', index=False)
    print('\n--- Exact McNemar tests: held-out test predictions ---')
    print(result.to_string(index=False))
    print('Interpret P_Holm at alpha=0.05; these tests concern error rates, not ROC-AUC or F1.')
    return result
