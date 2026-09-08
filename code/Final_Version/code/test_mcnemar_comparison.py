import unittest
import numpy as np
import pandas as pd
from mcnemar_comparison import compare_predictions


class McNemarTests(unittest.TestCase):
    def test_known_exact_and_holm_values(self):
        result = compare_predictions(np.zeros(10), {
            'A': np.zeros(10), 'B': np.ones(10), 'C': np.ones(10)})
        np.testing.assert_allclose(result.P_exact, [2 / 1024, 2 / 1024, 1])
        np.testing.assert_allclose(result.P_Holm, [6 / 1024, 6 / 1024, 1])
        self.assertEqual(result.Significant_Holm.tolist(), [True, True, False])
        counts = result[['Both_correct', 'A_correct_B_wrong',
                         'A_wrong_B_correct', 'Both_wrong']].sum(axis=1)
        self.assertTrue((counts == 10).all())

    def test_balanced_discordance(self):
        row = compare_predictions([0, 0, 0, 0], {
            'A': [0, 0, 1, 1], 'B': [0, 1, 0, 1]}).iloc[0]
        self.assertEqual(row.P_exact, 1)
        self.assertEqual(row.Accuracy_difference_A_minus_B, 0)
        self.assertEqual(row.Both_wrong, 1)

    def test_rejects_unpaired_inputs(self):
        with self.assertRaises(ValueError):
            compare_predictions([0, 1], {'A': [0], 'B': [0, 1]})
        with self.assertRaises(ValueError):
            compare_predictions(pd.Series([0, 1], index=[1, 2]), {
                'A': pd.Series([0, 1], index=[2, 1]), 'B': [0, 1]})


if __name__ == '__main__':
    unittest.main()
