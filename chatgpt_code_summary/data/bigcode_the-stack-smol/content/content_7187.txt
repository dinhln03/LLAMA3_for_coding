# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import print_function

from calicoml.core.metrics import ppv, npv, ROC
from calicoml.core.metrics import compute_averaged_metrics, accuracy_from_confusion_matrix, ConditionalMeansSelector
from calicoml.core.metrics import f_pearson
from calicoml.core.serialization.model import roc_auc_function

import numpy as np
import nose
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr


def test_ppv():
    """Verifies correctness of the PPV calculation"""
    nose.tools.eq_(ppv([1], [1]), 1.0)
    nose.tools.eq_(ppv([1, 1], [1, 0]), 1.0)
    nose.tools.eq_(ppv([1, 0, 0, 1], [1, 1, 1, 1]), 0.5)
    nose.tools.eq_(ppv([1, 0, 0, 1], [0, 1, 1, 0]), 0.0)
    nose.tools.eq_(ppv([1, 0, 0, 1], [1, 1, 0, 1]), 2.0 / 3)

    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), 1.0)
    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]), 0.8)
    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]), 0.6)
    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]), 0.4)
    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]), 0.2)
    nose.tools.eq_(ppv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), 0.0)

    # Bad values should fail
    nose.tools.assert_raises(AssertionError, lambda: ppv([1, 0, 1], [1, 0]))
    nose.tools.assert_raises(AssertionError, lambda: ppv([1, 0], [1, 0, 1]))
    nose.tools.assert_raises(AssertionError, lambda: ppv([1, 0, 2], [1, 0, 1]))
    nose.tools.assert_raises(AssertionError, lambda: ppv([1, 0, 1], [1, 0, 2]))


def test_npv():
    """Verifies correctness of the NPV calculation"""
    nose.tools.eq_(npv([0], [0]), 1.0)
    nose.tools.eq_(npv([0, 0], [0, 1]), 1.0)
    nose.tools.eq_(npv([0, 1], [0, 0]), 0.5)
    nose.tools.eq_(npv([1, 0, 0, 1], [0, 0, 0, 0]), 0.5)
    nose.tools.eq_(npv([1, 0, 0, 1], [0, 1, 1, 0]), 0.0)
    nose.tools.eq_(npv([0, 1, 1, 0], [0, 0, 1, 0]), 2.0 / 3)

    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), 1.0)
    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]), 0.8)
    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]), 0.6)
    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]), 0.4)
    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]), 0.2)
    nose.tools.eq_(npv([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), 0.0)

    # Bad values should fail
    nose.tools.assert_raises(AssertionError, lambda: npv([1, 0, 1], [1, 0]))
    nose.tools.assert_raises(AssertionError, lambda: npv([1, 0], [1, 0, 1]))
    nose.tools.assert_raises(AssertionError, lambda: npv([1, 0, 2], [1, 0, 1]))
    nose.tools.assert_raises(AssertionError, lambda: npv([1, 0, 1], [1, 0, 2]))


def test_roc():
    """Tests the ROC class"""
    def checkme(y_true, y_pred, expected_auc):
        """Tests the ROC for a single set of predictions. Mostly sanity checks since all the computation is done
        by scikit, which we assume is correct"""
        roc = ROC.from_scores(y_true, y_pred)
        nose.tools.assert_almost_equal(roc.auc, expected_auc)
        nose.tools.ok_(all(0 <= fpr_val <= 1 for fpr_val in roc.fpr))
        nose.tools.ok_(all(0 <= tpr_val <= 1 for tpr_val in roc.tpr))

        nose.tools.assert_list_equal(list(roc.dataframe['tpr']), list(roc.tpr))
        nose.tools.assert_list_equal(list(roc.dataframe['thresholds']), list(roc.thresholds))

        for prop in ['fpr', 'tpr', 'thresholds']:
            nose.tools.assert_list_equal(list(roc.dataframe[prop]), list(getattr(roc, prop)))
            nose.tools.assert_greater_equal(len(roc.dataframe[prop]), 2)  # needs to have at least the two extremes

    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], 1.0
    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], 0.0
    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0], 0.5
    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0], 0.75


def test_auc_ci():
    """Validates the AUC confidence interval by comparing with R's pROC"""
    def checkme(y_true, y_pred):
        """Test utility"""
        roc = ROC.from_scores(y_true, y_pred)
        print(roc.auc_ci)
        np.testing.assert_allclose(roc.auc_ci.estimate, roc.auc, atol=0.01)

        proc = importr('pROC')
        r_ci_obj = proc.ci(proc.roc(FloatVector(y_true), FloatVector(y_pred), ci=True), method='bootstrap')
        r_ci_dict = dict(list(r_ci_obj.items()))
        np.testing.assert_allclose(r_ci_dict['2.5%'], roc.auc_ci.low, atol=0.02)
        np.testing.assert_allclose(r_ci_dict['97.5%'], roc.auc_ci.high, atol=0.02)

    np.random.seed(0xC0FFEE)
    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0] * 10, [1, 1, 1, 1, 0, 0, 0, 0] * 10
    yield checkme, [1, 1, 1, 1, 0, 0, 0, 0] * 10, [1, 0, 1, 0, 1, 0, 1, 0] * 10

    for _ in range(5):
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.normal(size=100)
        y_pred[y_true == 1] += np.abs(np.random.normal())
        yield checkme, y_true, y_pred


def test_compute_averaged_metrics():
    """ Tests compute_averaged_metrics function"""
    y_truth = [0, 1, 2, 0, 1, 2]
    scores1 = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    result1 = compute_averaged_metrics(y_truth, scores1, roc_auc_function)
    nose.tools.assert_almost_equal(1.0, result1, delta=1e-6)
    scores2 = [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
    result2 = compute_averaged_metrics(y_truth, scores2, roc_auc_function)
    nose.tools.assert_almost_equal(0.375, result2, delta=1e-6)


def test_pearson():
    """ Validate pearson correlation"""
    X = np.asarray([[1, 2], [-2, 8], [3, 5]])
    y = np.asarray([-1, -2, 0])
    rs_pearson, ps_pearson = f_pearson(X, y)
    nose.tools.assert_almost_equal(0.073186395040328034, ps_pearson[0], delta=1e-6)
    nose.tools.assert_almost_equal(0.66666666666666663, ps_pearson[1], delta=1e-6)
    nose.tools.assert_almost_equal(0.993399267799, rs_pearson[0], delta=1e-6)
    nose.tools.assert_almost_equal(-0.5, rs_pearson[1], delta=1e-6)


def test_accuracy_from_confusion_matrix():
    """ test accuracy computations from confusion matrix """
    y_truth = [0, 1, 2, 0, 1, 2]
    y_score = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]
    y_pred = [0, 1, 2, 1, 1, 0]
    computed_confusion_matrix = confusion_matrix(y_truth, y_pred)
    accuracy = accuracy_from_confusion_matrix(y_truth, y_score, computed_confusion_matrix)
    nose.tools.assert_almost_equal(0.6666667, accuracy, delta=1e-6)


def test_conditional_means_selector():
    """ test ConditionalMeansSelector class """
    # first check means in reverse order
    cms = ConditionalMeansSelector(f_pearson)
    test_y = np.asarray([3, 2, 1, 0, 3, 2, 1, 0])
    test_x = np.asarray([[0, 3], [5, 2], [9, 1], [13, 0], [0, 3], [5, 2], [9, 1], [13, 0]])
    rs_cond_means, ps_cond_means = cms.selector_function(test_x, test_y)
    nose.tools.assert_almost_equal(1.0, rs_cond_means[0], delta=1e-6)
    nose.tools.assert_almost_equal(1.0, rs_cond_means[1], delta=1e-6)
    nose.tools.assert_almost_equal(0.0, ps_cond_means[0], delta=1e-6)
    nose.tools.assert_almost_equal(0.0, ps_cond_means[1], delta=1e-6)

    # check that direct call does not produce right result, do NOT use as code pattern !!!
    rs_cond_means_wrong, _ = f_pearson(test_x, test_y)
    nose.tools.assert_not_almost_equal(1.0, rs_cond_means_wrong[0], delta=1e-6)

    # check means in same order
    cms_pairwise = ConditionalMeansSelector(pearsonr, True)
    rs_cond_means_pw, ps_cond_means_pw = cms_pairwise.selector_function(test_x, test_y)
    nose.tools.assert_almost_equal(1.0, rs_cond_means_pw[0], delta=1e-6)
    nose.tools.assert_almost_equal(1.0, rs_cond_means_pw[1], delta=1e-6)
    nose.tools.assert_almost_equal(0.0, ps_cond_means_pw[0], delta=1e-6)
    nose.tools.assert_almost_equal(0.0, ps_cond_means_pw[1], delta=1e-6)
