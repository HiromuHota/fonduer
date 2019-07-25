import logging
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fonduer import init_logging
from fonduer.candidates.candidates import Candidate
from fonduer.learning.disc_models.logistic_regression import LogisticRegression

ABSTAIN = 0
FALSE = 1
TRUE = 2


def test_check_input(caplog):
    """Test if the input is a tuple."""
    init_logging(
        log_dir="log_folder",
        format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    disc_model = LogisticRegression()
    X = [Candidate(id=1, type="type")]
    F = [csr_matrix((len(X), 1), dtype=np.int8)]
    assert not disc_model._check_input((X))
    assert disc_model._check_input((X,))
    assert disc_model._check_input((X, F))


def test_preprocess_data(caplog):
    init_logging(
        log_dir="log_folder",
        format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    disc_model = LogisticRegression()
    disc_model._preprocess_data()


def test_predict(caplog):
    init_logging(log_dir=tempfile.gettempdir())
    # Create sample candidates and their features
    X = [
        Candidate(id=1, type="type"),
        Candidate(id=2, type="type"),
        Candidate(id=3, type="type"),
    ]
    F = csr_matrix((len(X), 1), dtype=np.int8)
    disc_model = LogisticRegression()

    # binary
    disc_model.cardinality = 2

    # mock the marginals so that 0.4 for FALSE and 0.6 for TRUE
    def mock_marginals(X):
        marginal = np.ones((len(X), 1)) * 0.4
        return np.concatenate((marginal, 1 - marginal), axis=1)

    disc_model.marginals = mock_marginals

    # marginal < b gives FALSE predictions.
    Y_pred = disc_model.predict((X, F), b=0.6, pos_label=TRUE)
    np.testing.assert_array_equal(Y_pred, np.array([FALSE, FALSE, FALSE]))

    # b <= marginal gives TRUE predictions.
    Y_pred = disc_model.predict((X, F), b=0.59, pos_label=TRUE)
    np.testing.assert_array_equal(Y_pred, np.array([TRUE, TRUE, TRUE]))

    # return_probs=True should return marginals too.
    _, Y_prob = disc_model.predict((X, F), b=0.59, pos_label=TRUE, return_probs=True)
    np.testing.assert_array_equal(Y_prob, mock_marginals(X))

    # When cardinality == 2, pos_label other than [1, 2] raises an error
    with pytest.raises(ValueError):
        Y_pred = disc_model.predict((X, F), b=0.6, pos_label=3)

    # tertiary
    disc_model.cardinality = 3

    # mock the marginals so that 0.2 for class 1, 0.2 for class 2, and 0.6 for class 3
    def mock_marginals(X):
        a = np.ones((len(X), 1)) * 0.2
        b = np.ones((len(X), 1)) * 0.2
        return np.concatenate((a, b, 1 - a - b), axis=1)

    disc_model.marginals = mock_marginals

    # class 3 has 0.6 of marginal.
    Y_pred = disc_model.predict((X, F))
    np.testing.assert_array_equal(Y_pred, np.array([3, 3, 3]))

    # class 3 has 0.6 of marginal.
    _, Y_prob = disc_model.predict((X, F), return_probs=True)
    np.testing.assert_array_equal(Y_prob, mock_marginals(X))
