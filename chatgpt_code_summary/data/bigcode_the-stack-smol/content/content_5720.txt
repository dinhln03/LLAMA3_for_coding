from typing import Any, Dict

import pandas
import numpy as np
from sklearn import datasets

from opticverge.core.chromosome.abstract_chromosome import AbstractChromosome
from opticverge.core.enum.objective import Objective
from opticverge.core.log.logger import data_logger, DATA
from opticverge.core.solver.abstract_solver import AbstractSolver
from opticverge.external.scikit.enum.normaliser import Normaliser
from opticverge.external.scikit.enum.scoring_function import Scoring
from opticverge.external.scikit.problem.abstract_regression_problem import AbstractRegressionProblem


class RedWineQualityPredictionProblem(AbstractRegressionProblem):
    def __init__(self, scoring_function: Scoring, normaliser: Normaliser = None, folds: int = 1):

        df = pandas.read_csv("./winequality-red.csv", sep=";", usecols=[
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"
        ])

        data = np.array(df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]])
        target = np.array(df["quality"])

        super(RedWineQualityPredictionProblem, self).__init__(
            Objective.Minimisation,
            "Red Wine Quality Prediction",
            data_x=data,
            target_x=target,
            normaliser=normaliser,
            folds=folds,
            scoring_function=scoring_function
        )

    def log_chromosome(self, chromosome: AbstractChromosome, solver: AbstractSolver,
                       additional_data: Dict[str, Any] = None, separator="|"):
        data_str = super(RedWineQualityPredictionProblem, self).log_chromosome(
            chromosome,
            solver,
            None
        )

        data_logger.log(DATA, data_str)

    def objective_function(self, chromosome: AbstractChromosome):
        super(RedWineQualityPredictionProblem, self).objective_function(chromosome)
