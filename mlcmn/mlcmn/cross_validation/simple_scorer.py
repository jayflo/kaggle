
import numpy as np

from sklearn import cross_validation
from sklearn.cross_validation import KFold


def _class01(X):
    X[X < 0.5] = 0
    X[X >= 0.5] = 1

    return X

"""
Parameters
----------
@type ll: list
@param ll: a list of array-likes of size n_samples x n_classes and whose values
    are the probability that the sample predicts the class associated to that
    column (see predict_proba).
"""
def _list_avg(ll):
    if len(ll) < 1:
        return

    # compute average probability class == 1
    z = ll[0][:, 1]

    for i in range(2, len(ll)):
        z += ll[i][:, 1]

    z = z / len(ll)

    # use average to predict 0 or 1
    return _class01(z)


"""
A class for computing the accuracy of (possibly) many algorithms on the same
dataset.
"""
class cv_scorer():

    """
    Parameters
    ----------
    @type df: pandas.DataFrame
    @param df: the dataframe containing predictor columns

    @type target: string
    @param target: the name of column to be predicted

    @type cv: integer
    @param cv: the number of folds

    @type proba_combine: function
    @param proba_combine: when `mean` is True, this function that receives a
        list of array-likes of size n_samples x n_classes and whose value at
        row=i, col=j is the probability that sample i predicted the class given
        by column j (see predict_proba).  The default assumes n_classes=2 and
        averages the probability that each sample has outcome 1 then predicts 1
        for the sample when this probability is at least 0.5.  It must return
        an array-like used for comparison with the target column, e.g.

            def my_post_process(X):
                X[X < 0.5] = 0
                X[X >= 0.5] = 1

                return X
    """
    def __init__(self, df, target, cv=3, proba_combine=_list_avg,
                 post_process=_class01):
        self.df = df
        self.cv = cv
        self.target = target
        self.proba_combine = proba_combine
        self.post_process = post_process
        self.__init__tar

    """
    Compute the accuracy of (possibly) many algorithms.  Note: all optional
    parameters default to the values passed into the constructor.

    Parameters
    ----------
    @type list_alg: list
    @param list_alg: a list of dictionaries each of which has the following
        properties:
            alg: the algorithm to be executed.  Must be passable to
                sklearn.cross_validation.cross_val_score and have
                `fit`/`predict_proba` methods.
            predictors: a list of strings denoting the dataframe columns
                to be used a predctors for the target.

    @type target: string
    @param target: the column of the dataframe to be predicted.

    @type cv: integer
    @param cv: the number of cross validations tests/folds to be used.

    @type proba_combine: function
    @param proba_combine: when `mean` is True, this function that receives a
        list of array-likes of size n_samples x n_classes and whose value at
        row=i, col=j is the probability that sample i predicted the class given
        by column j (see predict_proba).  The default assumes n_classes=2 and
        averages the probability that each sample has outcome 1 then predicts 1
        for the sample when this probability is at least 0.5.  It must return
        an array-like used for comparison with the target column, e.g.

            def my_post_process(X):
                X[X < 0.5] = 0
                X[X >= 0.5] = 1

                return X

    @type mean: boolean
    @param mean: when true and len(list_alg) > 1, the predictions of the algorithms
        will be averaged, rather than computed independently.

    Returns
    -------
    @rtype: list
    @return: the scores for each algorithm or their average.
    """
    def cross_val_score(self, list_alg, target=None, cv=None, proba_combine=None,
                        alg_type='', mean=False):
        target = target or self.target
        cv = cv or self.cv
        proba_combine = proba_combine or self.proba_combine

        if mean:
            return self._score_many(
                self, list_alg, cv, target, alg_type
            )
        else:
            return self._score_mean(
                self, list_alg, cv, target, alg_type, proba_combine
            )

    def _score_single(self, alg, predictors, cv, target, alg_type):
        scores = cross_validation.cross_val_score(
            alg, self.df[predictors], self.df[target], cv=cv
        )

        self._print_score(alg_type, scores.mean())

        return scores.mean()

    def _score_many(self, list_alg, cv, target, alg_type):
        scores = []

        for algos in list_alg:
            alg = algos.alg
            predictors = algos.predictors

            scores.append(self._score_single(
                alg, predictors, cv, target, alg_type
            ))

        return scores

    def _score_mean(self, list_alg, cv, target, alg_type, proba_combine):
        df = self.df
        predictions = []
        kf = KFold(df.shape[0], n_folds=cv, random_state=1)

        for train, test in kf:
            train_target = df[target].iloc[train]
            test_predictions = []

            for algos in list_alg:
                alg = algos.alg
                predictors = algos.predictors
                alg.fit(df[predictors].iloc[train, :], train_target)
                alg_predictions = alg.predict_proba(df[predictors].iloc[test, :])
                test_predictions.append(alg_predictions)

            predictions.append(proba_combine(test_predictions))

        predictions = np.concatenate(predictions, axis=0)
        score = len(df[df[target] == predictions]) / len(df.index)

        self._print_score(alg_type, score)

        return score

    def _print_score(self, alg_type, score):
        print(alg_type, 'score:', score)
