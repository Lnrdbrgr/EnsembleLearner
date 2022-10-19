"""EnsembleLearner
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone

class EnsembleLearner:
    """Class to train an ensemble learner of various scikit-learn models.
    The EnsembleLearner is build and tested for classification problems.

    Args:
        X (pd.DataFrame): Training observations.
        y (pd.Series): Labels to the training observations. 
        models (list): List of models which should be included in the
        ensemble learner. Choose from knn, GaussianNB, DecisionTree,
        MLP, SVM, RandomForest, LogisticRegression, XGBoost.
        eval_size (float): Fraction of the data that is used to train
        the ensemble learner. The models are trained on (1-eval_size)
        of the data and predict the on the eval_size. These are used
        to train the ensemble learner.
        ensemble_learner (str): Model that should be used as the ensemble
        learner. Choose from 'DecisionTree'.

    Attributes:
        X (pd.DataFrame): Training observations.
        y (pd.Series): Labels to the training observations. 
        models (list): List of models which should be included in the
        ensemble learner. Choose from knn, GaussianNB, DecisionTree,
        MLP, SVM, RandomForest, LogisticRegression, XGBoost.
        eval_size (float): Fraction of the data that is used to train
        the ensemble learner. The models are trained on (1-eval_size)
        of the data and predict the on the eval_size. These are used
        to train the ensemble learner.
        ensemble_learner (str): Model that should be used as the ensemble
        learner. Choose from 'DecisionTree'.
        model_mapping (dict): Maps the models specified as characters to
        the corresponding scikit-learn model.
        ensemble_learner_model_mapping (dict): Similar to model_mapping
        only for the ensemble learner.
        ensemble_columns (list): The columns of the data that should be
        predicted on might need to be adapted to the columns that the
        model were trained on since dummy attributes are created.
        Present only after the ensemble learner was trained.
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        models: list,
        eval_size: float = 0.3,
        ensemble_learner: str = 'DecisionTree'
    ):
        self.X = X
        self.y = y
        self.models = models
        self.eval_size = eval_size
        self.ensemble_learner = ensemble_learner
        self.model_mapping = {
            'knn': KNeighborsClassifier(n_neighbors=int(np.sqrt(self.X.shape[0]))),
            'GaussianNB': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=0),
            'MLP': MLPClassifier(max_iter=300, solver='adam', alpha=1e-5, hidden_layer_sizes=(int(self.X.shape[0]/2))),
            'SVM': svm.SVC(),
            'RandomForest': RandomForestClassifier(max_depth=15, n_estimators=1000),
            'LogisticRegression': LogisticRegression(),
            'XGBoost': GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=15)
        }
        self.ensemble_learner_model_mapping = {
            'DecisionTree': DecisionTreeClassifier(random_state=0)
        }

    def __train_test_split(
        self
    ):
        """Private method to split the given data in train- and test-set.

        Returns:
            X_train (pd.DataFrame): Training data.
            X_test (pd.DataFrame): Test data.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
        """
        return train_test_split(self.X, self.y, test_size=self.eval_size)

    def __get_forecaster(
        self,
        model: str
    ):
        """Private method to return the scikit-learn model.

        Returns:
            sklearn.model: The corresponding scikit-learn model.
        """
        return self.model_mapping[model]

    def __get_ensemble_forecaster(
        self,
        model: str
    ):
        """Private method to return the scikit-learn model.

        Returns:
            sklearn.model: The corresponding scikit-learn model.
        """
        return self.ensemble_learner_model_mapping[model]
        
    def __train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
        msg: str = None
    ):
        """Private method to train the individual models.

        Args:
            X (pd.DataFrame): Data used for training.
            y (pd.Series): Labels for training data.
            verbose (bool): If true training progress is printed.
            msg (str): Character that is printed along with the training
            progress. Can be used to distinguish for which state the
            models are trained if function is called repeatedly.
        """
        for model in self.models:
            forecaster = self.__get_forecaster(model=model)
            forecaster.fit(X, y)
            if verbose:
                print(f'Model {model} done training {msg}')

    def __predict_models(
        self,
        X,
        y = None
    ):
        """Private method to predict on data using the internal models.

        Args:
            X (pd.DataFrame): Observations used for predictions.
            y (pd.Series): If specified the true y-values are included
            in the returned prediction frame.
        
        Returns:
            pred_df (pd.DataFrame): Returns the class prediction for every
            model in a dataframe.
        """
        if y is not None:
            pred_df = pd.DataFrame({'truth': y})
        else:
            pred_df = pd.DataFrame()
        for model in self.models:
            forecaster = self.__get_forecaster(model=model)
            pred_df[model] = forecaster.predict(X)
        return pred_df

    def __run_inclass_predictions(
        self,
        verbose: bool = True
    ):
        """Trains the specified models and predicts on the test-set.
        Splits the data specified in the class description first. Loops
        through the specified models, trains them and predicts on the
        test-set. All predictions are stored and returned.

        Args:
            verbose (bool): If true model training progress is printed.

        Returns:
            pred_df (pd.DataFrame): Dataframe that returns ground truth and 
            predictions of all specified models.
        """
        if self.models == 'ALL':
            pass
        else:
            X_train, X_test, y_train, y_test = self.__train_test_split()
            self.__train_models(X=X_train, y=y_train, verbose=verbose, msg='Ensemble Learner Training')
            pred_df = self.__predict_models(X=X_test, y=y_test)
        return pred_df



    def train_ensemble_learner(
        self,
        return_training_predictions: bool = False,
        verbose: bool = True
    ):
        """Trains the ensemble learner on the model predictions.
        Calls the function to train and predict the models. Takes the
        predictions and ground truth and trains the specified
        ensemble learner.

        Args:
            return_training_predictions (bool): If true returns the training
            predictions of the ensemble learner along with the model
            predictions on the training data.
            verbose (bool): If true model training progress is printed.

        Returns:
            pd.DataFrame: If specified returns the training predictions
            of the ensemble learner along with the model predictions.
        """
        ensemble_train_set = self.__run_inclass_predictions(verbose=verbose)
        ensemble_train_X = ensemble_train_set.drop('truth', axis=1)
        ensemble_train_X = pd.get_dummies(ensemble_train_X)
        self.ensemble_columns = ensemble_train_X.columns
        ensemble_train_y = ensemble_train_set['truth']
        ensemble_forecaster = self.__get_ensemble_forecaster(model=self.ensemble_learner)
        ensemble_forecaster.fit(ensemble_train_X, ensemble_train_y)
        self.__train_models(X=self.X, y=self.y, verbose=verbose, msg='Final Model Training')
        if return_training_predictions:
            ensemble_forecaster = self.__get_ensemble_forecaster(model=self.ensemble_learner)
            training_predictions = ensemble_forecaster.predict(ensemble_train_X)
            return(pd.concat([pd.DataFrame({'EL_pred': training_predictions}).reset_index(drop=True), ensemble_train_set.reset_index(drop=True)], axis=1))

    def predict(
        self,
        X: pd.DataFrame
    ):
        """Produces the final predictions of the ensemble learner.
        Trains the models on the whole data and predicts on the unseen
        data. Feeds these predictions to the ensemble learner trained
        before to conclude the final predictions.

        Args:
            X_final (pd.DataFrame): Data
        """
        pred_df = self.__predict_models(X=X)
        pred_dfd = pd.get_dummies(pred_df)
        col_list = list(set(self.ensemble_columns) - set(pred_dfd.columns))
        for col in col_list: pred_dfd[col] = 0
        ensemble_forecaster = self.__get_ensemble_forecaster(model=self.ensemble_learner)
        final_predictions = ensemble_forecaster.predict(pred_dfd)
        return pd.concat([pd.DataFrame({'EL_pred': final_predictions}), pred_df], axis=1)
