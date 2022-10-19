"""
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
    """EnsembleLearner
    """
    
    def __init__(
        self,
        X,
        y,
        models,
        eval_size = 0.3,
        ensemble_learner = 'DecisionTree'
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

    def __train_test_split(self):
        """
        """
        return train_test_split(self.X, self.y, test_size=self.eval_size)

    def __get_forecaster(self, model):
        """
        """
        return clone(self.model_mapping[model])
        

    def __run_inclass_predictions(self, verbose=True):
        """
        """
        if self.models == 'ALL':
            pass
        else:
            X_train, X_test, y_train, y_test = self.__train_test_split()
            pred_df = pd.DataFrame({'truth': y_test})
            for model in self.models:
                forecaster = self.__get_forecaster(model=model)
                forecaster.fit(X_train, y_train)
                pred_df[model] = forecaster.predict(X_test)
                if verbose:
                    print(f'Model {model} done training for Ensemble Learner')
        return pred_df

    def train_ensemble_learner(self, return_training_predictions=False, verbose=True):
        """
        """
        ensemble_train_set = self.__run_inclass_predictions(verbose=verbose)
        ensemble_train_X = ensemble_train_set.drop('truth', axis=1)
        ensemble_train_X = pd.get_dummies(ensemble_train_X)
        self.ensemble_columns = ensemble_train_X.columns
        ensemble_train_y = ensemble_train_set['truth']
        ensemble_forecaster = self.__get_forecaster(model=self.ensemble_learner)
        self.ensemble_forecaster = ensemble_forecaster.fit(ensemble_train_X, ensemble_train_y)
        if return_training_predictions:
            training_predictions = self.ensemble_forecaster.predict(ensemble_train_X)
            return(pd.concat([pd.DataFrame({'EL_pred': training_predictions}).reset_index(drop=True), ensemble_train_set.reset_index(drop=True)], axis=1))

    def predict(self, X_final, verbose=True):
        """
        """
        pred_df = pd.DataFrame()
        for model in self.models:
            forecaster = self.__get_forecaster(model=model)
            forecaster.fit(self.X, self.y)
            pred_df[model] = forecaster.predict(X_final)
            if verbose:
                print(f'Model {model} done training for Final Predictions')
        pred_dfd = pd.get_dummies(pred_df)
        col_list = list(set(self.ensemble_columns) - set(pred_dfd.columns))
        for col in col_list: pred_dfd[col] = 0
        final_predictions = self.ensemble_forecaster.predict(pred_dfd)
        return pd.concat([pd.DataFrame({'EL_pred': final_predictions}), pred_df], axis=1)