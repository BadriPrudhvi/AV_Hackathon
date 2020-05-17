import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from ml_modules.custom_fold_generator import CustomFolds, FoldScheme

'''
modify the scoring_metric to use custom metric, for not using roc_auc_score
scoring metric should accept y_true and y_predicted as parameters
add a functionality to give folds as an iterable
'''

class_instance = lambda a, b: eval("{}(**{})".format(a, b if b is not None else {}))

class Estimator(object):

    def __init__(self, model, n_splits=5, random_state=100, shuffle=True, validation_scheme=FoldScheme.StratifiedKFold,
                 cv_group_col=None, early_stopping_rounds=None, categorical_features_indices=None, verbose=100,
                 task_type = 'classification', eval_metric='auc', scoring_metric=roc_auc_score, over_sampling=False,
                 n_jobs=-1, **kwargs
                 ):
        try:
            # build model instance from tuple/list of ModelName and params
            # model should be imported before creating the instance
            self.model = class_instance(model[0], model[1])
        except Exception as e:
            # model instance is already passed
            self.model = clone(model)

        self.n_splits = n_splits
        self.random_state = random_state
        self.seed = random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        if isinstance(validation_scheme, str) or isinstance(validation_scheme, str):
            self.validation_scheme = FoldScheme(validation_scheme)
        else:
            self.validation_scheme = validation_scheme
        self.cv_group_col = cv_group_col
        self.categorical_features_indices=categorical_features_indices
        self.verbose = verbose
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.scoring_metric = scoring_metric
        self.over_sampling = over_sampling
        

    def get_params(self):
        return {
            'model': (self.model.__class__.__name__, self.model.get_params()),
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'shuffle': self.shuffle,
            'n_jobs': self.n_jobs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_scheme': self.validation_scheme,
            'cv_group_col': self.cv_group_col,
            'task_type': self.task_type,
            'eval_metric': self.eval_metric,
            'scoring_metric': self.scoring_metric,
            
        }

    def fit(self, x, y, use_oof=False, n_jobs=-1):
        if not hasattr(self.model, 'fit') :
            raise Exception ("Model/algorithm needs to implement fit()")
        fitted_models = []

        if use_oof:
            folds = CustomFolds(num_folds=self.n_splits, random_state=self.random_state, shuffle=self.shuffle, validation_scheme=self.validation_scheme)
            self.indices = folds.split(x,y,group=self.cv_group_col)

            for i, (train_index, test_index) in enumerate(self.indices):
                model = clone(self.model)
                model.n_jobs = n_jobs
                
                if (isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor)) and (self.early_stopping_rounds is not None):
                    model.fit(X=x.loc[train_index], y=y[train_index], 
                              eval_set=[(x.loc[train_index],y[train_index]), (x.loc[test_index],y[test_index])],
                              verbose=self.verbose, eval_metric=self.eval_metric, early_stopping_rounds=self.early_stopping_rounds,
                              eval_names=['train', 'valid'])
                                    
                elif (isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor)) and (self.early_stopping_rounds is not None):
                    model.fit(X=x.loc[train_index], y=y[train_index], 
                              eval_set=[(x.loc[test_index], y[test_index])],
                              verbose=self.verbose, eval_metric=self.eval_metric, early_stopping_rounds=self.early_stopping_rounds)
                    
                elif (isinstance(model, CatBoostClassifier) or isinstance(model, CatBoostRegressor)) and (self.early_stopping_rounds is not None):
                    model.fit(x.loc[train_index], y[train_index], cat_features=self.categorical_features_indices,
                             eval_set=(x.loc[test_index], y[test_index]), 
                             use_best_model=True, verbose=self.verbose, early_stopping_rounds=self.early_stopping_rounds)
                    
                else:
                    x_train, y_train = x.loc[train_index], y[train_index]
                    if self.over_sampling:
                        print("oversampling")
                        x_train, y_train = SMOTE().fit_resample(x_train, y_train)
                    model.fit(x_train, y_train)
                     
                fitted_models.append(model)
        else:
            model = clone(self.model)
            model.n_jobs = n_jobs
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size =0.2, shuffle=True, random_state=100)
            if isinstance(model, LGBMClassifier):
                if self.early_stopping_rounds is not None:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val, y_val), (x_train, y_train)],
                              verbose=False, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds,
                              eval_names=['valid', 'train'])

            elif isinstance(model, XGBClassifier):
                if self.early_stopping_rounds is not None:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val,y_val)],
                        verbose=False, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds)

            model.fit(x, y)
            
            fitted_models.append(model)
        self.fitted_models = fitted_models
        return self

    def feature_importances(self, columns=None):
        if not hasattr(self, 'fitted_models') :
            raise Exception ("Model/algorithm needs to implement fit()")
        if isinstance(self.model, LogisticRegression):
            feature_importances = np.column_stack(m.coef_[0] for m in self.fitted_models)
        elif isinstance(self.model, LinearRegression) :
            feature_importances = np.column_stack(m.coef_ for m in self.fitted_models)
        else:
            feature_importances = np.column_stack(m.feature_importances_ for m in self.fitted_models)
        importances = np.mean(1.*feature_importances/feature_importances.sum(axis=0), axis=1)
        if columns is not None:
            if len(columns) != len(importances):
                raise ValueError("Columns length Mismatch")
            df = pd.DataFrame(zip(columns, importances), columns=['column', 'feature_importance'])
        else:
            df = pd.DataFrame(zip(range(len(importances)), importances), columns=['column_index', 'feature_importance'])
        df.sort_values(by='feature_importance', ascending=False, inplace=True)
        df['rank'] = np.arange(len(importances)) + 1
        return df

    def transform(self, x):
        if not hasattr(self, 'fitted_models') :
            raise Exception ("Model/algorithm needs to implement fit()")
        if self.task_type == 'classification':
            all_preds = []
            for est in self.fitted_models:
                all_preds.append(est.predict_proba(x))
            return np.mean(all_preds, axis=0)
        else:
            return np.mean(np.column_stack((est.predict(x) for est in self.fitted_models)), axis=1)
    
    def fit_transform(self, x, y):
        self.fit(x, y, use_oof=True)
        predictions = np.zeros((x.shape[0], len(np.unique(y))))
        for i, (train_index, test_index) in enumerate(self.indices):
            if self.task_type == 'classification': 
                predictions[test_index] = self.fitted_models[i].predict_proba(x.loc[test_index])#[:,1]
            else:
                predictions[test_index] = self.fitted_models[i].predict(x.loc[test_index]) 
        predictions = np.argmax(predictions, axis=-1) + 1 #converting to class labels
        self.cv_scores = [
            self.scoring_metric(y[test_index], predictions[test_index])
            for i, (train_index, test_index) in enumerate(self.indices)
        ]
        self.avg_cv_score = np.mean(self.cv_scores)
        self.overall_cv_score = self.scoring_metric(y, predictions)
        return predictions

    def save_model(self):
        pass

    def load_model(self):
        pass

    def predict(self, x):
        return self.transform(x)
    
    def predict_proba(self, x):
        return self.transform(x)

    def get_repeated_out_of_folds(self, x, y, num_repeats=1):
        cv_scores = []
        fitted_models = []
        for iteration in range(num_repeats):
            self.random_state = self.seed*(iteration+1)
            predictions = self.fit_transform(x, y)
            cv_scores.extend(self.cv_scores)
            fitted_models.extend(self.fitted_models)
            self.random_state = self.seed

        self.fitted_models = fitted_models
        return {
            'cv_scores': cv_scores,
            'avg_cv_score': np.mean(cv_scores),
            'var_scores': np.std(cv_scores),
            'overall_cv_score': self.overall_cv_score,
        }

    def get_nested_scores(self, x, y):
        pass