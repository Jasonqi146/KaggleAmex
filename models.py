import pickle
import gc
import joblib

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import config as CFG
from metrics import amex_metric, lgb_amex_metric


class LGBM():
    def __init__(self):
        self.metric = lgb_amex_metric
        self.model = lgb
        self.params = {
            'objective': 'binary',
            'metric': "binary_logloss",
            'boosting': 'dart',
            'seed': CFG.seed,
            'num_leaves': 100,
            'learning_rate': 0.01,
            'feature_fraction': 0.20,
            'bagging_freq': 10,
            'bagging_fraction': 0.50,
            'n_jobs': -1,
            'lambda_l2': 2,
            'min_data_in_leaf': 40,
            'early_stopping_rounds': 100,
            'verbosity': -1,
            # Connect to GPU
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

    def train(self, x_train, y_train, x_val, y_val, cat_features):
        lgb_train = lgb.Dataset(
            x_train, y_train, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature=cat_features)

        self.model = self.model.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=10500,
            valid_sets=[lgb_train, lgb_valid],
            feval=self.metric
        )

        del lgb_train, lgb_valid
        gc.collect()

        return self

    def predict(self, x_pred):
        return self.model.predict(x_pred)

    # :TODO Fix bug session crash
    def save_model(self, file_name):
        joblib.dump(self.model, file_name)

        return self

    def load_model(self, file_name):
        self.model = joblib.load(file_name)

        return self


class XGB():
    def __init__(self, params=None):
        self.metric = amex_metric
        self.params = {
            'lambda': 2.594,
            'alpha': 0.0019,
            'n_estimators': 960,
            'max_depth': 9,
            'min_child_weight': 56,
            'learning_rate': 0.027,
            'subsample': 0.8,
            'colsample_bytree': 0.5,
            'objective': 'binary:logistic',
            'random_state': CFG.seed,
            # GPU params
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor'
        } if not params else params
        self.model = XGBClassifier(**self.params, enable_categorical=True)

    def train(self, x_train, y_train, x_val, y_val, cat_features):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_val, label=y_val)

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            num_boost_round=9999,
            early_stopping_rounds=100,
            verbose_eval=100)
        
        del dtrain, dvalid
        gc.collect()
        
        return self

    def predict(self, x_pred):
        dx_pred = xgb.DMatrix(x_pred)
        return self.model.predict(dx_pred)

    def save_model(self, file_name):
        self.model.save_model(file_name.replace('.pkl', '.json'))

        return self

    def load_model(self, file_name):
        self.model = xgb.Booster(model_file=file_name.replace('.pkl', '.json'))

        return self


class CatBoost():
    def __init__(self, params=None):
        self.metric = amex_metric
        self.params = {
            'objective': 'CrossEntropy',
            'depth': 9,
            'random_strength': 0.785,
            'learning_rate': 0.0088,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.821
        } if params is None else params
        
        self.model = CatBoostClassifier(**self.params, iterations=7000, random_state=CFG.seed, task_type='GPU')


    def train(self, x_train, y_train, x_val, y_val, cat_features):
        self.model.fit(x_train, y_train, eval_set=[
                       (x_val, y_val)], cat_features=cat_features, verbose=100)
        return self

    def predict(self, x_pred):
        return self.model.predict_proba(x_pred)[:, 1]

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)

        return self

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            self.model = pickle.load(f)

        return self
