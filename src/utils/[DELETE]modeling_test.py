import pandas as pd
import numpy as np
from copy import copy
import time
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score

from xgboost import XGBRegressor


class Model:
    chosen_models = dict()

    def __init__(self, df, target_name, index=None):
        self.target_name = target_name
        self.index = index
        self.df = df

    @property
    def dataframe(self):
        if self.index:
            return self.df.set_index(self.index)
        else:
            return self.df

    @staticmethod
    def send_pickle():
        pass

    def split_dataframe(self, train_num=0.7, random_num=43, scaler=None):
        self.random_num = random_num
        X = self.dataframe.drop(columns=self.target_name)
        y = self.dataframe[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_num, random_state=self.random_num)
        if scaler:
            self.scaler = eval(scaler + '()')
            self.scaler_name = ' (' + scaler + ')'
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)        
        else:
            self.scaler_name = ''
        return (self.X_train, self.X_test, self.y_train, self.y_test)

    def apply_models(self, selected_list=None, excluded_list=None, params_list=None):
        self.models = self.chosen_models.copy()
        if not excluded_list:
            excluded_list = []
        if not selected_list:
            selected_list = []
        current_time = time.time()
        self.models_previous = self.models.copy()
        for element in self.models_previous.keys():
            if (len(selected_list) >= 1 and element not in selected_list) or element in excluded_list:
                self.models.pop(element)
        for model_name in self.models.keys():
            self.models[model_name] = eval(model_name + '()')
        if params_list:
            for params in params_list:
                self.models[params[0] + ': ' + params[1]] = eval(params[0] + '(' + params[1] + ')')
            for params in params_list:
                    if params[0] in self.models:
                        try:
                            self.models.pop(params[0])
                        except Exception:
                            continue
        if self.kfolds_num:
            print(f'-- {self.type.capitalize()}{self.scaler_name}: using best of {self.kfolds_num} {self.kfold}s --')
        else:
            print(f'-- {self.type.capitalize()} --')
        total_time = time.time() - current_time
        for model_name, model in self.models.items():
            start_time = time.time()
            print(f'Starting {model_name}:')
            if self.kfolds_num:
                score_string = 'accuracy'
                if self.type == 'regression':
                    score_string = 'neg_mean_absolute_error'
                cross_val = cross_validate(model, self.X_train, self.y_train, cv=self.kfolds, return_estimator=True, scoring=score_string)
                best_score = max(cross_val['test_score'])             
                for index, element in enumerate(cross_val['test_score']):
                    if element == best_score:
                        best_score = index
                model = cross_val['estimator'][best_score]
            else:
                model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            self.models[model_name] = {'test': np.array(self.y_test), 'prediction': self.y_pred, 'model': model}
            execution_time = time.time() - start_time
            total_time += execution_time
            print(f'- {model_name} done in {round(execution_time, 2)} sec(s). Total time: {round(total_time, 2)}')
        return self.models

    def evaluate_metrics(self, selection_list=None):
        self.models_evaluated_previous = self.models
        self.models_evaluated = copy(self.models_evaluated_previous)
        if selection_list:
            for element in self.models_evaluated_previous.keys():
                if element not in selection_list:
                    self.models_evaluated.pop(element)


class Regression(Model):

    def __init__(self, dataframe, target_name, index=None):
        super().__init__(dataframe, target_name, index)
        self.type = 'regression'

    @classmethod
    def add_models(cls, regression_list):
        if regression_list:
            for element in regression_list:
                cls.chosen_models[element] = ''

    @classmethod
    def remove_models(cls, regression_list):
        if regression_list:
            for element in regression_list:
                cls.chosen_models.pop(element)

    def apply_models(self, selected_list=None, excluded_list=None, params_list=None, kfolds_num=None):
        self.kfolds_num = kfolds_num
        if kfolds_num:
            self.kfolds = KFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
            self.kfold = 'fold'
        super().apply_models(selected_list, excluded_list, params_list)

    def evaluate_metrics(self):
        super().evaluate_metrics(selection_list=None)
        for model_name, model_results in self.models_evaluated.items():
            rmse = mean_squared_error(model_results['test'], model_results['prediction'], squared=False)
            mse = mean_squared_error(model_results['test'], model_results['prediction'])
            mae = mean_absolute_error(model_results['test'], model_results['prediction'])
            r2 = r2_score(model_results['test'], model_results['prediction'])
            mape = mean_absolute_percentage_error(model_results['test'], model_results['prediction'])
            self.models_evaluated[model_name]['metrics'] = {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2_score': r2, 'mape': mape}
        return self.models_evaluated

    def create_dataframe(self):
        self.models_metrics = self.models_evaluated.copy()
        best_values_list = []
        worst_values_list = []
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name]['metrics']
            if len(self.models_metrics) > 1:
                model_values = [value if type(value) is not list else sum([row[index] for index, row in enumerate(value)]) for value in self.models_evaluated[model_name]['metrics'].values()]
                if not best_values_list:
                    best_values_list = [[model_name, value] for value in model_values]
                    worst_values_list = [[model_name, value] for value in model_values]
                else:
                    for index, value in enumerate(model_values):
                        if value < best_values_list[index][1]:
                            if index != 3:
                                best_values_list[index][1] = value
                                best_values_list[index][0] = model_name
                            else:
                                worst_values_list[index][1] = value
                                worst_values_list[index][0] = model_name
                        if value > worst_values_list[index][1]:
                            if index != 3:
                                worst_values_list[index][1] = value
                                worst_values_list[index][0] = model_name
                            else:
                                best_values_list[index][1] = value
                                best_values_list[index][0] = model_name
        df = pd.DataFrame(data=self.models_metrics)
        if best_values_list:
            best_values_list = [element[0] for element in best_values_list]
            worst_values_list = [element[0] for element in worst_values_list]
            df['BEST'] = best_values_list
            df['WORST'] = worst_values_list
        return df


class Classification(Model):

    def __init__(self, dataframe, target_name, index=None):
        super().__init__(dataframe, target_name, index)
        self.type = 'classification'

    @classmethod
    def add_models(cls, classification_list):
        if classification_list:
            for element in classification_list:
                cls.chosen_models[element] = ''

    @classmethod
    def remove_models(cls, classification_list):
        if classification_list:
            for element in classification_list:
                cls.chosen_models.pop(element)

    def apply_models(self, selected_list=None, excluded_list=None, params_list=None, kfolds_num=None):
        if kfolds_num:
            self.kfolds = StratifiedKFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
            self.kfolds_num = kfolds_num
            self.kfold = 'stratified fold'
        super().apply_models(selected_list, excluded_list, params_list)

    def evaluate_metrics(self, params_list=None):
        super().evaluate_metrics(selection_list=None)
        for model_name, model_results in self.models_evaluated.items():
            accuracy = "accuracy_score (model_results['test'], model_results['prediction']"
            recall = "recall_score (model_results['test'], model_results['prediction']"
            precision = "precision_score (model_results['test'], model_results['prediction']"
            f1 = "f1_score (model_results['test'], model_results['prediction']"
            matrix = "confusion_matrix (model_results['test'], model_results['prediction']"
            list_of_metrics = []
            for element in (accuracy, recall, precision, f1, matrix):
                if params_list:
                    for params in params_list:
                        if params[0] == element.split()[0]:
                            list_of_metrics.append(eval(element + "," + params[1] + ")"))
                        else:
                            list_of_metrics.append(eval(element + ")"))
                            continue
                else:
                    list_of_metrics.append(eval(element + ")"))
            confusion = [element for element in list_of_metrics[-1]]
            self.models_evaluated[model_name]['metrics'] = {'accuracy': list_of_metrics[0], 'recall': list_of_metrics[1], 'precision': list_of_metrics[2], 'f1_score': list_of_metrics[3], 'confusion_matrix': confusion}
        return self.models_evaluated

    def create_dataframe(self):
        self.models_metrics = self.models_evaluated.copy()
        best_values_list = []
        worst_values_list = []
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name]['metrics']
            if len(self.models_metrics) > 1:
                model_values = [value if type(value) is not list else sum([row[index] for index, row in enumerate(value)]) for value in self.models_evaluated[model_name]['metrics'].values()]
                if not best_values_list:
                    best_values_list = [[model_name, value] for value in model_values]
                    worst_values_list = [[model_name, value] for value in model_values]
                else:
                    for index, value in enumerate(model_values):
                        if value > best_values_list[index][1]:
                            best_values_list[index][1] = value
                            best_values_list[index][0] = model_name
                        if value < worst_values_list[index][1]:
                            worst_values_list[index][1] = value
                            worst_values_list[index][0] = model_name
        df = pd.DataFrame(data=self.models_metrics)
        if best_values_list:
            best_values_list = [element[0] for element in best_values_list]
            worst_values_list = [element[0] for element in worst_values_list]
            df['BEST'] = best_values_list
            df['WORST'] = worst_values_list
        return df
