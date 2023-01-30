import pandas as pd
import numpy as np
from copy import copy
import time
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score


class Model:
    
    def __init__(self, dataframe, target_name, type='regression', index=None):
        self.target_name = target_name
        if index:
            self.dataframe = dataframe.set_index(index)
        else:
            self.dataframe = dataframe
        self.type = type

    def split_dataframe(self, train_num=0.7, random_num=43):
        self.random_num = random_num
        '''Splits the dataframe, required to apply the models'''
        X = self.dataframe.drop(columns=self.target_name)
        y = self.dataframe[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_num, random_state=self.random_num)
        return (self.X_train, self.X_test, self.y_train, self.y_test)
    
    def apply_models(self, selected_list=None, excluded_list=None, params_list=None, kfolds_num=None):
        '''Applies every selected model, all of them if none is selected'''
        if not excluded_list:
            excluded_list = []
        if not selected_list:
            selected_list = []
        fold = 'KFold'
        current_time = time.time()
        self.models_regression = {'LinearRegression': '', # Write your regression models names as keys of the dict. Must be imported
                                    'Ridge': '', 
                                    'DecisionTreeRegressor': '', 
                                    'KNeighborsRegressor': '',
                                    'RandomForestRegressor': '',
                                    'SVR': ''
                                    }
                                    
        self.models_classification = {'BernoulliNB': '',  # Write your classification models names as keys of the dict. Must be imported
                                        'GaussianNB': '',
                                        'LogisticRegression': '',
                                        'DecisionTreeClassifier': '',
                                        'KNeighborsClassifier': '',
                                        'RandomForestClassifier': '',
                                        'SVC': ''
                                        }
        if self.type == 'regression':
            self.models = copy(self.models_regression)
            if kfolds_num:
                kfolds = KFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
        elif self.type == 'classification':
            self.models = copy(self.models_classification)
            if kfolds_num:
                kfolds = StratifiedKFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
                fold = 'StratifiedKFold'
        self.models_previous = self.models.copy()
        for element in self.models_previous.keys():
            if (len(selected_list) >= 1 and element not in selected_list) or element in excluded_list:
                self.models.pop(element)
        for model_name in self.models.keys():
            self.models[model_name] = eval(model_name + '()')
        if params_list:
            for params in params_list:
                self.models[params[0]] = eval(params[0] + '(' + params[1] + ')')
                self.models[params[0] + ': ' + params[1]] = self.models[params[0]]
            for params in params_list:
                    if params[0] in self.models:
                        self.models.pop(params[0])
        if kfolds_num:
            print(f'-- {self.type.capitalize()}: using best of {fold} --')
        else:
            print(f'-- {self.type.capitalize()} --')
        total_time = time.time() - current_time
        for model_name, model in self.models.items():
            print(model)
            start_time = time.time()
            print(f'Starting {model_name}:')
            if kfolds_num:
                cross_val = cross_validate(model, self.X_train, self.y_train, cv=kfolds, return_estimator=True)
                best_score = cross_val['test_score'][0]
                for index, element in enumerate(cross_val['test_score']):
                    if element is not cross_val['test_score'][0]:
                        if element > cross_val['test_score'][index - 1]:
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

    def evaluate_metrics(self, selection=None, params_list=False):
        '''Anotates regression metrics based on the real values and the predicion'''
        self.models_evaluated_previous = self.models
        self.models_evaluated = copy(self.models_evaluated_previous)
        if selection:
            for element in self.models_evaluated_previous.keys():
                if element not in selection:
                    self.models_evaluated.pop(element)
        if self.type == 'regression':
            for model_name, model_results in self.models_evaluated.items():
                rmse = mean_squared_error(model_results['test'], model_results['prediction'], squared=False)
                mse = mean_squared_error(model_results['test'], model_results['prediction'])
                mae = mean_absolute_error(model_results['test'], model_results['prediction'])
                r2 = r2_score(model_results['test'], model_results['prediction'])
                mape = mean_absolute_percentage_error(model_results['test'], model_results['prediction'])
                self.models_evaluated[model_name]['metrics'] = {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2_score': r2, 'mape': mape}
        elif self.type == 'classification':
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
        '''Returns a dataframe with the metrics of each model'''
        self.models_metrics = self.models_evaluated.copy()
        metrics_list = []
        best_values_list = []
        worst_values_list = []
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name]['metrics']
            model_values = [value if type(value) is not list else sum([row[index] for index, row in enumerate(value)]) for value in self.models_evaluated[model_name]['metrics'].values()]
            if not metrics_list:
                metrics_list += [key for key in self.models_evaluated[model_name]['metrics'].keys()]
            if not best_values_list:
                best_values_list = [[model_name, value] for value in model_values]
                worst_values_list = [[model_name, value] for value in model_values]
            else:
                for index, value in enumerate(model_values):
                    if value > best_values_list[index][1] and self.type == 'classification':
                        best_values_list[index][1] = value
                        best_values_list[index][0] = model_name
                    if value < best_values_list[index][1] and self.type == 'regression':
                        best_values_list[index][1] = value
                        best_values_list[index][0] = model_name
                    if value < worst_values_list[index][1] and self.type == 'classification':
                        worst_values_list[index][1] = value
                        worst_values_list[index][0] = model_name
                    if value > worst_values_list[index][1] and self.type == 'regression':
                        worst_values_list[index][1] = value
                        worst_values_list[index][0] = model_name                    
        df = pd.DataFrame(data=self.models_metrics)
        best_values_list = [element[0] for element in best_values_list]
        worst_values_list = [element[0] for element in worst_values_list]
        if self.type == 'regression':
            not_worst_r2 = worst_values_list[-2]
            not_best_r2 = best_values_list[-2]
            worst_values_list = [element if element is not not_worst_r2 else not_best_r2 for element in worst_values_list]
            best_values_list = [element if element is not not_best_r2 else not_worst_r2 for element in worst_values_list]
        df['BEST'] = best_values_list
        df['WORST'] = worst_values_list
        return df

    def send_pickle(self):
        pass