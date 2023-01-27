import pandas as pd
import numpy as np
from copy import copy
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score


class Data_modeling:
    
    def __init__(self, dataframe, target_name, type='regression', index=None):
        self.target_name = target_name
        if index:
            self.dataframe = dataframe.set_index(index)
        else:
            self.dataframe = dataframe
        self.type = type

    def split_dataframe(self, train_size):
        '''Splits the dataframe, required to apply the models'''
        X = self.dataframe.drop(columns=self.target_name)
        y = self.dataframe[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_size, random_state=43)
    
    def apply_models(self, selection=None, params=None):
        '''Applies every selected model, all of them if none is selected'''
        self.models_regression = {'LinearRegression': LinearRegression(), 'Ridge': Ridge()}
        self.models_classification = {'LogisticRegression': LogisticRegression()}
        if self.type == 'regression':
            self.models = copy(self.models_regression)
        elif self.type == 'classification':
            self.models = copy(self.models_classification)
        self.models_previous = self.models.copy()
        if selection:
            for element in self.models_previous.keys():
                if element not in selection:
                    self.models.pop(element)
        if params:
            self.models[params[0]] = eval(params[0] + '(' + params[1] + ')')
            self.models[params[0] + ': ' + params[1]] = self.models.pop(params[0])
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            self.models[model_name] = {'test': np.array(self.y_test), 'prediction': self.y_pred}
        return self.models

    def evaluate_metrics(self, selection=None):
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
                accuracy = accuracy_score(model_results['test'], model_results['prediction'])
                recall = recall_score(model_results['test'], model_results['prediction'])
                precision = precision_score(model_results['test'], model_results['prediction'])
                confusion = confusion_matrix(model_results['test'], model_results['prediction'])
                diagonal_total = 0
                for index, element in enumerate(confusion):
                    diagonal_total += element[index]
                f1 = f1_score(model_results['test'], model_results['prediction'])
                self.models_evaluated[model_name]['metrics'] = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'confusion_diagonal': diagonal_total, 'f1_score': f1}
        return self.models_evaluated

    def create_dataframe(self):
        '''Returns a dataframe with the metrics of each model'''
        self.models_metrics = self.models_evaluated.copy()
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name]['metrics']
        df = pd.DataFrame(data=self.models_metrics)
        return df
