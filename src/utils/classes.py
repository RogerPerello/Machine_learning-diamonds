import pandas as pd
import numpy as np
import time
import joblib
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

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



class Model:
    '''
    Parent class of Regression and Classification. Used only through child classes, never directly.
    Check child classes documentation for a full overview of the available methods as well as the recommended order of use.
    The info below is not complete and only shows some of the features accessible through the child classes

    ...

    Methods
    -------
    send_pickle(model, path)
        Saves model as pickle to the chosen path. This is a static method and theoretically could be used at any time

    split_dataframe(train_num=0.7, random_num=43, scaler=None, return_entire_Xy=False)
        Splits dataframe into X_train, X_test, y_train and y_test. 
        Used after adding models to the chosen_models attribute of the selected child class
    
    prepare_models(selected_list=None, excluded_list=None, params_list=None)
        Makes models suitable for application. Used after the split
    
    apply_models()
        Predicts using the split dataframe and the prepared models. No folds. 
        Used after prepare_models(); can use apply_and_evaluate_kfolds() (see in child class) instead
    
    create_dataframe(best_values_list, worst_values_list)
        Creates a dataframe with the metrics of the models. Complements a method in child class. 
        Used after evaluate_metrics() or apply_and_evaluate_kfolds() (see both in child class)
    
    visualize(*metrics_selection)
        Creates a lineplot with the metrics of the models. Used after create_dataframe()
    '''


    def __init__(self, df, target_name, index=None):
        '''

        Parameters
        ----------
        df : dataframe
            The dataframe to which the models will be used
        target_name : str
            The name of the target column
        index : str (defaut=None)
            The name of the column to become index, if any. 
            Useful if the dataframe has not been cleaned and/or the analysis is to be performed without changing it directly
        '''

        self.target_name = target_name
        self.index = index
        self.df = df


    def __str__ (self):
        '''

        Returns
        -------
        str
           Type of instance and recommended next step
        '''

        return f'{self.type.capitalize()}. Progress: {self.progress}'


    @property
    def dataframe(self):
        '''Automatically changes the index if any is passed to the constructor

        Returns
        -------
        dataframe
            The dataframe with the chosen index
        '''

        if self.index:
            return self.df.set_index(self.index)
        else:
            return self.df


    @staticmethod
    def send_pickle(model, path):
        '''Saves model as pickle to the chosen path

        Parameters
        ----------
        model : model object
            The model to be saved
        path : string
            The path where the model will be saved

        Returns
        -------
        string
            Confirmation
        '''
        joblib.dump(model, path)
        return 'Done'


    def split_dataframe(self, train_num=0.7, random_num=43, scaler=None, return_entire_Xy=False):
        '''Splits dataframe into X_train, X_test, y_train and y_test.  
Preferably used after adding models to the chosen_models attribute with add_models()

        Parameters
        ----------
        train_num : float (default=0.7)
            Proportion of the train splits
        random_num : int (default=43)
            random_state for the split
        scaler : str (default=None)
            The name of the chosen scaler, if any
        return_entire_Xy: boolean (default=False)
            Changes return to take in the entire X and y instead of the splits

        Returns
        -------
        tuple of pandas objects
            X_train, X_test, y_train and y_test
        '''

        self.random_num = random_num
        X = self.dataframe.drop(columns=self.target_name)
        y = self.dataframe[self.target_name]
        stratify_option = None
        if self.type == 'classification':
            stratify_option = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=train_num, random_state=self.random_num, stratify=stratify_option)
        if scaler:
            self.scaler = eval(scaler + '()')
            self.scaler_name = ' (' + scaler + ')'
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)        
            if return_entire_Xy:
                self.scaler = eval(scaler + '()')
                X = self.scaler.fit_transform(X)
        else:
            self.scaler_name = ''
        self.progress = 'the dataframe has been split. Now, you may prepare your models with the prepare_models() method'
        if return_entire_Xy:
            return (X, y)
        else:
            return (self.X_train, self.X_test, self.y_train, self.y_test)


    def prepare_models(self, selected_list=None, excluded_list=None, params_list=None):
        '''Makes models suitable for application. Used after the split

        Parameters
        ----------
        selected_list : list of str (default=None)
            Limits the models to be prepared to the ones in this list, if any
        excluded_list : list of str (default=None)
            Excludes the models of the list from the preparation, if any
        params_list : list of lists (default=None)
            For every model (first element of each inner list), applies the chosen hiperparameters 
            (second element of each inner list, all together in a string), if any

        Returns
        -------
        str
            Confirmation
        '''

        self.models = self.chosen_models.copy()
        if not excluded_list:
            excluded_list = []
        if not selected_list:
            selected_list = []
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
        self.progress = 'your models are now prepared. Apply them with apply_models() or use apply_and_evaluate_kfolds() instead if you want kfolds to be used'
        return 'Done'


    def apply_models(self):
        '''Predicts using the split dataframe and the prepared models. No folds.
Used after prepare_models(); may use apply_and_evaluate_kfolds() instead

        Returns
        -------
        dict
            Dictionary with the test, prediction and model function for each model string
        '''

        print(f'-- {self.type.capitalize()} --')
        current_time = time.time()
        total_time = time.time() - current_time
        for model_name, model in self.models.items():
            start_time = time.time()
            print(f'Starting {model_name}:')
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            self.models[model_name] = {'test': np.array(self.y_test), 'prediction': self.y_pred, 'model': model}
            execution_time = time.time() - start_time
            total_time += execution_time
            print(f'- {model_name} done in {round(execution_time, 2)} sec(s). Total time: {round(total_time, 2)}')
        self.progress = 'your models have been applied to the data passed. Get the metrics using evaluate_metrics()'
        return self.models


    def create_dataframe(self, best_values_list, worst_values_list):
        '''Creates a dataframe with the metrics of each model.
Used after evaluate_metrics() or apply_and_evaluate_kfolds()

        Parameters
        ----------
        best_values_list : list of numbers
            Ordered list of the best values and their model
        worst_values_list : list of str
            Ordered list of the worst values and their model

        '''

        self.df = pd.DataFrame(data=self.models_metrics)
        if best_values_list:
            best_values_list = [element[0] for element in best_values_list]
            worst_values_list = [element[0] for element in worst_values_list]
            self.df['BEST'] = best_values_list
            self.df['WORST'] = worst_values_list


    def visualize(self, *metrics_selection):
        '''Creates a lineplot with the metrics of the models. Has no return but shows a graphic. Used after create_dataframe()

        Parameters
        ----------
        metrics_selection : str (*args)
            Selects the metrics for visualization. If none provided, selects all
        '''

        visualization_dict = {'models': [model_name for model_name in self.models_metrics.keys() for metric in self.models_metrics[model_name] if (not metrics_selection or metric in metrics_selection)],
                              'metrics': [metric for model_name in self.models_metrics.keys() for metric in self.models_metrics[model_name] if (not metrics_selection or metric in metrics_selection)],
                              'values': [self.models_metrics[model_name][metric] for model_name in self.models_metrics.keys() for metric in self.models_metrics[model_name] if (not metrics_selection or metric in metrics_selection)]
                              }
        sns.lineplot(data=visualization_dict, x='models', y='values', hue='metrics')
        plt.tick_params(axis='x', labelrotation=90)
        plt.title(f'{self.chosen_format.capitalize()} comparison')
        self.progress = 'you may call create_dataframe() and visualize() multiple times to change how and which metrics are displayed (mean, variance...), or use send_pickle() to save your chosen model'
        plt.show()



class Regression(Model):
    '''
    Child class of Model for regression algorithms. To be used directly instead of Model.
    Methods are numbered by recommended order of use.
    Print instance at any time to see the recommended next step

    ...

    Attributes
    -------
    chosen_models : dict
        Empty dict of models to work with (to be filled with empty keys using add_models class method)

    Class methods
    -------
    add_models(*regression)
        1) Adds models to the class attribute chosen_models. Required first, before putting instances to work, since chosen_models is empty by default.
        It should be called only once, except if more models are needed at some point for further analysis

    remove_models(*regression)
        Removes models from the class attribute chosen_models

    Instance methods
    -------
    apply_and_evaluate_kfolds(kfolds_num=5)
        4 & 5) Used when models are prepared (3). Applies the models to the splits of the dataframe using kfolds and gets the metrics. 
        Can use apply_models() (4) + evaluate_metrics() (5) instead if kfolds are not wanted
    
    evaluate_metrics()
       5) Extracts the metrics for models already applied with apply_models() (4). No folds
    
    create_dataframe(chosen_format='mean')
        6) Creates a dataframe with the metrics of the models. Partially inherited.
        Used after evaluate_metrics() (5) or apply_and_evaluate_kfolds() (4 & 5)

    Methods inherited from parent class
    -------
    send_pickle(model, path)
        8) Saves model as pickle to the chosen path. This is a static method and theoretically could be used at any time

    split_dataframe(train_num=0.7, random_num=43, scaler=None, return_entire_Xy=False)
        2) Splits dataframe into X_train, X_test, y_train and y_test. 
        Used after adding models to the chosen_models attribute (1)
    
    prepare_models(selected_list=None, excluded_list=None, params_list=None)
        3) Makes models suitable for application. Used after the split (2)
    
    apply_models()
        4) Comes after prepare_models() (3). Predicts using the split dataframe and the prepared models. 
        No folds. Can use apply_and_evaluate_kfolds() (4 & 5) instead
    
    visualize(*metrics_selection)
        7) Creates a lineplot with the metrics of the models. Used after create_dataframe() (6)
    '''

    chosen_models = dict()


    def __init__(self, dataframe, target_name, index=None):
        '''

        Parameters
        ----------
        dataframe : dataframe
            The dataframe to which the models will be used
        target_name : str
            The name of the target column
        index : str (defaut=None)
            The name of the column to become index, if any
        '''
       
        super().__init__(dataframe, target_name, index)
        self.type = 'regression'
        self.progress = f'if you have not already, add some models to the class with {self.type.capitalize()}.add_models(). Then, use split_dataframe() with your instance'


    def __str__ (self):
        '''

        Returns
        -------
        str
           Type of instance and recommended next step
        '''
        
        return super().__str__()


    @classmethod
    def add_models(cls, *regression):
        '''Adds models of the list to the class attribute chosen_models. Required first, before putting the instance to work, since chosen_models is empty by default

        Parameters
        ----------
        regression : str (*args)
            Model names to be put into the chosen_models class attribute
        '''

        for element in regression:
            cls.chosen_models[element] = ''


    @classmethod
    def remove_models(cls, *regression):
        '''Removes models of the list from the class attribute chosen_models

        Parameters
        ----------
        regression : str (*args)
             List of model names to be removed from the chosen_models class attribute
        '''

        for element in regression:
            cls.chosen_models.pop(element)

  
    def apply_and_evaluate_kfolds(self, kfolds_num=5):
        '''Applies models to the dataframe splits using kfolds and evaluates the subsequent metrics.
Can use apply_models() + evaluate_metrics() instead if kfolds are not wanted

        Parameters
        ----------
        kfolds_num : int (default=5)
            Number of folds

        Returns
        -------
        dict
            Dictionary including models and metrics
        '''

        self.kfolds_num = kfolds_num
        self.kfolds = KFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
        self.kfold = 'fold'
        metrics = ['neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'neg_mean_absolute_percentage_error']
        self.models_evaluated = dict()
        print(f'-- {self.type.capitalize()}{self.scaler_name}: using mean of {self.kfolds_num} {self.kfold}s --')
        current_time = time.time()
        total_time = time.time() - current_time
        for model_name, model in self.models.items():
            print(f'Starting {model_name}:')
            start_time = time.time()
            cross_val = cross_validate(model, self.X_train, self.y_train, cv=self.kfolds, return_estimator=True, scoring=metrics)
            list_of_metrics = list(cross_val.keys())[3:]
            self.models_evaluated[model_name] = dict()
            self.models_evaluated[model_name]['models'] = cross_val['estimator']
            self.models_evaluated[model_name]['metrics'] = {'rmse': abs(np.mean(list(cross_val.values())[3:][0])), 
                                                            'mse': abs(np.mean(list(cross_val.values())[3:][1])), 
                                                            'mae': abs(np.mean(list(cross_val.values())[3:][2])), 
                                                            'r2_score': np.mean(list(cross_val.values())[3:][3]), 
                                                            'mape': abs(np.mean(list(cross_val.values())[3:][4]))
                                                            }
            self.models_evaluated[model_name]['all_metrics'] = {'rmse': list(map(abs, list(cross_val.values())[3:][0])), 
                                                                'mse': list(map(abs, list(cross_val.values())[3:][1])), 
                                                                'mae': list(map(abs, list(cross_val.values())[3:][2])), 
                                                                'r2_score': list(map(abs, list(cross_val.values())[3:][3])), 
                                                                'mape': list(map(abs, list(cross_val.values())[3:][4]))
                                                                }
            self.models_evaluated[model_name]['variance'] = {'rmse': np.var(list(cross_val.values())[3:][0]), 
                                                                'mse': np.var(list(cross_val.values())[3:][1]), 
                                                                'mae': np.var(list(cross_val.values())[3:][2]), 
                                                                'r2_score': np.var(list(cross_val.values())[3:][3]), 
                                                                'mape': np.var(list(cross_val.values())[3:][4])
                                                                }
            execution_time = time.time() - start_time
            total_time += execution_time
            print(f'- {model_name} done in {round(execution_time, 2)} sec(s). Total time: {round(total_time, 2)}')
        self.progress = 'your folds have been applied and the metrics evaluated. Put the results into a dataframe with create_dataframe()'
        return self.models_evaluated


    def evaluate_metrics(self):
        '''Extracts the metrics for the models already applied with apply_models(). No kfolds

        Returns
        -------
        dict
           Dictionary with the test, prediction, metrics and model function for each model string
        '''

        self.models_evaluated = self.models.copy()
        for model_name, model_results in self.models_evaluated.items():
            rmse = mean_squared_error(model_results['test'], model_results['prediction'], squared=False)
            mse = mean_squared_error(model_results['test'], model_results['prediction'])
            mae = mean_absolute_error(model_results['test'], model_results['prediction'])
            r2 = r2_score(model_results['test'], model_results['prediction'])
            mape = mean_absolute_percentage_error(model_results['test'], model_results['prediction'])
            self.models_evaluated[model_name]['metrics'] = {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2_score': r2, 'mape': mape}
            self.progress = 'your metrics have been evaluated. Put the results into a dataframe with create_dataframe()'
        return self.models_evaluated


    def create_dataframe(self, chosen_format='mean'):
        '''Creates a dataframe with the metrics of each model.
Used after evaluate_metrics() or apply_and_evaluate_kfolds()

        Parameters
        ----------
        chosen_format : str (default='mean')
            Selected metric format to appear in the resulting dataframe

        Returns
        -------
        dataframe
            Dataframe with models, metrics and BEST/WORST columns
        '''

        self.models_metrics = self.models_evaluated.copy()
        best_values_list = []
        worst_values_list = []
        if chosen_format == 'mean':
            self.chosen_format = 'metrics'
        else:
            self.chosen_format = chosen_format
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name][self.chosen_format]
            if len(self.models_metrics) > 1:
                model_values = [value if type(value) is not list else sum([row[index] for index, row in enumerate(value)]) for value in self.models_evaluated[model_name][self.chosen_format].values()]
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
        super().create_dataframe(best_values_list, worst_values_list)
        self.progress = 'your dataframe with metrics has been created. You may use visualize() to display it graphically'

        return self.df



class Classification(Model):
    '''
    Child class of Model for classification algorithms. To be used directly instead of Model. 
    Methods are numbered by recommended order of use.
    Print instance at any time to see the recommended next step

    ...

    Attributes
    -------
    chosen_models : dict
        Empty dict of models to work with (to be filled with empty keys using add_models class method)

    Class methods
    -------
    add_models(*classification)
        1) Adds models to the class attribute chosen_models. Required first, before putting instances to work, since chosen_models is empty by default.
        It should be called only once, except if more models are needed at some point for further analysis

    remove_models(*classification)
        Removes models from the class attribute chosen_models

    Instance methods
    -------
    apply_and_evaluate_kfolds(kfolds_num=5)
        4 & 5) Used when models are prepared (3). Applies the models to the splits of the dataframe using stratified kfolds and gets the metrics. 
        Can use apply_models() (4) + evaluate_metrics() (5) instead if kfolds are not wanted
    
    evaluate_metrics()
        5) Extracts the metrics for models already applied with apply_models() (4). No folds
    
    create_dataframe(chosen_format='mean')
        6) Creates a dataframe with the metrics of the models. Partially inherited.
        Used after evaluate_metrics() (5) or apply_and_evaluate_kfolds() (4 & 5)

    Methods inherited from parent class
    -------
    send_pickle(model, path)
        8) Saves model as pickle to the chosen path. This is a static method and theoretically could be used at any time

    split_dataframe(train_num=0.7, random_num=43, scaler=None, return_entire_Xy=False)
        2) Splits dataframe into X_train, X_test, y_train and y_test. 
        Used after adding models to the chosen_models attribute (1)
    
    prepare_models(selected_list=None, excluded_list=None, params_list=None)
        3) Makes models suitable for application. Used after the split (2)
    
    apply_models()
        4) Comes after prepare_models() (3). Predicts using the split dataframe and the prepared models. 
        No folds. Can use apply_and_evaluate_kfolds() (4 & 5) instead
    
    visualize(*metrics_selection)
        7) Creates a lineplot with the metrics of the models. Used after create_dataframe() (6)
    '''

    chosen_models = dict()


    def __init__(self, dataframe, target_name, index=None):
        '''

        Parameters
        ----------
        dataframe : dataframe
            The dataframe to which the models will be used
        target_name : str
            The name of the target column
        index : str (defaut=None)
            The name of the column to become index, if any
        '''

        super().__init__(dataframe, target_name, index)
        self.type = 'classification'
        self.progress = f'if you have not already, add some models to the class with {self.type.capitalize()}.add_models(). Then, use split_dataframe() with your instance'


    @classmethod
    def add_models(cls, *classification):
        '''Adds models of the list to the class attribute chosen_models. Required first, before instantiation, since chosen_models is empty by default

        Parameters
        ----------
        classification_list : str (*args)
            Model names to be put into the chosen_models class attribute
        '''

        for element in classification:
            cls.chosen_models[element] = ''


    @classmethod
    def remove_models(cls, *classification):
        '''Removes models of the list from the class attribute chosen_models

        Parameters
        ----------
        classification_list : str (*args)
             Model names to be removed from the chosen_models class attribute
        '''

        for element in classification:
            cls.chosen_models.pop(element)


    def apply_and_evaluate_kfolds(self, kfolds_num=5, multiclass_average=None):
        '''Applies models to the dataframe splits using stratified kfolds and evaluates the subsequent metrics.
Can use apply_models() + evaluate_metrics() instead if kfolds are not wanted

        Parameters
        ----------
        kfolds_num : int (default=5)
            Number of stratified folds
        multiclass_average: str (default=None)
            Type of evaluation, if any. Used generally for multiclass precision, recall and f1 (micro, macro, samples or weighted)

        Returns
        -------
        dict
            Dictionary including models and metrics
        '''

        self.kfolds = StratifiedKFold(n_splits=kfolds_num, shuffle=True, random_state=self.random_num)
        self.kfolds_num = kfolds_num
        self.kfold = 'stratified fold'
        metrics = ['accuracy', 'recall', 'precision', 'f1']
        if multiclass_average == 'micro':
            metrics = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'] 
        elif multiclass_average == 'macro':
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'] 
        elif multiclass_average == 'samples':
            metrics = ['accuracy', 'precision_samples', 'recall_samples', 'f1_samples'] 
        elif multiclass_average == 'weighted':
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'] 
        self.models_evaluated = dict()
        print(f'-- {self.type.capitalize()}{self.scaler_name}: using mean of {self.kfolds_num} {self.kfold}s --')
        current_time = time.time()
        total_time = time.time() - current_time
        for model_name, model in self.models.items():
            print(f'Starting {model_name}:')
            start_time = time.time()
            cross_val = cross_validate(model, self.X_train, self.y_train, cv=self.kfolds, return_estimator=True, scoring=metrics)
            self.models_evaluated[model_name] = dict()
            self.models_evaluated[model_name]['models'] = cross_val['estimator']
            self.models_evaluated[model_name]['metrics'] = {'accuracy': abs(np.mean(list(cross_val.values())[3:][0])), 
                                                            'recall': abs(np.mean(list(cross_val.values())[3:][1])), 
                                                            'precision': abs(np.mean(list(cross_val.values())[3:][2])), 
                                                            'f1_score': np.mean(list(cross_val.values())[3:][3])
                                                            }
            self.models_evaluated[model_name]['all_metrics'] = {'accuracy': list(map(abs, list(cross_val.values())[3:][0])), 
                                                                'recall': list(map(abs, list(cross_val.values())[3:][1])), 
                                                                'precision': list(map(abs, list(cross_val.values())[3:][2])), 
                                                                'f1_score': list(map(abs, list(cross_val.values())[3:][3]))
                                                                }
            self.models_evaluated[model_name]['variance'] = {'accuracy': np.var(list(cross_val.values())[3:][0]), 
                                                                'recall': np.var(list(cross_val.values())[3:][1]), 
                                                                'precision': np.var(list(cross_val.values())[3:][2]), 
                                                                'f1_score': np.var(list(cross_val.values())[3:][3])
                                                                }
            execution_time = time.time() - start_time
            total_time += execution_time
            print(f'- {model_name} done in {round(execution_time, 2)} sec(s). Total time: {round(total_time, 2)}')
        self.progress = 'your folds have been applied and the metrics evaluated. Put the results into a dataframe with create_dataframe()'
        return self.models_evaluated


    def evaluate_metrics(self, **params):
        '''Extracts the metrics for the models already applied with apply_models(). No kfolds

        Parameters
        ----------
        params : dict (**kwargs)
            Parameters to be added for each metric, if any. The key is the metric, and the value is conformed by all the parameters in a single string
            Usually used only if the target column is multiclass

        Returns
        -------
        dict
            Dictionary with the test, prediction, metrics and model function for each model string
        '''

        self.models_evaluated = self.models.copy()
        for model_name, model_results in self.models_evaluated.items():
            accuracy = "accuracy_score (model_results['test'], model_results['prediction']"
            recall = "recall_score (model_results['test'], model_results['prediction']"
            precision = "precision_score (model_results['test'], model_results['prediction']"
            f1 = "f1_score (model_results['test'], model_results['prediction']"
            matrix = "confusion_matrix (model_results['test'], model_results['prediction']"
            list_of_metrics = []
            for index, element in enumerate([accuracy, recall, precision, f1, matrix], 1):
                for key, value in params.items():
                    if key == element.split()[0]:
                        element += ', ' + value + ')'
                if element[-1] == ']':
                    element += ')'
                list_of_metrics.append(eval(element))
            confusion = [element for element in list_of_metrics[-1]]
            self.models_evaluated[model_name]['metrics'] = {'accuracy': list_of_metrics[0], 'recall': list_of_metrics[1], 'precision': list_of_metrics[2], 'f1_score': list_of_metrics[3], 'confusion_matrix': confusion}
        self.progress = 'your metrics have been evaluated. Put the results into a dataframe with create_dataframe()'
        return self.models_evaluated


    def create_dataframe(self, chosen_format='mean'):
        '''Creates a dataframe with the metrics of each model.
Used after evaluate_metrics() or apply_and_evaluate_kfolds()

        Parameters
        ----------
        chosen_format : str (default='mean')
            Selected metric to appear in the resulting dataframe

        Returns
        -------
        dataframe
            Dataframe with models, metrics and BEST/WORST columns
        '''

        self.models_metrics = self.models_evaluated.copy()
        best_values_list = []
        worst_values_list = []
        if chosen_format == 'mean':
            self.chosen_format = 'metrics'
        else:
            self.chosen_format = chosen_format
        for model_name, model_results in self.models_evaluated.items():
            self.models_metrics[model_name] = self.models_metrics[model_name][self.chosen_format]
            if len(self.models_metrics) > 1:
                model_values = [value if type(value) is not list else sum([row[index] for index, row in enumerate(value)]) for value in self.models_evaluated[model_name][self.chosen_format].values()]
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
        super().create_dataframe(best_values_list, worst_values_list)
        self.progress = 'your dataframe with metrics has been created. You may use visualize() to display it graphically'
        return self.df
