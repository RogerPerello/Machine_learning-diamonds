import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Cleansing:

    def __init__(self, dataframes, target_name, index=None):
        self.train = dataframes[0]
        self.test = dataframes[1]
        self.target_name = target_name
        self.target = self.train[target_name]
        if index:
            self.train = self.train.set_index(index)
            self.test = self.test.set_index(index)

    def impute_boxplot_min_max(self, list_of_columns, min=True, max=True):
        '''Imputes the outliers of a boxplot for the chosen columns to its min and max values'''
        for column in list_of_columns:
            q3, q1 = np.percentile(self.train[column], [75, 25])
            iqr = q3 - q1
            if min:
                self.train.loc[self.train[column] < q1 - 1.5*iqr, column] = q1 - 1.5*iqr
            if max:
                self.train.loc[self.train[column] > q3 + 1.5*iqr, column] = q3 + 1.5*iqr


    def remove_elements(self, conditioned_columns_list, condition, number):
        '''Removes the rows of a dataframe based on a condition'''
        for column in conditioned_columns_list:
            if condition == 'equal':
                self.train.drop(self.train[(self.train[column] == number)].index, inplace=True)
            elif condition == 'bigger':
                self.train.drop(self.train[(self.train[column] > number)].index, inplace=True)
            elif condition == 'bigger_or_equal':
                self.train.drop(self.train[(self.train[column] >= number)].index, inplace=True)   
            elif condition == 'smaller':
                self.train.drop(self.train[(self.train[column] < number)].index, inplace=True)   
            elif condition == 'smaller_or_equal':
                self.train.drop(self.train[(self.train[column] <= number)].index, inplace=True)


    def apply_scalar(self, method, list_of_columns=None):
        '''Applies the selected scalar method to a list of train and test dataframes for the chosen columns'''
        if method == 'log' and list_of_columns:
            for df in (self.train, self.test):
                for column in list_of_columns:
                    df[column] = np.log(df[column])
        elif method == 'standard':
            scaler = StandardScaler().fit(self.train.values)
            scaled_test = self.test.join(self.target)
            for df in (self.train, scaled_test):
                df.loc[:, :] = scaler.transform(df.values)
            self.test = scaled_test.drop(columns=self.target_name)
        elif method == 'minmax':
            scaler = MinMaxScaler().fit(self.train.values)
            scaled_test = self.test.join(self.target)
            for df in (self.train, scaled_test):
                df.loc[:, :] = scaler.transform(df.values)
            self.test = scaled_test.drop(columns=self.target_name)
        print(f'{method.capitalize()} scaler completed')
