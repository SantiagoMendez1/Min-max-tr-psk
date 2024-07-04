# Importing libraries
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import json
import os
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Hyperparametrization
from sklearn.model_selection import GridSearchCV


# Models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb 
from lightgbm import LGBMRegressor 

# Metrics
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

import pickle
import time
import warnings
warnings.filterwarnings("ignore")

def set_seeds(seed=42):
    """
    Set seeds for reproducibility across numpy, random, and TensorFl ow.

    Args:
        seed (int, optional): The seed value to set for all random number generators.
                              Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

class DataAnalyzer:
    def __init__(self, file_path):
        """
        Initializes an instance of the class with a CSV file.

        Args:
            file_path (str): Relative or absolute path of CSV file (stock_move.csv)
        """
        try:
            self.USER_PATH = '/Users/Camilo/Documents/presik/DB_MINMAXTODOREP'
            self.data = pd.read_csv(self.USER_PATH + file_path)            
        except FileNotFoundError:
            print(f'No se pudo encontrar el archivo: {file_path}')
            self.data = pd.DataFrame()
    
    def pre_processing(self):
        """
        Preprocess the input data for time series analysis.
        """
        try:
            try:
                self.data['effective_date'] = pd.to_datetime(self.data['effective_date'])
            except KeyError as e:
                print(f"KeyError: {e}. Ensure 'effective_date' is a column in the data.")
                return None
            except Exception as e:
                print(f"Error converting 'effective_date' to datetime: {e}")
                return None

            try:
                self.data = self.data.sort_values(by='effective_date')
                most_selled_products = self.data['product'].value_counts().reset_index()
                most_selled_products.columns = ['product', 'count']
                most_selled_products = most_selled_products[most_selled_products['count'] > 500]
                self.filter_products = list(most_selled_products['product'])
            except KeyError as e:
                print(f"KeyError: {e}. Ensure 'product' and 'quantity' are columns in the data.")
                return None
            except Exception as e:
                print(f"Error processing most sold products: {e}")
                return None

            try:
                filtered_df = self.data[self.data['product'].isin(self.filter_products)]
                pivot_df = filtered_df.pivot_table(index='effective_date', columns='product', values='quantity', aggfunc='sum', fill_value=0)
                pivot_df = pivot_df.reindex(columns=self.filter_products)
            except KeyError as e:
                print(f"KeyError: {e}. Ensure 'effective_date', 'product', and 'quantity' are columns in the data.")
                return None
            except Exception as e:
                print(f"Error creating pivot table: {e}")
                return None

            try:
                compl_date_day = pd.date_range(start=pivot_df.index.min(), end=pivot_df.index.max(), freq='D')
                df_compl_day = pd.DataFrame({'effective_date': compl_date_day})
                one_prod_df_all_date = pd.merge(df_compl_day, pivot_df, on='effective_date', how='left').fillna(0)
                one_prod_df_all_date.reset_index(drop=True)
            except Exception as e:
                print(f"Error merging and filling dates: {e}")
                return None

            try:
                df_daily = one_prod_df_all_date.groupby('effective_date').sum().reset_index()
                df_daily['effective_date'] = pd.to_datetime(df_daily['effective_date'])
                df_daily.set_index('effective_date', inplace=True)
            except Exception as e:
                print(f"Error processing daily data: {e}")
                return None

            try:
                df_weekly = df_daily.resample('W').sum()
                df_weekly['year'] = df_weekly.index.year
                df_weekly['week'] = df_weekly.index.isocalendar().week
            except Exception as e:
                print(f"Error resampling data weekly: {e}")
                return None

            try:
                split = round(len(df_weekly) * 90 / 100)
                self.train = df_weekly.iloc[:split]
                self.test = df_weekly.iloc[split:]
            except Exception as e:
                print(f"Error splitting the data into train and test sets: {e}")
                return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def model_hyperparam_tuning(self, run=False):
        """
        Perform hyperparameter tuning for a RandomForestRegressor model.
        
        Args:
        run (bool): Flag to indicate whether to perform hyperparameter tuning or not. If False, attempts to load from file.
        """
        out_filename = f"{self.USER_PATH}/model/all_best_params_rf.json"
        products_to_train = self.filter_products[0:3]

        if run is False:
            try:
                print(f"reading {out_filename}")
                with open(out_filename,"r") as f:
                    self.all_best_params = json.load(f)
            except:
                run is True

        if run:
            self.all_best_params = {}
           
            for t in products_to_train:
                X_train = self.train[['week', 'year']]
                y_train = self.train[t]

                X_test = self.test[['week', 'year']]
                y_test = self.test[t]

                param_grid = {'n_estimators': np.arange(400, 2200, 200),
                              'max_depth': [10, 15, 18, 20],
                              'max_features': ['sqrt', 'log2'],
                              'min_samples_split': [2, 5, 8, 10]}

                random_forest_model = RandomForestRegressor(random_state=42)

                grid_search = GridSearchCV(estimator=random_forest_model,
                                           param_grid=param_grid, 
                                           cv=5, 
                                           n_jobs=-1, 
                                           scoring='neg_mean_squared_error')

                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                best_model = grid_search.best_estimator_

                for key, value in best_params.items():
                    if isinstance(value, np.int32):
                        best_params[key] = int(value)
                    elif isinstance(value, np.float32):
                        best_params[key] = float(value)

                self.all_best_params[t] = best_params

                
            with open(out_filename, 'w') as json_file:
                json.dump(self.all_best_params, json_file)
                #with open(f'{self.USER_PATH}/model/all_best_params_rf.json', 'w') as json_file:
                #    json.dump(self.all_best_params, json_file)

            train_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            doc_path = f'{self.USER_PATH}/model/train.txt'
            doc_data = f'TRAIN DATE:\n{train_date}\n\nTRAINED PRODUCTS:\n{products_to_train}'

            with open(doc_path, 'w') as file:
                file.write(doc_data)
                
        return products_to_train

    def prediction(self, product, months_to_predict):
        """
        Predict future quantities for a given product using a trained RandomForestRegressor model.
        
        Args:
        product (int): The product ID for which predictions are to be made.
        months_to_predict (int): The number of months into the future for which predictions are needed.

        Returns:
        df_pred_monthly: A dataframe with the predictions of the months_to_predict.
        """
        self.product = product
        self.months_to_predict = months_to_predict
        best_params = self.all_best_params[str(self.product)]
        
        X_train = self.train[['week', 'year']]
        y_train = self.train[self.product]

        X_test = self.test[['week', 'year']]
        y_test = self.test[self.product]

        y_pred = y_test.copy()
        last_date = y_pred.index[-1]
        final_date = last_date + pd.DateOffset(months=self.months_to_predict)
        new_dates = pd.date_range(start=last_date, end=final_date, freq='W-SUN')[1:]
        df_new_dates = pd.DataFrame(index=new_dates)
        df_new_dates['week'] = df_new_dates.index.isocalendar().week
        df_new_dates['year'] = df_new_dates.index.year
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        self.y_hat_test = np.round(np.abs(best_model.predict(X_test)))
        rmse_test = np.sqrt(mean_squared_error(y_pred, self.y_hat_test))
        sigma = np.abs(y_pred - self.y_hat_test).std()
        y_pred = pd.DataFrame({'Quantity': y_pred}, index=y_test.index)
        y_pred['Quantity Prediction'] = self.y_hat_test
        self.y_pred = y_pred.resample('M').sum()
        self.y_pred['Lower Quantity'] = np.round(self.y_pred['Quantity Prediction'] - 1.96 * sigma).astype(int)
        self.y_pred['Higher Quantity'] = np.round(self.y_pred['Quantity Prediction'] + 1.96 * sigma).astype(int)

        y_hat = np.round(np.abs(best_model.predict(df_new_dates[['week', 'year']]))).astype(int)
        df_new_dates['rf_prediction'] = y_hat
        self.df_pred_monthly = df_new_dates['rf_prediction'].resample('M').sum()
        self.df_pred_monthly = pd.DataFrame({'Quantity Prediction':self.df_pred_monthly}, index=self.df_pred_monthly.index)
        self.df_pred_monthly.index.name = 'Date' 
        
        self.df_pred_monthly['Lower Quantity'] = np.round(self.df_pred_monthly['Quantity Prediction'] - 1.96 * sigma).astype(int)
        self.df_pred_monthly['Higher Quantity'] = np.round(self.df_pred_monthly['Quantity Prediction'] + 1.96 * sigma).astype(int)
        self.df_pred_monthly = self.df_pred_monthly.iloc[:-1]

        return self.df_pred_monthly
    
    def plot_test(self):
        """
        Plot the test data and predictions with a 95% confidence interval.

        Args:
        product (int): The identifier of the product for which the predictions are plotted.
        """
        try:
            fig, ax = plt.subplots(figsize=(15, 6))

            ax.plot(self.y_pred.index, 
                    self.y_pred['Quantity'], 
                    color='blue', 
                    label='Test Data')
            
            ax.plot(self.y_pred.index, 
                    self.y_pred['Quantity Prediction'], 
                    color='red', 
                    label='Prediction')

            ax.fill_between(self.y_pred.index, 
                            self.y_pred['Lower Quantity'], 
                            self.y_pred['Higher Quantity'],
                            color='gray',
                            alpha=0.2,
                            label='95% confidence interval')
            
            for i, (date, upper) in enumerate(zip(self.y_pred.index, self.y_pred['Quantity'])):
                ax.annotate(f'{upper:.2f}', 
                            xy=(date, upper), 
                            xytext=(3, 15), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='blue')

            for i, (date, pred) in enumerate(zip(self.y_pred.index, self.y_pred['Quantity Prediction'])):
                ax.annotate(f'{pred:.2f}', 
                            xy=(date, pred), 
                            xytext=(3, 3), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='red')

            for i, (date, lower) in enumerate(zip(self.y_pred.index, self.y_pred['Lower Quantity'])):
                ax.annotate(f'{lower:.2f}', 
                            xy=(date, lower), 
                            xytext=(3, -15), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='gray')

            for i, (date, upper) in enumerate(zip(self.y_pred.index, self.y_pred['Higher Quantity'])):
                ax.annotate(f'{upper:.2f}', 
                            xy=(date, upper), 
                            xytext=(3, 15), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='gray')

            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity Prediction')
            plt.xticks(rotation=45)
            ax.legend()
            ax.set_title(f'Prediction for test data of the {self.product}')
            ax.grid(True)
            plt.tight_layout()

            try:
                plt.savefig(f'{self.USER_PATH}/model/rf_prediction_{str(self.product)}_test.png')
            except IOError as e:
                print(f"Error saving plot: {e}")
            
            plt.close(fig) 

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def plot_months_to_predict(self):
        """
        Plot the predicted quantities for a given number of months with a 95% confidence interval.

        Args:
        product (int): The identifier of the product for which the predictions are plotted.
        """
        try:
            fig, ax = plt.subplots(figsize=(15, 6))

            ax.plot(self.df_pred_monthly.index, 
                    self.df_pred_monthly['Quantity Prediction'], 
                    color='red', 
                    label='Prediction')

            ax.fill_between(self.df_pred_monthly.index, 
                            self.df_pred_monthly['Lower Quantity'], 
                            self.df_pred_monthly['Higher Quantity'],
                            color='gray',
                            alpha=0.2,
                            label='95% confidence interval')

            for i, (date, pred) in enumerate(zip(self.df_pred_monthly.index, self.df_pred_monthly['Quantity Prediction'])):
                ax.annotate(f'{pred:.2f}', 
                            xy=(date, pred), 
                            xytext=(3, 3), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='red')

            for i, (date, lower) in enumerate(zip(self.df_pred_monthly.index, self.df_pred_monthly['Lower Quantity'])):
                ax.annotate(f'{lower:.2f}', 
                            xy=(date, lower), 
                            xytext=(3, -15), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='gray')

            for i, (date, upper) in enumerate(zip(self.df_pred_monthly.index, self.df_pred_monthly['Higher Quantity'])):
                ax.annotate(f'{upper:.2f}', 
                            xy=(date, upper), 
                            xytext=(3, 15), 
                            textcoords='offset points', 
                            ha='center', 
                            fontsize=12, 
                            color='gray')

            ax.set_xlabel('Date')
            ax.set_ylabel('Quantity Prediction')
            plt.xticks(rotation=45)
            ax.legend()
            ax.set_title(f'Prediction for the {self.product}')
            ax.grid(True)
            plt.tight_layout()

            try:
                plt.savefig(f'{self.USER_PATH}/model/rf_prediction_{str(self.product)}_{str(self.months_to_predict)}_months.png')
            except IOError as e:
                print(f"Error saving plot: {e}")
            
            plt.close(fig)  

        except KeyError as e:
            print(f"Key error: {e}. Please check if the necessary columns are present in the dataframe.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
