import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import datetime as dt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

class DataVisualizer:
    """ Class to visualize data
    """    
    
    def __init__(self, data: pd.DataFrame):
        """Constructor for DataVisualizer

        Parameters
        ----------
        data : pd.DataFrame
            Data to visualize

        Raises
        ------
        ValueError
            If data is not a pandas DataFrame
        """
                
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        
        self.data = data
        
    def plot_missing_values(self):
        """Plot missing values in the dataset
        """
        
        msno.matrix(self.data)
        plt.show()
        
    def plot_distribution(self, columns: list, df: str):
        """Plot distribution of columns in the dataset

        Parameters
        ----------
        columns : list
            List of columns to plot
        df : str
            Name of the dataset

        Raises
        ------
        ValueError
            If columns is not a list
        ValueError
            If df is not a string
        """ 
        
        if not isinstance(columns, list):
            raise ValueError("columns must be a list")
        
        if not isinstance(df, str):
            raise ValueError("df must be a string")
        
        for column in columns:
            if column is 'date_hour':
                pass
            
            if column not in self.data.columns:
                print(f"{column} not in the dataset")
                
            else:
                sns.histplot(self.data[column])
                plt.title(f"Countplot for {column}")
                
                plt.savefig(f'Figures/{df}_countplot_{column}.png')
                
                plt.show()
                
    def plot_correlation(self, df: str, method: str = 'pearson'):
        """Plot correlation heatmap of the dataset

        Parameters
        ----------
        df : str
            Name of the dataset

        Raises
        ------
        ValueError
            If df is not a string
        """ 
        
        if not isinstance(df, str):
            raise ValueError("df must be a string")
        
        corr = self.data.corr(numeric_only=True, method=method)
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap='RdBu')
        plt.title(f"Correlation heatmap for {df} using {method} method")
        
        plt.savefig(f'Figures/{df}_{method}_correlation_heatmap.png')
        
        plt.show()
        
    def lineplot(self, x: str, y: str, title: str, color: str = 'blue', figsize: tuple = (10, 6), path: str = 'Figures/lineplot.png'):
        """Plot a lineplot for the given x and y columns

        Parameters
        ----------
        x : str
            Column for the x-axis
        y : str
            Column for the y-axis
        title : str
            Title of the plot
        color : str, optional
            Line color (default is 'blue')
        figsize : tuple, optional
            Figure size (default is (10, 6))
        path : str, optional
            Path to save the plot (default is 'Figures/lineplot.png')

        Raises
        ------
        ValueError
            If x or y is not a string
        ValueError
            If title is not a string
        KeyError
            If x or y is not in the dataset
        """ 
        
        if not isinstance(x, str):
            raise ValueError("x must be a string")
        
        if not isinstance(y, str):
            raise ValueError("y must be a string")
        
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        
        if x not in self.data.columns:
            raise KeyError(f"{x} is not a column in the dataset")
        
        if y not in self.data.columns:
            raise KeyError(f"{y} is not a column in the dataset")
        
        plt.figure(figsize=figsize)
        sns.lineplot(x=x, y=y, data=self.data, color=color)
        plt.title(title)
        plt.grid()
        plt.savefig(path)
        plt.show()
            
class TimeSeriesDecomposer:
    def __init__(self, series: pd.Series, period: int = 12):
        """Constructor for TimeSeriesDecomposer

        Parameters
        ----------
        series : pd.Series
            Time series data
        period : int, optional
            Period used for decomposing data, by default 12
            
        Raises
        ------
        ValueError
            If series is not a pandas Series
        ValueError
            If period is not an integer
        """
                
        if not isinstance(series, pd.core.series.Series):
            raise ValueError("series must be a pandas Series")
        
        if not isinstance(period, int):
            raise ValueError("period must be an integer")
                
        self.series = series
        self.period = period
        self.trend = None
        self.seasonal = None
        self.residual = None

    def decompose(self):
        """Decompose the time series into trend, seasonal, and residual components.

        Returns
        -------
        trend : pd.Series
            Series with the trend data.
        seasonal : pd.Series
            Series with the seasonal data.
        residual : pd.Series
            Series with the residual data.
        """
                
        decomposition = seasonal_decompose(self.series, period=self.period)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        return self.trend, self.seasonal, self.residual

    def plot_decomposition(self, trend: pd.Series = None, seasonal: pd.Series = None, residual: pd.Series = None, title: str = 'Decomposition of Time Series', save_path: str = 'Figures/decomposition.png'):
        """Plot the decomposition of the time series.

        Parameters
        ----------
        trend : pd.Series, optional
            Series with the trend data, by default None
        seasonal : pd.Series, optional
            Series with the seasonal data, by default None
        residual : pd.Series, optional
            Series with the residual data, by default None
        title : str, optional
            Title of the plot, by default 'Decomposition of Time Series'
        save_path : str, optional
            File location and name, by default 'Figures/decomposition.png'

        Raises
        ------
        ValueError
            If trend is not a pandas Series
        ValueError
            If seasonal is not a pandas Series
        ValueError
            If residual is not a pandas Series
        ValueError
            If title is not a string
        ValueError
            If save_path is not a string
        """
                        
        if trend is not None:
            if not isinstance(trend, pd.core.series.Series):
                raise ValueError("trend must be a pandas Series")
            self.trend = trend
        if seasonal is not None:
            if not isinstance(seasonal, pd.core.series.Series):
                raise ValueError("seasonal must be a pandas Series")
            self.seasonal = seasonal
        if residual is not None:
            if not isinstance(residual, pd.core.series.Series):
                raise ValueError("residual must be a pandas Series")
            self.residual = residual
        
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        
        if not isinstance(save_path, str):
            raise ValueError("save_path must be a string")
        
        fig, ax = plt.subplots(3, 1, figsize=(15, 10))

        ax[0].plot(self.series, label='Original Series', color='blue')
        ax[0].plot(self.trend, label='Trend', color='orange')
        ax[0].set_title('Original Series and Trend')
        ax[0].legend(loc='best')

        ax[1].plot(self.seasonal, label='Seasonal', color='green')
        ax[1].set_title('Seasonal')
        ax[1].legend(loc='best')

        ax[2].plot(self.residual, label='Residual', color='red')
        ax[2].set_title('Residual')
        ax[2].legend(loc='best')

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

def create_timeseries_features(data: pd.DataFrame):
    """Create features from the date_hour column

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the date_hour column

    Returns
    -------
    data : pd.DataFrame
        Dataframe with new features
    
    Raises
    ------
    ValueError
        If data is not a pandas DataFrame
    """    
    
    if not isinstance(data, pd.core.frame.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
        
    data['date_hour'] = pd.to_datetime(data['date_hour'])
    data['year'] = data['date_hour'].dt.year
    data['month'] = data['date_hour'].dt.month
    data['week'] = data['date_hour'].dt.week
    data['day'] = data['date_hour'].dt.day
    data['hour'] = data['date_hour'].dt.hour
    data['day_of_week'] = data['date_hour'].dt.dayofweek
    return data

class StatisticalTests:
    def __init__(self, data: pd.DataFrame):
        """Constructor for DataInspector

        Parameters
        ----------
        data : pd.DataFrame
            Data to inspect

        Raises
        ------
        ValueError
            If data is not a pandas DataFrame
        """
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        
        self.data = data
        
    def stationary_test(self, col_name: str, alpha: float = 0.05):
        """Perform the Augmented Dickey-Fuller test

        Parameters
        ----------
        col_name : str
            Name of the column to test
        alpha : float, optional
            Significance level, by default 0.05

        Raises
        ------
        ValueError
            If col_name is not a string
        ValueError
            If alpha is not a float
        KeyError
            If col_name is not in the dataset
        """
        
        if not isinstance(col_name, str):
            raise ValueError("col_name must be a string")
        
        if not isinstance(alpha, float):
            raise ValueError("alpha must be a float")
        
        if col_name not in self.data.columns:
            raise KeyError(f"{col_name} is not a column in the dataset")
        
        result = adfuller(self.data[col_name])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f'Critical Values: {result[4]}')
        
        if result[1] < alpha:
            print("\nReject the null hypothesis, the data is stationary")
        else:
            print("\nFail to reject the null hypothesis, the data is non-stationary")
            
    def fourier_analysis(self, col_name: str, height: int | float = 50):
        """Perform Fourier analysis on the data

        Parameters
        ----------
        col_name : str
            Name of the column to analyze
        height : int or float, optional
            Minimum height for peaks, by default 20

        Raises
        ------
        ValueError
            If col_name is not a string
        ValueError
            If height is not an integer
        KeyError
            If col_name is not in the dataset
        """
        
        if not isinstance(col_name, str):
            raise ValueError("col_name must be a string")
        
        if not isinstance(height, int) and not isinstance(height, float):
            raise ValueError("n must be an integer")
        
        if col_name not in self.data.columns:
            raise KeyError(f"{col_name} is not a column in the dataset")
        
        N = len(self.data[col_name])
        
        data = self.data[col_name].values
        self.fft_values = fft(data)
        
        magnitude = 2.0/N * np.abs(self.fft_values[:N//2])
        
        self.peaks, _ = find_peaks(magnitude, height=height)
        
        print(f'Peaks found at: {self.peaks}\nWith amplitude: {magnitude[self.peaks]}')
        
        plt.plot(magnitude)
        plt.xlabel('Frequency (1/N)')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform')
        
        plt.ylim(0, 90)
        
        plt.plot(self.peaks, magnitude[self.peaks], "x", markersize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('Figures/fourier_transform.png')
        plt.show()
        
    def inverted_fourier_transform(self, col_name: str):
        """Perform inverted Fourier transform on the data
        
        Parameters
        ----------
        col_name : str
            Name of the column to analyze
            
        Raises
        ------
        ValueError
            If col_name is not a string
        KeyError
            If col_name is not in the dataset
        """
        
        if not isinstance(col_name, str):
            raise ValueError("col_name must be a string")
        
        if col_name not in self.data.columns:
            raise KeyError(f"{col_name} is not a column in the dataset")
        
        self.fft_values[self.peaks] = 0
        
        ifft_values = ifft(self.fft_values)
        
        plt.plot(ifft_values)
        plt.title('Inverted Fourier Transform')
        
        plt.tight_layout()
        plt.savefig('Figures/inverted_fourier_transform.png')
        plt.show()
        
class GridSearch:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, target: str, model: object, param_grid: dict, n_splits: int = 5, split_size: int = None, scoring: str = 'neg_root_mean_squared_error', order: int = None): 
        """Constructor for GridSearch
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Testing data
        target : str
            Name of the target column
        model : object
            Model to use
        param_grid : dict
            Dictionary of hyperparameters
        n_splits : int, optional
            Number of splits, by default 5
        split_size : int, optional
            Size of the test set, by default len(test_data)
        scoring : str, optional
            Scoring metric to use, by default 'neg_root_mean_squared_error'
        order : int, optional
            Order of the time series, by default None
            
        Raises
        ------
        ValueError
            If train_data or test_data is not a pandas DataFrame
        ValueError
            If target is not a string
        ValueError
            If n_splits is not an integer
        ValueError
            If split_size is not an integer
        ValueError
            If scoring is not a string
        """
        
        if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
            raise ValueError("train_data and test_data must be pandas DataFrames")
        
        if not isinstance(target, str):
            raise ValueError("target must be a string")
        
        if not isinstance(n_splits, int):
            raise ValueError("n_splits must be an integer")
        
        if split_size is not None and not isinstance(split_size, int):
            raise ValueError("split_size must be an integer")
        
        if not isinstance(scoring, str):
            raise ValueError("scoring must be a string")
        
        if order is not None and not isinstance(order, int):
            raise ValueError("order must be an integer")
        
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.X_train = train_data.drop(target, axis=1)
        self.y_train = train_data[target]
        self.model = model
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.split_size = split_size if split_size is not None else len(test_data)
        self.scoring = scoring
        
        if order is not None:
            dp = DeterministicProcess(index=self.y_train.index, constant=False, order=order, drop=True)
            self.X_train_con = dp.in_sample()
            self.test_data_con = dp.out_of_sample(forecast_index=self.test_data.index, steps=len(self.test_data))
            self.X_train = pd.concat([self.X_train, self.X_train_con], axis=1)
            self.test_data = pd.concat([self.test_data, self.test_data_con], axis=1)
        
        self.tscv = CustomTimeSeriesSplit(n_splits=self.n_splits, test_size=self.split_size)
        
    def fit(self):
        """Fit the model
        
        Returns
        -------
        tuple
            Best hyperparameters and best score
        """
        
        grid = GridSearchCV(self.model, 
                            param_grid=self.param_grid, 
                            cv=self.tscv, 
                            scoring=self.scoring, 
                            n_jobs=-1)
        
        grid.fit(self.X_train, self.y_train)
        print(f'Best parameters: {grid.best_params_}, Best score: {grid.best_score_}')
        
        self.best_estimator = grid.best_estimator_
        
        return grid.best_params_, grid.best_score_
    
    def predict(self, to_pred_col: list):
        """Predict on the test data
        
        Parameters
        ----------
        to_pred_col : list
            List of indeces to predict
            
        Returns
        -------
        df_preds : pd.DataFrame
            Dataframe with predictions
        """
        
        self.predictions = self.best_estimator.predict(self.test_data)
        
        self.df_preds = pd.DataFrame({'date_hour': to_pred_col, 'cnt': self.predictions}, index=self.test_data.index)
        return self.df_preds
    
    def to_csv(self, model: str, path_add: str = None):
        """Write predictions to a CSV file
        
        Parameters
        ----------
        model : str
            Name of the model
        path_add : str, optional
            Addition to path to save the file, by default None
            
        Raises
        ------
        ValueError
            If model is not a string
        ValueError
            If path is not a string
        """
        
        if not isinstance(model, str):
            raise ValueError("model must be a string")
        
        if not isinstance(path_add, str) and path_add is not None:
            raise ValueError("path must be a string")
        
        path = f'Predictions/{model}_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}_{path_add}.csv' if path_add is not None else f'Predictions/{model}_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        
        self.df_preds.to_csv(path, index=False)
        
class CustomTimeSeriesSplit(TimeSeriesSplit):
    def __init__(self, n_splits, test_size):
        super().__init__(n_splits=n_splits)
        self.test_size = test_size
        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        
        n_samples = X.shape[0]
        n_splits = self.n_splits
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        
        test_starts = range(n_samples - n_splits*self.test_size, n_samples, self.test_size)
        
        for test_start in test_starts:
            yield (indices[:test_start-self.test_size], indices[test_start:test_start+self.test_size])
            
class SARIMAXModel:
    def __init__(self, data: pd.Series, param_grid: dict, trend='n'):
        """
        Constructor for SARIMAXModel.
        
        Parameters
        ----------
        data : pd.Series
            Data to fit the model.
        param_grid : dict
            Dictionary of SARIMAX parameters to search.
        trend : str, optional
            Type of trend to use, by default 'n'.
            
        Raises
        ------
        ValueError
            If data is not a pandas Series.
        ValueError
            If param_grid is not a dictionary.
        """
        
        if not isinstance(data, pd.Series):
            raise ValueError("data must be a pandas Series")
        
        if not isinstance(param_grid, dict):
            raise ValueError("param_grid must be a dictionary")
        
        self.data = data
        self.param_grid = param_grid
        self.trend = trend
        self.best_score = float('inf')
        self.best_params = None
        self.best_model = None
    
    def grid_search(self):
        """
        Perform grid search to find the best SARIMAX parameters.
        """
        
        grid = ParameterGrid(self.param_grid)

        for params in tqdm(grid, desc="GridSearch iterations"):
            try:
                model = SARIMAX(self.data, order=params['order'], seasonal_order=params['seasonal_order'], trend=self.trend)
                model_fit = model.fit(disp=False)

                predictions = model_fit.fittedvalues
                rmse = np.sqrt(mean_squared_error(self.data, predictions))

                if rmse < self.best_score:
                    self.best_score = rmse
                    self.best_params = params
                    self.best_model = model_fit
            except Exception as e:
                print(f'Error with {params}: {e}')

        print(f'Best params: {self.best_params}')
        print(f'Best score: {self.best_score}')
        
    def fit_final_model(self):
        """
        Fit the final SARIMAX model with the best found parameters.
        """
        if self.best_params is None:
            print("No best parameters found. Run grid search first.")
            return None

        final_model = SARIMAX(self.data, order=self.best_params['order'], seasonal_order=self.best_params['seasonal_order'], trend=self.trend)
        final_model_fit = final_model.fit(disp=False)
        self.best_model = final_model_fit

        return final_model_fit
    
    def predict(self, start_date: pd.Timestamp, end_date: pd.Timestamp, test_data_pred_col: list):
        """
        Predict on the test data.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start date of the test data.
        end_date : pd.Timestamp
            End date of the test data.
        test_data_pred_col : list
            List of indeces to predict.
        
        Raises
        ------
        ValueError
            If start_date is not a pandas Timestamp.
        ValueError
            If end_date is not a pandas Timestamp.
        ValueError
            If test_data_pred_col is not a list.
        """
        
        if not isinstance(start_date, pd.Timestamp):
            raise ValueError("start_date must be a pandas Timestamp")
        
        if not isinstance(end_date, pd.Timestamp):
            raise ValueError("end_date must be a pandas Timestamp")
        
        if not isinstance(test_data_pred_col, list):
            raise ValueError("test_data_pred_col must be a list")
        
        if self.best_model is None:
            print("Model is not fitted yet.")
            return None

        final_pred = self.best_model.predict(start=start_date, end=end_date)
        self.df_preds = pd.DataFrame({'date_hour': test_data_pred_col, 'cnt': final_pred}, index=test_data_pred_col.index)

    def save_predictions(self):
        """
        Save the predictions to a CSV file.
        """
        self.df_preds.to_csv(f'Predictions/SARIMAX_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)