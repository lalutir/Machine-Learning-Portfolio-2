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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
            print("Reject the null hypothesis, the data is stationary")
        else:
            print("Fail to reject the null hypothesis, the data is non-stationary")
            
    def fourier_analysis(self, col_name: str, height: int = 20):
        """Perform Fourier analysis on the data

        Parameters
        ----------
        col_name : str
            Name of the column to analyze
        height : int, optional
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
        
        if not isinstance(height, int):
            raise ValueError("n must be an integer")
        
        if col_name not in self.data.columns:
            raise KeyError(f"{col_name} is not a column in the dataset")
        
        N = len(self.data[col_name])
        
        data = self.data[col_name].values
        fft_values = fft(data)
        
        magnitude = 2.0/N * np.abs(fft_values[:N//2])
        
        peaks, _ = find_peaks(magnitude, height=height)
        
        print(f'Peaks found at: {peaks}\nWith amplitude: {magnitude[peaks]}')
        
        plt.plot(magnitude)
        plt.xlabel('Frequency (1/N)')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform')
        
        plt.ylim(0, 90)
        
        plt.plot(peaks, magnitude[peaks], "x", markersize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('Figures/fourier_transform.png')
        plt.show()

class TimeSeriesModel:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, test_size=None, models=None):
        """Constructor for TimeSeriesModel
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Testing data
        test_size : int, optional
            Size of the test set, by default len(test_data)
        models : dict, optional
            Dictionary of models to use, by default {'dt': (DecisionTreeRegressor(), {})}
        
        Raises
        ------
        ValueError
            If train_data or test_data is not a pandas DataFrame
        """
        
        if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
            raise ValueError("train_data and test_data must be pandas DataFrames")
        
        self.train_data = train_data
        self.test_data = test_data
        self.test_size = test_size if test_size is not None else len(test_data)
        self.models = models if models is not None else {'dt': (DecisionTreeRegressor(), {})}

    def time_series_data(self, order: int):
        """Generate time series data for training and testing
        
        Parameters
        ----------
        order : int
            Order of the deterministic process
        
        Returns
        -------
        tuple
            Train and test data with deterministic process applied
            
        Raises
        ------
        ValueError
            If order is not an integer
        """
        
        if not isinstance(order, int):
            raise ValueError("order must be an integer")
        
        dp = DeterministicProcess(index=self.train_data.index, constant=False, order=order, drop=True)
        train = dp.in_sample()
        test = dp.out_of_sample(forecast_index=self.test_data.index, steps=len(self.test_data))
        train = pd.concat([train, self.train_data], axis=1)
        test = pd.concat([test, self.test_data], axis=1)
        self.new_train_data = train
        self.new_test_data = test
        return train, test

    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = None, scoring: str = 'neg_root_mean_squared_error'):
        """Perform grid search on the models
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training data
        y_train : pd.Series
            Target data
        n_splits : int, optional
            Number of splits for cross-validation, by default len(self.train_data) - len(self.test_data)
        scoring : str, optional
            Scoring metric to use, by default 'neg_root_mean_squared_error'
            
        Raises
        ------
        ValueError
            If X_train is not a pandas DataFrame
        ValueError
            If y_train is not a pandas Series
        ValueError
            If n_splits is not an integer
        ValueError
            If scoring is not a string
        """
        
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame")
        
        if not isinstance(y_train, pd.core.series.Series):
            raise ValueError("y_train must be a pandas Series")
        
        if n_splits is not None and not isinstance(n_splits, int):
            raise ValueError("n_splits must be an integer")
        
        if not isinstance(scoring, str):
            raise ValueError("scoring must be a string")
        
        if n_splits is None:
            n_splits = len(self.train_data) - len(self.test_data)
        
        self.best_params = {}
        self.feature_importances = {}
        
        kf = TimeSeriesSplit(n_splits=n_splits)
        
        for model_name, model in self.models.items():
            grid = GridSearchCV(model[0], param_grid=model[1], cv=kf, scoring=scoring)
            grid.fit(X_train, y_train)
            print(f'{model_name} - Best Params: {grid.best_params_} - Best Score: {grid.best_score_}')
            self.best_params[model_name] = (grid.best_params_, grid.best_score_)
            
            if hasattr(grid.best_estimator_, 'feature_importances_'):
                self.feature_importances[model_name] = grid.best_estimator_.feature_importances_
                
    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
        """Fit the models and make predictions
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training data
        y_train : pd.Series
            Target data
        X_test : pd.DataFrame
            Testing data
            
        Raises
        ------
        ValueError
            If X_train is not a pandas DataFrame
        ValueError
            If y_train is not a pandas Series
        ValueError
            If X_test is not a pandas DataFrame
        """
        
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame")
        
        if not isinstance(y_train, pd.core.series.Series):
            raise ValueError("y_train must be a pandas Series")
        
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame")
        
        self.predictions = {}
        
        for model_name, model in self.models.items():
            model[0].set_params(**self.best_params[model_name][0])
            model[0].fit(X_train, y_train)
            self.predictions[model_name] = model[0].predict(X_test)
        
    def write_to_csv(self, pred_col: list):
        """Write predictions to a CSV file
        
        Parameters
        ----------
        pred_col : list
            List of dates for the predictions
            
        Raises
        ------
        ValueError
            If pred_col is not a list
        """
        
        if not isinstance(pred_col, list):
            raise ValueError("pred_col must be a list")
        
        for model_name, preds in self.predictions.items():
            df = pd.DataFrame({'date_hour': pred_col, 'cnt': preds}, index=self.test_data.index)
            df.to_csv(f'Predictions/{model_name}_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    
    def plot_feature_importances(self):
        """Plot feature importances
        
        Raises
        ------
        ValueError
            If feature_importances is not available
        """
        
        if not hasattr(self, 'feature_importances'):
            raise ValueError("Feature importances are not available")
        
        for model_name, importances in self.feature_importances.items():
            plt.figure(figsize=(10, 6))
            plt.barh(self.X_train.columns, importances)
            plt.title(f'Feature Importances for {model_name}')
            plt.savefig(f'Figures/feature_importances_{model_name}.png')
            plt.show()