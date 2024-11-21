import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.fft import fft, ifft
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
class TimeSeriesDecomposer:
    def __init__(self, series: pd.Series, period: int = 12):
        """Constructor for TimeSeriesDecomposer

        Parameters
        ----------
        series : pd.Series
            Time series data
        period : int, optional
            Period used for decomposing data, by default 12
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
