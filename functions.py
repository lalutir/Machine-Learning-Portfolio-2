import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
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