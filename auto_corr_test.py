# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:14:24 2023

@author: 35003
"""
#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import math
import logging

class DataClean(object):
    
    def __init__(self, log_filename='app.log'):
        # Configure logging for the instance
        logging.basicConfig(filename=log_filename, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Configure logging for the instance
        #logging.basicConfig(filename=log_filename, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        #self.logger = logging.getLogger(__name__)
    
    def clean_na(self,input_na):
        """
        Detect nan values, point out their positions, and replace them with the previous value.
        
        """

        input_na = self._validate_input_( input_na )
        
        # Detect Na values
        na_positions = input_na.isna()

        for column in na_positions.columns:
            for index, is_na in enumerate(na_positions[column]):
                if is_na:
                    self.logger.info(f"NaN value found in column '{column}' at index {index}.")
                    print(f"NaN value found in column '{column}' at index {index}.")
                    
                    if index > 0:
                        input_na.at[index, column] = input_na.at[index - 1, column]
                    else:
                        # If the first row is na, find the next non-na value in the column
                        next_non_na_index = input_na[column].first_valid_index()
                        if next_non_na_index is not None:
                            input_na.at[index, column] = input_na.at[next_non_na_index, column]
        return input_na
    
    def clean_zero(self, input_zero):
        """
        Detect 0 values in a DataFrame or Series, log their positions, and replace them with the previous value.
        
        
        """
        
        input_zero = self._validate_input_(input_zero)
        
        # Detect 0 values
        zero_positions = (input_zero == 0)

        for column in zero_positions.columns:
            for index, is_zero in enumerate(zero_positions[column]):
                if is_zero:
                    #self.logger.info(f"0 value found in column '{column}' at index {index}.")
                    print(f"0 value found in column '{column}' at index {index}.")
                    # Replace 0 value with the previous value
                    if index > 0:
                        input_zero.at[index, column] = input_zero.at[index - 1, column]

        return input_zero

    def clean_abnormal(self, input_abnormal, val=0.2):
        """
        Detect values fluctuate more than a given percentage compared with previous row.
        
        Parameters:
        
        - val: float, the allowed fluctuation percentage (e.g. 0.2 for 20%)
        
       
        """
        
        # Check the type of the input
        if isinstance(input_abnormal, pd.Series):
            input_abnormal = input_abnormal.to_frame()
        elif isinstance(input_abnormal, pd.DataFrame):
            pass
        else:
            print("需要输入dataframe格式或series格式")
            return
        
        #detect vol
        for column in input_abnormal.columns:
            #skip datetime column
            if pd.api.types.is_datetime64_any_dtype(input_abnormal[column]):
                continue
            
            for index in range(1, len(input_abnormal)):
                try:
                    prev_value = float(input_abnormal.at[index - 1, column])
                    current_value = float(input_abnormal.at[index, column])
                except ValueError:  # If transform to float fails, skip 
                    continue
                if prev_value == 0:  # Avoid 0 becomes demomenator
                    continue
                
                fluctuation = abs((current_value - prev_value) / prev_value)
                
                if fluctuation > val:
                    #self.logger.info(f"abnormal fluctuation detected in column '{column}' at index {index}. Replacing {current_value} with {prev_value}.")
                    print(f"abnormal fluctuation detected in column '{column}' at index {index}. Replacing {current_value} with {prev_value}.")
                    input_abnormal.at[index, column] = prev_value

        return input_abnormal

    def decomposition( self, df, period = 14, seasonal=21 ):
        results = [ ]
        df = self._validate_input_( df )


        for column in df.columns:
            series = df[ column ].dropna()

            try:
                stl = STL( series, period = period, seasonal=seasonal )
                result = stl.fit()
                result.plot()

                # 设置图像的标题为当前列的名字
                # plt.title( column )

                plt.show()
                components = pd.concat( [ result.trend, result.seasonal, result.resid ], axis = 1 )
                components.columns = [ f"{column}_trend", f"{column}_seasonal", f"{column}_resid" ]
                results.append( components )
            except ValueError as e:
                print( f"ValueError for column {column}: {e}" )
                results.append( df[ [ column ] ].rename( columns = { column: f"{column}_orig" } ) )

        # Combine the results to a single DataFrame
        results_df = pd.concat( results, axis = 1 )

        # Check for all-zero columns and delete them
        for column in results_df.columns:
            if (results_df[ column ] == 0).all():
                print( f"{column} not existed, so deleted" )
                results_df = results_df.drop( columns = [ column ] )

        return results_df

    def _validate_input_(self,data):
        
        ''' 判断输入是否为Series，如果是，则转化为DataFrame'''
        if isinstance(data, pd.Series):
            return data.to_frame()
        # 检查输入是否为DataFrame
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("The input should be a pandas Series or DataFrame.")
        return data

class AutoCorr (object):
    
    def __init__(self,data_set):
        self.DATA_SET_all=data_set
        self.DATA_SET=None
        
        #global data_stationary
        self.data_stationary_diff = pd.DataFrame()
        #global data_non_stationary
        self.data_non_stationary_diff = pd.DataFrame()
        
        #global data_stationary_original
        self.data_stationary_original = pd.DataFrame()
        #global data_non_stationary_original
        self.data_non_stationary_original = pd.DataFrame()
        
    def differencing(self,lags=24,periods=1):
        if isinstance(self.DATA_SET_all, pd.DataFrame):
            for column in self.DATA_SET_all.columns:
                #print(column)
                self.DATA_SET = self.DATA_SET_all[column]#只有单独一列
                #self.DATA_SET = np.log(self.DATA_SET)
                self.DATA_SET = self.DATA_SET.diff(periods=periods)
                self.DATA_SET.fillna(0, inplace=True)
                self.DATA_SET = np.array(self.DATA_SET.values).reshape(-1, 1)
                self.DATA_SET = np.where(np.isnan(self.DATA_SET) | np.isinf(self.DATA_SET), 0, self.DATA_SET)
                
            
                # 绘制 ACF 图
                plt.figure(figsize=(12, 4))
                plot_acf(self.DATA_SET, lags=lags)
                plt.xlabel("Lags")
                plt.ylabel("ACF")
                plt.title(f"{column}'s Autocorrelation Function")
                plt.show()
        
                # 绘制 PACF 图
                plt.figure(figsize=(12, 4))
                plot_pacf(self.DATA_SET, lags=lags)
                plt.xlabel("Lags")
                plt.ylabel("PACF")
                plt.title(f"{column}'s Autocorrelation Function")
                plt.show()
        
                #  adf
                adf_result = adfuller(self.DATA_SET)
        
                # Print the test results
                print(f"ADF Statistic (differenced data): {adf_result[0]}")
                print(f"p-value (differenced data): {adf_result[1]}")
               
                self.DATA_SET = pd.Series(self.DATA_SET.ravel())
                
                # Check if the differenced time series is stationary
                if adf_result[1] < 0.025:
                    self.ADF_test_result=True
                    self.data_stationary_diff[column]=self.DATA_SET
                    
                    if column in self.data_non_stationary_diff.columns:
                        self.data_non_stationary_diff.drop(columns=[column], inplace=True)
                    else:
                        pass
                    print(f"The time series of {column} is stationary.")
                    
                else:
                    self.ADF_test_result=False
                    self.data_non_stationary_diff[column]=self.DATA_SET
                    print(f"The time series of {column} is not stationary.")
                    
        elif isinstance(self.DATA_SET_all, pd.Series):
            self.DATA_SET = self.DATA_SET_all#只有单独一列
            #self.DATA_SET = np.log(self.DATA_SET)
            data_name=self.DATA_SET.name
            self.DATA_SET = self.DATA_SET.diff()
            self.DATA_SET.fillna(0, inplace=True)
            self.DATA_SET = np.array(self.DATA_SET.values).reshape(-1, 1)
            self.DATA_SET = np.where(np.isnan(self.DATA_SET) | np.isinf(self.DATA_SET), 0, self.DATA_SET)
            
        
            # 绘制 ACF 图
            plt.figure(figsize=(12, 4))
            plot_acf(self.DATA_SET, lags=lags)
            plt.xlabel("Lags")
            plt.ylabel("ACF")
            plt.title("Autocorrelation Function")
            plt.show()
    
            # 绘制 PACF 图
            plt.figure(figsize=(12, 4))
            plot_pacf(self.DATA_SET, lags=lags)
            plt.xlabel("Lags")
            plt.ylabel("PACF")
            plt.title("Partial Autocorrelation Function")
            plt.show()
    
            #  adf
            adf_result = adfuller(self.DATA_SET)
    
            # Print the test results
            print(f"ADF Statistic (differenced data): {adf_result[0]}")
            print(f"p-value (differenced data): {adf_result[1]}")
           
            self.DATA_SET = pd.Series(self.DATA_SET.ravel())
            
            # Check if the differenced time series is stationary
            if adf_result[1] < 0.025:
                self.ADF_test_result=True
                self.data_stationary_diff[data_name]=self.DATA_SET
                
                if data_name in self.data_non_stationary_diff.columns:
                    self.data_non_stationary_diff.drop(columns=[data_name], inplace=True)
                else:
                    pass
                print(f"The time series of {data_name} is stationary.")
                
            else:
                self.ADF_test_result=False
                self.data_non_stationary_diff[data_name]=self.DATA_SET
                print(f"The time series of {data_name} is not stationary.")
        else:
            print("please input dataframe or series")    
        return self.DATA_SET

    def ADF_test(self,lags=24):
        
        if isinstance(self.DATA_SET_all, pd.DataFrame):
            for column in self.DATA_SET_all.columns:
                #print(column)
                self.DATA_SET = self.DATA_SET_all[column]
                self.DATA_SET.fillna(0, inplace=True)
                # 绘制 ACF 图
                plt.figure(figsize=(12, 4))
                plot_acf(self.DATA_SET, lags=lags)
                plt.xlabel("Lags")
                plt.ylabel("ACF")
                plt.title("Autocorrelation Function")
                plt.show()
        
                # 绘制 PACF 图
                plt.figure(figsize=(12, 4))
                plot_pacf(self.DATA_SET, lags=lags)
                plt.xlabel("Lags")
                plt.ylabel("PACF")
                plt.title("Partial Autocorrelation Function")
                plt.show()
        
                #  adf
                adf_result = adfuller(self.DATA_SET)
        
                # Print the test results
                print(f"ADF Statistic (differenced data): {adf_result[0]}")
                print(f"p-value (differenced data): {adf_result[1]}")
        
                # Check if the differenced time series is stationary
                data_name=self.DATA_SET.name
                if adf_result[1] < 0.025:
                    self.ADF_test_result=True
                    self.data_stationary_original[data_name]=self.DATA_SET
                    # data_stationary[column]=self.DATA_SET
                    print(f"The time series of {data_name} is stationary.")
                else:
                    self.ADF_test_result=False
                    self.data_non_stationary_original[data_name]=self.DATA_SET
                    print(f"The time series of {data_name} is not stationary.")
                    
        elif isinstance(self.DATA_SET_all, pd.Series):
            self.DATA_SET = self.DATA_SET_all
            self.DATA_SET.fillna(0, inplace=True)
            # 绘制 ACF 图
            plt.figure(figsize=(12, 4))
            plot_acf(self.DATA_SET, lags=lags)
            plt.xlabel("Lags")
            plt.ylabel("ACF")
            plt.title("Autocorrelation Function")
            plt.show()
    
            # 绘制 PACF 图
            plt.figure(figsize=(12, 4))
            plot_pacf(self.DATA_SET, lags=lags)
            plt.xlabel("Lags")
            plt.ylabel("PACF")
            plt.title("Partial Autocorrelation Function")
            plt.show()
    
            #  adf
            adf_result = adfuller(self.DATA_SET)
    
            # Print the test results
            print(f"ADF Statistic (differenced data): {adf_result[0]}")
            print(f"p-value (differenced data): {adf_result[1]}")
    
            # Check if the differenced time series is stationary
            data_name=self.DATA_SET.name
            if adf_result[1] < 0.025:
                self.ADF_test_result=True
                self.data_stationary_original[data_name]=self.DATA_SET
                # data_stationary[column]=self.DATA_SET
                print(f"The time series of {data_name} is stationary.")
            else:
                self.ADF_test_result=False
                self.data_non_stationary_original[data_name]=self.DATA_SET
                print(f"The time series of {data_name} is not stationary.")
        else:
            print("please input dataframe or series")   
        return


class DataNormalizer(object):
       
    def __init__(self, input_normalize):
        '''
        initialize   data
        
        '''
        if isinstance(input_normalize, pd.Series):
            self.input_normalize = input_normalize.to_frame()
        elif isinstance(input_normalize, pd.DataFrame):
            self.input_normalize = input_normalize
        else:
            raise ValueError("Input data must be a pandas DataFrame or Series.")
            
    
       
    def normalize(self, method='z-score', columns=None):
        """
        normalize data (DataFrame or Series) using specified method(z-score,min-max and robust included).
        
        Parameters:
        - method: normalization method

        """
        
        if columns:
            self.input_normalize[columns] = self.input_normalize[columns].apply(lambda x: self._normalize_method(x, method))
        else:
            self.input_normalize = self.input_normalize.apply(lambda x: self._normalize_method(x, method))
        
        return self.input_normalize
    
    
    def _normalize_method(self, column, method):
        if method == 'z-score':
            return (column - column.mean()) / column.std()
        
        elif method == 'min-max':
            return (column - column.min()) / (column.max() - column.min())
        
        elif method == 'robust':
            return (column - column.median()) / (column.quantile(0.75) - column.quantile(0.25))
        
        else:
            raise ValueError("Invalid normalization method. Choose 'z-score', 'min-max', or 'robust'.")

################################################################################

################################################################################

################################################################################

#%%
'''
auto corre test
'''

if __name__ == '__main__':
    
    '''read data and arrange format'''
    ###read price
    path = os.getcwd()
    print(path)
    #OneDrive\\Desktop\\dadi\\
    df1=pd.read_excel(path+'\\'+'TA.xlsx')
    #clean data
    df1.drop([0, 2], inplace=True)
    df1.iloc[0,0]='time'
    df1.iloc[0,1]='close'
    df1.columns = df1.iloc[0]
    df1 = df1[1:]
    #change time to date base
    df1.iloc[:, 0] = df1.iloc[:, 0].str.strip()
    df1 = df1.reset_index(drop=True)
    df = pd.to_datetime(df1.iloc[:,0], format='%Y/%m/%d-%H:%M')
    df = df.to_frame()
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.date
    ###read rf
    df2=pd.read_excel(path+'\\'+'rf_15min.xlsx')
    #clean data
    df2.iloc[0,0]='time'
    df2.iloc[0,1]='1Y_rf_15min'
    df2.columns = df2.iloc[0]
    df2 = df2[1:]
    #reverse
    df2 = df2.iloc[::-1].reset_index(drop=True)
    #date time
    df2['time'] = pd.to_datetime(df2['time'])
    df2['time'] = df2['time'].dt.date
    ###merge
    df['close']=df1['close']
    df = df.merge(df2[['time', '1Y_rf_15min']], on='time', how='left')
    df.iloc[:,0]=df1.iloc[:,0]
    #na test
    #df.iloc[:3, 1] = None
    
    '''data cleaning'''
    obj1=DataClean()
    df=obj1.clean_na(df)
    df=obj1.clean_zero(df)
    df=obj1.clean_abnormal(df,0.7)
    df=obj1.decomposition(df)
    
#%%
    #日志读取测试
    # with open(os.path.join(os.getcwd(), 'app.log'), 'r') as log_file:
    #     log_contents = log_file.read()
    #     print(log_contents)
    
    ###other test
    # test=np.array([])
    # data=[0, 1, 1, 0, 0]
    # test = pd.Series(data)
    # type(test)
    # obj1.clean_abnormal(test,0.3)
    # obj1.clean_na(test)
    # obj1.clean_zero(test)
    
    '''stationary test'''
    data_set = df.iloc[:,1:]
    obj = AutoCorr(data_set)
    obj.ADF_test()
    print(obj.data_non_stationary_original)
    type(obj.data_non_stationary_original)
    obj.differencing()
    df_stationary=obj.data_stationary_diff
    print(df_stationary)
    
    '''normalize'''
    obj2=DataNormalizer(df_stationary)
    df_normalized=obj2.normalize()




