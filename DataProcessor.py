import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def describe_data(self):
        print(self.data.describe())
        
    def handle_missing_values(self):
        #check missing values
        sum_of_missing_values = self.data.isnull().sum()
        if sum_of_missing_values.sum() == 0:
            print("No missing values found")
        else:
            air_quality_data = self.data.dropna()
            print("Missing values found, replaced them with NA")
            print(self.data.head())
            
    def preprocess_timestamp(self):
        self.data["Timestamp"] = pd.to_datetime(self.data["Timestamp"])
    
    def sort_by_timestamp(self):
        self.data = self.data.sort_values(by="Timestamp")
        
    def select_columns(self, columns):
        self.data = self.data[columns]
        
    def scale_pm25(self):
        scaler = StandardScaler()
        self.data["PM2.5"] = scaler.fit_transform(self.data["PM2.5"].values.reshape(-1,1))
        
    def plot_pm25_levels(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.data["Timestamp"], self.data["PM2.5"], label="PM2.5 Levels")
        plt.title("PM2.5 Levels Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("PM2.5 Levels")
        plt.legend()
        plt.show()
