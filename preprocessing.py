import numpy as np
import pandas as pd

import argparse
import os

import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = ['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea',
       'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex',
       'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory',
       'VehiclePrice', 'PolicyNumber', 'RepNumber', 'Deductible',
       'DriverRating', 'Days:Policy-Accident', 'Days:Policy-Claim',
       'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',
       'PoliceReportFiled', 'WitnessPresent', 'AgentType',
       'NumberOfSuppliments', 'AddressChange-Claim', 'NumberOfCars', 'Year',
       'BasePolicy', 'FraudFound']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()
    print(f"recieved args: {args}")
    
    input_data_path = os.path.join("/opt/ml/processing/input", "carclaims.csv")
    
    data = pd.read_csv(input_data_path)
    data = pd.DataFrame(data, columns=columns)
    
    data.loc[data["Age"] == 0, "Age"] = np.nan
    data.dropna(inplace=True)
    encoder = LabelEncoder()
    categorical = data.columns[data.dtypes==object]
    for c in categorical:
        encoder.fit(data[c])
        data[c] = encoder.transform(data[c])
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print(c,":",mapping)
  
    x = data.drop(["FraudFound", "PolicyNumber"], axis=1)
    y = data.FraudFound
    
    scaler = StandardScaler()
    x["Age"] = scaler.fit_transform(x[["Age"]])
    


    split_ratio = args.train_test_split_ratio
    print(f"split: {split_ratio}")
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=split_ratio, random_state=42)
    
    
    Xtrain_output_path = os.path.join("/opt/ml/processing/train", "Xtrain.csv")
    Xtest_output_path = os.path.join("/opt/ml/processing/train", "Xtest.csv")
    ytrain_output_path = os.path.join("/opt/ml/processing/test", "ytrain.csv")
    ytest_output_path = os.path.join("/opt/ml/processing/test", "ytest.csv")
   
    print("Saving features:")
    
    pd.DataFrame(x_train).to_csv(Xtrain_output_path, header=False, index=False)
    pd.DataFrame(x_test).to_csv(Xtest_output_path, header=False, index=False)
    pd.DataFrame(y_train).to_csv(ytrain_output_path, header=False, index=False)
    pd.DataFrame(y_test).to_csv(ytest_output_path, header=False, index=False)