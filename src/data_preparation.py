import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    hotels = pd.read_csv(file_path)
    hotels = hotels.drop(labels='is_canceled', axis=1)
    
    irrelevant_cols = ['agent', 'company', 'reservation_status_date', 'arrival_date_day_of_month', 
                       'arrival_date_year', 'arrival_date_week_number', 'country']
    hotels = hotels.drop(labels=irrelevant_cols, axis=1)
    
    categorical_data = ['distribution_channel', 'hotel', 'meal', 'market_segment', 'assigned_room_type', 
                        'reserved_room_type', 'arrival_date_month', 'deposit_type', 'customer_type']
    hotels = pd.get_dummies(hotels, columns=categorical_data, dtype=int)
    
    hotels['reservation_status'] = hotels['reservation_status'].replace({'Check-Out':2, 'Canceled':1, 'No-Show':0})
    
    return hotels

def prepare_features_and_target(data):
    X = data.drop(['reservation_status'], axis=1)
    y = data['reservation_status']
    
    nan_columns = X.columns[X.isna().any()]
    X = X.drop(columns=nan_columns)
    
    return X, y

def split_and_scale_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=12)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
