import pandas as pd
from sklearn.preprocessing import RobustScaler

def load_and_preprocess_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    df = df.sort_values('Time').reset_index(drop=True)
    
    print("Scaling Amount and Time with RobustScaler...")
    rob_scaler = RobustScaler()
    
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    features = df.drop('Class', axis=1).values
    labels = df['Class'].values
    
    print(f"Data ready. Total transactions: {len(features)}")
    return features, labels