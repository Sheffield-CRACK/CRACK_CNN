import numpy as np
import pandas as pd

# Global variables
lambda_1 = 12355.
spec_type_code = {'Ia':0,
                  'Ibc':1,
                  'II':2}

# Define method to get SN metadata
def get_sn_metadata(df_row):
    
    sn_name = df_row['name']
    spec_type = df_row['grouped_type']
    spec_type_num = spec_type_code[spec_type]
    
    return sn_name, spec_type_num

# Define method to get heatmap values
def get_sn_heatmap(sn_name, data_dir, norm=True):
        
    # Get heatmap from data file
    sn_file = np.load('{}/{}.npy'.format(data_dir, sn_name), allow_pickle=True)
    heatmap = sn_file.item().get('flux_grid')
    
    # Normalise heatmap
    if norm==True:
        heatmap = heatmap/np.max(heatmap)
    
    return heatmap

# Create data-label pairs
def create_xy(df, data_dir):
    
    X = []
    y = []
    for i, row in df.iterrows():
        sn_name, sn_type = get_sn_metadata(row)
        
        # If heatmap file doesn't exist, skip
        try:
            heatmap = get_sn_heatmap(sn_name, data_dir)
        except FileNotFoundError:
            continue
            
        X.append(heatmap)
        y.append(sn_type)
    
    return np.array(X), np.array(y)

def main():

    # Data file directories
    train_data_dir = 'heatmap_data/trainset'
    val_data_dir = 'heatmap_data/valset'
    test_data_dir = 'heatmap_data/testset'

    # Load metadata
    train_metadata = pd.read_csv('metadata/train_metadata.csv')
    val_metadata = pd.read_csv('metadata/validation_metadata.csv')
    test_metadata = pd.read_csv('metadata/test_metadata.csv')

    # Create training, validation, test datasets
    X_train, y_train = create_xy(train_metadata, train_data_dir)
    X_val, y_val = create_xy(val_metadata, val_data_dir)
    X_test, y_test = create_xy(test_metadata, test_data_dir)

    # Save data
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

if __name__ == "__main__":
    main()