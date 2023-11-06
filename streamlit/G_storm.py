def predict_G_storm(omni, sunspot, phase):
    ### Load the python libraries
    import pandas as pd
    import numpy as np
    import pickle
    import seaborn as sns
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    ### Load the saved model
    
    with open('../pickles/model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    
    path = '../Data_collection/Data/' + omni

    new = pd.read_csv(path, skiprows=None)
    new.columns = ['date','B','T[K]','Np','Ur']

    new['date'] = pd.to_datetime(new['date'])
    new.set_index(new['date'],inplace = True)
    new_sw = new.drop(columns = ['date'])
    new_sw  = new_sw.resample('3H').mean()


    ### Categorize the wind speed into slow, and fast solar wind speed
    new_sw.loc[(new_sw['Ur'] > 200.0) & (new_sw['Ur'] < 450.0), 'Wind_Category'] = 'slow'
    new_sw.loc[(new_sw['Ur'] >=450) & (new_sw['Ur'] < 850.0), 'Wind_Category'] = 'fast'
    new_sw.loc[new_sw['Ur'] >= 850, 'Wind_Category'] = 'extreme'

    ### Categorize the plasma density at 1au
    new_sw.loc[(new_sw['Np'] >= 0.1) & (new_sw['Np'] < 7), 'Np_Category'] = 'low'
    new_sw.loc[(new_sw['Np'] >= 7) & (new_sw['Np'] < 60), 'Np_Category'] = 'high'
    new_sw.loc[new_sw['Np'] >= 60, 'Np_Category'] = 'extreme'

    ### Categorize the magnetic field at 1au
    new_sw.loc[(new_sw['B'] >= 0.1) & (new_sw['B'] < 6.3), 'B_Category'] = 'low'
    new_sw.loc[(new_sw['B'] >= 6.3) & (new_sw['B'] < 30), 'B_Category'] = 'high'
    new_sw.loc[new_sw['B'] >= 30, 'B_Category'] = 'extreme'

    ### Categorize the temperature at 1au
    new_sw.loc[(new_sw['T[K]'] >= 0) & (new_sw['T[K]'] < 110_000), 'T_Category'] = 'low'
    new_sw.loc[(new_sw['T[K]'] >= 110_000) & (new_sw['T[K]'] < 1_000_000), 'T_Category'] = 'high'
    new_sw.loc[new_sw['T[K]'] >= 1_000_000, 'T_Category'] = 'extreme'

    new_sw.drop(columns = ['B','T[K]','Np','Ur'],inplace = True)

    new_sw['phase'] = phase

    
    df3 = pd.read_csv('../Data_collection/Data/' + sunspot)

    new_sw['SN'] = np.repeat(np.array(df3['SN']), 8)
    new_sw['F10.7obs'] = np.repeat(np.array(df3['F10.7obs']), 8)

    # Columns to be One-Hot Encoded
    ohe_cols = ['Wind_Category','Np_Category','B_Category','T_Category','phase']
    
    # Columns to be passed without transformation
    passthrough = ['SN', 'F10.7obs']
    
    # Initializing the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(), ohe_cols),
            ('passthrough', 'passthrough', passthrough)
        ],
        remainder='drop'  # Drops the columns that are not specified
    )
    
    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(new_sw)

    
    columns = list(preprocessor.named_transformers_['ohe'].get_feature_names_out()) + passthrough
    
    # Convert the transformed data back to a DataFrame
    test = pd.DataFrame(transformed_data, columns= columns)
    
    test.columns = test.columns.str.lower()
    
    train_columns = ['wind_category_extreme',
                     'wind_category_fast',
                     'wind_category_slow',
                     'np_category_extreme',
                     'np_category_high',
                     'np_category_low',
                     'b_category_extreme',
                     'b_category_high',
                     'b_category_low',
                     't_category_extreme',
                     't_category_high',
                     't_category_low',
                     'phase_declining',
                     'phase_rising',
                     'phase_solar_max',
                     'phase_solar_min',
                     'sn',
                     'f10.7obs']

    ## the columns that should be added to the test data
    columns_test = [col for col in train_columns if col not in test.columns] 
    
    # Create a DataFrame with the new columns filled with zeros equaling the length of the 
    #dataframe that should be added to the test dataframe
    
    columns_test_df = pd.DataFrame({col: [0] * len(test) for col in columns_test})
    
    Test = pd.concat([test, columns_test_df], axis=1)
    
    new_sw['G'] = list(loaded_model.predict(Test[train_columns]))
    
    return sns.countplot(data=new_sw, x='G')
