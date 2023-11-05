import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


### Load the saved model
    
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)


img = "background.jpeg"


st.image(img)


st.title('Predicting Geomagnetic Storm')

# Fixed column names
columns = ['Wind_Category', 'Np_Category', 'B_Category', 'T_Category', 'phase', 'SN', 'F10.7obs']

# Create an empty dictionary to store user input data
data = {}

# Define button labels for respective categories
labels = {
    'Wind_Category': ['Fast', 'Slow', 'Extreme'],
    'Np_Category': ['Low', 'High', 'Extreme'],
    'B_Category': ['Low', 'High', 'Extreme'],
    'T_Category': ['Low', 'High', 'Extreme'],
    'phase': ['Solar Minimum', 'Rising Phase', 'Solar Maximum', 'Declining Phase']
}

# Collect data for each column from the user using buttons
for column_name in columns:
    if column_name in labels:
        button_clicked = st.radio(f"Select value for '{column_name}':", labels[column_name])
        if button_clicked:
            data[column_name] = [button_clicked]
    else:
        if column_name == 'SN':
            data[column_name] = st.text_input(f"Enter value for '{column_name}':", value='0')
        elif column_name == 'F10.7obs':
            data[column_name] = st.text_input(f"Enter value for '{column_name}':", value='0')
        else:
            data[column_name] = st.text_input(f"Enter value for '{column_name}':")        
    # else:
    #     value = st.text_input(f"Enter value for '{column_name}': ")
    #     data[column_name] = [value]


df = pd.DataFrame(data)

if all(v is not None for v in data.values()):
    # Create the DataFrame using the collected data
    st.dataframe(df)
    
    
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
    transformed_data = preprocessor.fit_transform(df)
    
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
    
    
    
    columns_test = [col for col in train_columns if col not in test.columns] ## the columns that should be added to the test data
    
    # Create a DataFrame with the new columns filled with zeros equaling the length of the dataframe that should be added to the test dataframe
    columns_test_df = pd.DataFrame({col: [0] * len(test) for col in columns_test})
    
    Test = pd.concat([test, columns_test_df], axis=1)
    
    preds = loaded_model.predict(Test[train_columns])
    
    
    st.write(preds)
    
    conditions = [
        preds == 'G0',
        preds == 'G1',
        preds == 'G2',
        preds == 'G3',
        preds == 'G4',
        preds == 'G5'
                    ]
    
    choices = [
        'Below storm level',
        'Minor storm',
        'Moderate storm',
        'Strong storm',
        'Severe storm',
        'Extreme storm'
                ]
    
    auroral_activity= np.select(conditions, choices)
    
    
    aurora = pd.DataFrame(list(auroral_activity))
    
    st.write(aurora)

st.divider()

st.title('Predicting solar wind data from 2023.01.01')

img = "sunspots.jpeg"

st.image(img)

st.title("Upload the sunspot data")

# Allow user to upload a CSV file
uploaded_file1 = st.file_uploader("", type=["csv"])

st.title("Upload the solar wind data")

# Allow user to upload a CSV file
uploaded_file2 = st.file_uploader("", type=["csv1"])

st.sidebar.write("**Kalpa Henadhira Arachchige**")
st.sidebar.write("**Date: 11/06/2023**")
st.sidebar.write("**DSI Capstone project**")

if uploaded_file1 and uploaded_file2 is not None:
    # Read the CSV file
    df3 = pd.read_csv(uploaded_file1)
    new = pd.read_csv(uploaded_file2)


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

    new_sw['phase'] = 'rising'

    
    new_sw['SN'] = np.repeat(np.array(df3['SN']), 8)
    new_sw['F10.7obs'] = np.repeat(np.array(df3['F10.7obs']), 8)
    
    st.divider()
    # Display the data in a table
    st.write("### Uploaded Data")
    st.write(new_sw)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=new_sw, x=new_sw.index, y=new_sw['SN'], color = 'blue', label = 'SN')
    sns.lineplot(data=new_sw, x=new_sw.index, y=new_sw['F10.7obs'], color = 'orange', label = 'F10.7obs')
    #ax.plot(new_sw.index, new_sw['SN'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Sunspot number')

    plt.xticks(rotation=45)

    if st.sidebar.button('Sunspot and F10.7obs'):
        st.sidebar.pyplot(fig)

    #### Make predictions

    # Fit and transform the data
    transformed_data = preprocessor.fit_transform(new_sw)

    
    columns = list(preprocessor.named_transformers_['ohe'].get_feature_names_out()) + passthrough
    
    # Convert the transformed data back to a DataFrame
    test = pd.DataFrame(transformed_data, columns= columns)
    
    test.columns = test.columns.str.lower()

    ## the columns that should be added to the test data
    columns_test = [col for col in train_columns if col not in test.columns] 
    
    # Create a DataFrame with the new columns filled with zeros equaling the length of the 
    #dataframe that should be added to the test dataframe
    
    columns_test_df = pd.DataFrame({col: [0] * len(test) for col in columns_test})
    
    Test = pd.concat([test, columns_test_df], axis=1)
    
    new_sw['G'] = list(loaded_model.predict(Test[train_columns]))

    proba = list(loaded_model.predict_proba(Test[train_columns]))

    probability = [max(row) for row in proba]

    new_sw['probability'] = np.round(probability,3)

    st.write('Predictions from the model!!!')
    st.write(new_sw.loc[:,'G':'probability'])

    st.write('Times had at least a minor storm!!!')
    st.write(new_sw[new_sw['G'] != 'G0'])

    st.write('Display the value counts')
    st.write(new_sw['G'].value_counts())


    fig1, ax = plt.subplots(figsize=(6, 4))

    # Filter 'G' column for values not equal to 'G0' and plot value counts as a horizontal bar plot using Seaborn
    filtered_values = new_sw['G'][new_sw['G'] != 'G0'].value_counts()
    sns.barplot(x=filtered_values.index, y=filtered_values.values, palette='viridis', ax=ax)  # 'Greens' is a sequential colormap


    ax.set_ylabel('Count')
    ax.set_xlabel('Storm type')
    ax.set_title('')

    if st.sidebar.button('G_scale'):
        st.sidebar.image('kp_g.jpeg')

    if st.sidebar.button('Solar Cycle'):
        st.sidebar.image('sn_kp.jpeg')
    
    if st.sidebar.button('Storm result'):
        st.sidebar.pyplot(fig1)




















