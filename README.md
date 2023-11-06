# Predicting Geomagnetic Activity

### Problem Statement

The main goal of this project is to predict the solar wind properties at the Earth's trajectory and to develop a machine learning web application that can predict them from the sunspot number. There are several physics-based models that are computationally expensive to predict solar wind, which include thermodynamics that heat and accelerate them, wave heating (ion-cyclotron waves), and micro-flare (or nano-flare) heating are the most commonly identified heating mechanisms for solar wind heating. The primary input of these models is the solar magnetogram data, which is the photospheric (solar surface) magnetic field observations.      

---

### Software Requirements

- **pandas:** Required for data manipulation.
- **numpy:** Essential for numerical operations.
- **os:** For interacting with the operating system.
- **glob:** Helps in file manipulation.
- **matplotlib:** Used for data visualization.
- **seaborn:** Enhances data visualization.
- **datetime:** For working with date and time.
- **scikit-learn:** For machine learning models and evaluation.
    - **sklearn.tree.DecisionTreeClassifier:** For Decision Tree based classification.
    - **sklearn.ensemble.RandomForestClassifier:** For Random Forest based classification.
- **imblearn.over_sampling.SMOTE:** For synthetic oversampling in imbalanced datasets.
- **scipy:** Provides signal processing capabilities.
- **selenium,ChromeDriver/SafariDriver:** For web scraping and automation.

### Application:

To run the application, install all required libraries above. And run the following command from the local repo **/streamlit** directory:

```console
streamlit run web_app.py
```

<img src =images/web_app.png/>

### What is Solar wind:

- Solar wind consists of particles emitted from the sun's atmosphere with a sufficiently large velocity to escape from the sun's gravitational acceleration.
- Solar wind can mainly be divided into slow and fast wind, the common state of the solar wind is slow wind.
  
       slow wind 200 km/s - 450 km/s
       fast wind 4500 km/s - 850 km/s

### Sunspot solar cycle:
- Solar cycles have an average duration of about 11 years.
- Solar maximum and solar minimum refer to periods of maximum and minimum sunspot counts.
- Cycles span from one minimum to the next.

      Phases of the Solar Cycle:
  
      Solar minimum
      Rising phase
      Solar maximum
      Declining phase

cite : https://www.swpc.noaa.gov/products/solar-cycle-progression      

### Solar wind Properties at 1~au:

    Plasma density (Np)
    Magnetic field strength (B)
    Plasma temperature (T[K])
    Solar Wind Speed (Ur)

---
### Kp-index

The Kp-index is a scale used to measure the magnitude of geomagnetic disturbances, with a geomagnetic storm starting at Kp5 and continuing with the G-scale.

### Geomagnatic scale with the Kp index

| Kp    | Kp in decimals | G-scale | Auroral activity |
|-------|----------------|---------|------------------|
| 0o    | 0.00           | G0      | Quiet            |
| 0+    | 0.33           | G0      | Quiet            |
| 1-    | 0.67           | G0      | Quiet            |
| 1o    | 1.00           | G0      | Quiet            |
| 1+    | 1.33           | G0      | Quiet            |
| 2-    | 1.67           | G0      | Quiet            |
| 2o    | 2.00           | G0      | Quiet            |
| 2+    | 2.33           | G0      | Quiet            |
| 3-    | 2.67           | G0      | Unsettled        |
| 3o    | 3.00           | G0      | Unsettled        |
| 3+    | 3.33           | G0      | Unsettled        |
| 4-    | 3.67           | G0      | Active           |
| 4o    | 4.00           | G0      | Active           |
| 4+    | 4.33           | G0      | Active           |
| 5-    | 4.67           | G1      | Minor storm      |
| 5o    | 5.00           | G1      | Minor storm      |
| 5+    | 5.33           | G1      | Minor storm      |
| 6-    | 5.67           | G2      | Moderate storm   |
| 6o    | 6.00           | G2      | Moderate storm   |
| 6+    | 6.33           | G2      | Moderate storm   |
| 7-    | 6.67           | G3      | Strong storm     |
| 7o    | 7.00           | G3      | Strong storm     |
| 7+    | 7.33           | G3      | Strong storm     |
| 8-    | 7.67           | G4      | Severe storm     |
| 8o    | 8.00           | G4      | Severe storm     |
| 8+    | 8.33           | G4      | Severe storm     |
| 9-    | 8.67           | G4      | Severe storm     |
| 9o    | 9.00           | G5      | Extreme storm    |

### Folder structure
 - code
   - **notebook.ipynb**
     
         This notebook included software requirements, data visualization, feature engineering, modeling, Model evaluation
     
             - Used RandomForestClassifier, and DecisionTreeClassifier as the classification models
             - Selected the best model out of that for predictions
           
 - Data_collection
    - **OMNI_websitedata_collection.ipynb**

         This notebook uses automation to download solar wind parameter data at 1~au using the NASA, Coordinated Data Analysis Web.

          Requirements:
          Selenium webdriver (chrome/safari)

    - Data
          This folder includes the following .csv and .csv1 files

            1992-2023.csv1
            test.csv1
            all_params.csv
            sunspot.csv
            test_sc.csv

        |   | Column                 | Non-Null Count  | Dtype  |
        |-----|------------------------|-----------------|--------|
        | 0   | Wind_Category_extreme  | 90520 non-null  | object |
        | 1   | Wind_Category_fast     | 90520 non-null  | object |
        | 2   | Wind_Category_slow     | 90520 non-null  | object |
        | 3   | Np_Category_extreme    | 90520 non-null  | object |
        | 4   | Np_Category_high       | 90520 non-null  | object |
        | 5   | Np_Category_low        | 90520 non-null  | object |
        | 6   | B_Category_extreme     | 90520 non-null  | object |
        | 7   | B_Category_high        | 90520 non-null  | object |
        | 8   | B_Category_low         | 90520 non-null  | object |
        | 9   | T_Category_extreme     | 90520 non-null  | object |
        | 10  | T_Category_high        | 90520 non-null  | object |
        | 11  | T_Category_low         | 90520 non-null  | object |
        | 12  | phase_declining        | 90520 non-null  | object |
        | 13  | phase_rising           | 90520 non-null  | object |
        | 14  | phase_solar_max        | 90520 non-null  | object |
        | 15  | phase_solar_min        | 90520 non-null  | object |
        | 16  | SN                     | 90520 non-null  | float64 |
        | 17  | F10.7obs               | 90520 non-null  | float64 |
        | 18  | G                      | 90520 non-null  | object |


- images
    - Images used /created throughout the project

- pickles
    - **model.pkl** - Final pickled model that is used in the Streamlit App this is chosen to be from the RandomForest model.
    - **predict_G_storm.pkl** - This pickled function is for the data processing and cleaning
 

- streamlit
    - **web_app.py** - streamlit App code
    - Test_Streamlit.ipynb - used to test the interactive web app
    - **G_storm.py** - Function for the data processing and cleaning

### Model Evaluation

***Model evaluation based on train/test score***

| Model          | Score on train | Score on test |
|----------------|----------------|---------------|
| Randomfc       |     0.982      |     0.951     |
| DecisionTree   |     0.968      |     0.935     |


<img src = images/confusion_matrix.png/>

***Classification report***

|   Storm scale | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
|  G0  |   0.965   |  0.956 |  0.960   |  21173  |
|  G1  |   0.920   |  0.915 |  0.917   |  21173  |
|  G2  |   0.925   |  0.920 |  0.923   |  21173  |
|  G3  |   0.941   |  0.950 |  0.946   |  21173  |
|  G4  |   0.962   |  0.973 |  0.967   |  21172  |
|  G5  |   0.992   |  0.992 |  0.992   |  21173  |
||||||
|  Accuracy  |         |        |  0.951   | 127037  |
|  Macro Avg |   0.951   |  0.951 |  0.951   | 127037  |
| Weighted Avg|  0.951   |  0.951 |  0.951   | 127037  |


### Datasets

- https://www.sidc.be/SILSO/datafiles
- https://cdaweb.gsfc.nasa.gov/
- https://www.gfz-potsdam.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index

### Author:
    
**Kalpa Henadhira Arachchige (he/him)** 
    - [Github](https://github.com/kharindra)
    - [LinkedIn](https://www.linkedin.com/in/kalpa-henadhira/)
    

### Special Thanks
- To our advisors: Musfiqur Rahman, Sonyah Seiden, and Eric Bayless

