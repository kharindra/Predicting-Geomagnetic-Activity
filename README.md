# Predicting Solar wind properties at 1~au

---

**Problem Statement**

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
    - **sklearn.svm.SVC:** For Support Vector Machines based classification.
    - **sklearn.ensemble.GradientBoostingClassifier:** For Gradient Boosting based classification.
- **imblearn.over_sampling.SMOTE:** For synthetic oversampling in imbalanced datasets.
- **scipy:** Provides signal processing capabilities.
- **selenium,ChromeDriver/SafariDriver:** For web scraping and automation.

### Application:

To run the application, install all required libraries above. And run the following command from the local repo **/streamlit** directory:

```console
streamlit run web_app.py
```

<img src =/images/web_app.png/>

**Background study**

What is Solar wind:

Sunspot solar cycle:

Solar wind Properties at 1~au:

- Plasma density (Np)
- Magnetic field strength (B)
- Plasma temperature (T[K])
- Solar Wind Speed (Ur)

Coronal mass ejections (CMEs)

---

**Kp-index**

- The Kp-index is a scale used to characterize the magnitude of geomagnetic disturbances. 
- A geomagnetic storm starts at Kp5 after which the G-scale is also used.

##
- Kp0 = Quiet
- Kp1 = Quiet
- Kp2 = Quiet
- Kp3 = Unsettled
- Kp4 = Active
- Kp5 = Minor storm (G1)
- Kp6 = Moderate storm (G2)
- Kp7 = Strong storm (G3)
- Kp8 = Severe storm (G4)
- Kp9 = Extreme storm (G5)
##

### Datasets

- https://www.sidc.be/SILSO/datafiles
- https://cdaweb.gsfc.nasa.gov/
- https://www.gfz-potsdam.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index


