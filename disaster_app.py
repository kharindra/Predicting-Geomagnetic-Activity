## Imports at the top
import pickle
import pandas as pd
import streamlit as st
from clean import clean_the_text
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the background image URL
background_image_url = 'https://www.computerhope.com/jargon/t/twitter.png'

# Create custom HTML and CSS to set the background image as a cover
page_bg = f"""
<style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
    }}
</style>
"""

## Adding a falling animation
flow = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          @keyframes falling {
            from {
              transform: translateY(-10%);
              opacity: 3;
            }
            to {
              transform: translateY(100%);
              opacity: 5;
            }
          }
        
          .raindrop {
            position: absolute;
            display: inline-block;
            font-size: 180px;
            animation: falling 20s linear infinite;
            right: 30%;
          }
        
        </style>
        </head>
        <body>
          <div class="rain-container">
            <div class="raindrop">☠️🏴‍☠️</div>
          </div>
        </body>
        </html>
        """
        
st.markdown(flow, unsafe_allow_html=True)


# Apply the custom CSS to the app
st.markdown(page_bg, unsafe_allow_html=True)

# Create the title for our app

st.title("Is this a disaster tweet!!!!")

#Load the model

with open('/Users/kalpahenadhira/Desktop/logr.pkl', 'rb') as model_file:
    logr = pickle.load(model_file)

# Load the function from the pickle file
with open('clean.pkl', 'rb') as file:
    clean = pickle.load(file)


#get user input

user_text = st.text_input('Add the tweet below: ', max_chars = 10000)


#Generate the predictions

prediction = logr.predict(clean(pd.Series([user_text])))[0]


#Display the prediction
if st.button("Enter"):
    if logr.predict(clean(pd.Series([user_text])))[0] == 1:
        st.write('<span style="font-size: 24px;">**Disaster tweet!!!!!**.</span>', unsafe_allow_html=True)
    else:
        st.write('<span style="font-size: 24px;">**Not a disaster tweet!!!!!**.</span>', unsafe_allow_html=True)

#Display the probability
    probability = np.round(logr.predict_proba(clean(pd.Series([user_text])))*100,2)
    
    st.write(f'<span style="font-size: 24px;"> **The probability that this tweet will be a disasater** :{np.round(logr.predict_proba(clean(pd.Series([user_text])))*100,2)[0][1]}%.</span>', unsafe_allow_html=True)



# Create a bar plot of probability values
    fig, ax = plt.subplots(figsize=(5, 4))
    
    classes = ['Not a disaster', 'Disaster']
    # Seaborn bar plot
    sns.barplot(x=classes, y=probability[0], palette="Blues")
    
    # Add labels and title
    plt.ylabel('Probability %')
    plt.title('Probability of the result')
    
    # Set the y-axis limit
    plt.ylim(0, 100)
    
    # Add a horizontal line at the probability = 50%
    ax.axhline(y=50, color='r', linestyle='--')

    # Set the ticks on both sides of the y axis
    ax.yaxis.set_ticks_position('both')
    
    # Display the plot using Streamlit
    st.pyplot(fig)
