# https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

# run these in terminal

# pip install streamlit
# streamlit run .\Streamlit-d1.py

import streamlit as st
import numpy as np
import pandas as pd

dataframe = pd.DataFrame(
    np.random.randn(10, 20), columns=("col %d" % i for i in range(20))
)

st.dataframe(dataframe.style.highlight_max(axis=0))

st.table(dataframe)  # static table

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)
st.map(map_data)

# Widgets

x = st.slider("x")  # widget
st.write(x, "squared is", x * x)

st.text_input("your name", key="name")
st.session_state.name

if st.checkbox("Show Dataframe"):
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
chart_data

df = pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})

option = st.selectbox("Which number do you like best?", df["first column"])
"you selected", option

add_selectebox = st.sidebar.selectbox(
    "How would you like to be contacted?", ("Email", "Home Phone", "Mobile Phone")
)

add_slider = st.sidebar.slider("Select range of values", 0.0, 100.0, (25.0, 75.0))

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button("Press me!")

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        "Sorting hat", ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
    st.write(f"You are in {chosen} house!")

# Advanced
# Read Advanced yourself
# https://docs.streamlit.io/get-started/fundamentals/advanced-concepts

# Try the chatbot as a side project
