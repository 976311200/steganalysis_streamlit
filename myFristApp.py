import streamlit as st
import numpy as np
import pandas as pd
st.title("This is my first app")
st.write("hello")
st.write("Here's our first attempt at using data to create a table:")

df = pd.DataFrame({
  'first column': [5, 6, 7, 8],
  'second column': [50, 60, 70, 80]
})

st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

df
# 绘制折线图
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [32.37, 118.4],
    columns=['lat', 'lon'])

st.map(map_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])
    chart_data

# option = st.selectbox(
#     'Which number do you like best?',
#      df['first column'])
#
# 'You selected: ', option

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option


left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")


import time
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'