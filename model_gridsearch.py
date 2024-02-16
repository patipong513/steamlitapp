import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error
import numpy as np
import plotly.graph_objects as go 
import streamlit as st
from datetime import timedelta , datetime

def calculate_metrics_model(df):


    # Classification Metrics
    confusion = confusion_matrix(df['Actual Congestion Class'], df['Predicted Congestion Class'])
    precision = precision_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)
    recall = recall_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)
    f1 = f1_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)

    # Filtering for Random Forest Metrics (MAE and MAPE)
    df_date_non_zero = df[df['Actual Y'] != 0]

    if not df_date_non_zero.empty:
        mae = mean_absolute_error(df_date_non_zero['Actual Y'], df_date_non_zero['Predicted Y'])
        mape = np.mean(np.abs((df_date_non_zero['Actual Y'] - df_date_non_zero['Predicted Y']) / df_date_non_zero['Actual Y'])) * 100
    else:
        mae = 0
        mape = 0

    metrics_model = {
        'Confusion Matrix': confusion,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MAE': mae,
        'MAPE': mape
    }

    return metrics_model

def calculate_metrics(df, date_str):
    # แปลงคอลัมน์ 'sensor date' เป็น datetime

    # แยกข้อมูลเป็นรายวัน
    daily_data = df[df['sensor date'] == date_str]

    # Classification Metrics
    confusion = confusion_matrix(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'])
    precision = precision_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)
    recall = recall_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)
    f1 = f1_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)

    # Filtering for Random Forest Metrics (MAE and MAPE)
    df_date_non_zero = daily_data[daily_data['Actual Y'] != 0]

    if not df_date_non_zero.empty:
        mae = mean_absolute_error(df_date_non_zero['Actual Y'], df_date_non_zero['Predicted Y'])
        mape = np.mean(np.abs((df_date_non_zero['Actual Y'] - df_date_non_zero['Predicted Y']) / df_date_non_zero['Actual Y'])) * 100
    else:
        mae = 0
        mape = 0

    metrics = {
        'Date': date_str,
        'Confusion Matrix': confusion,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MAE': mae,
        'MAPE': mape
    }

    return metrics

# อ่านข้อมูลจาก DataFrames
df = pd.read_csv('randomforest_gridsearch.csv')  # เปลี่ยน 'data.csv' เป็นชื่อไฟล์ข้อมูลของคุณ
df['sensor time'] = pd.to_datetime(df['sensor time'])
df['sensor date'] = pd.to_datetime(df['sensor date'], format='%d/%m/%Y')

start_date = pd.to_datetime('2023-05-01')
end_date = pd.to_datetime('2023-06-30')
# ส่วนของการเรียกใช้ function ใน Streamlit app
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = start_date

# ฟังก์ชันสำหรับเลื่อนวันที่ไปข้างหน้า
def increment_date():
    # แปลง selected_date ให้เป็น Timestamp ถ้าจำเป็น
    current_date = pd.to_datetime(st.session_state.selected_date)

    next_date = current_date + timedelta(days=1)
    if next_date <= end_date:
        st.session_state.selected_date = next_date.date()
        





# fig.show()
metrics_result_model = calculate_metrics_model(df)
st.title("Model Performance")
# แสดงผลลัพธ์บน Streamlit
st.write("Confusion Matrix:")
st.write(metrics_result_model['Confusion Matrix'])
st.write("Precision:", metrics_result_model['Precision'])
st.write("Recall:", metrics_result_model['Recall'])
st.write("F1 Score:", metrics_result_model['F1 Score'])
st.write("MAE:", metrics_result_model['MAE'])
st.write("MAPE:", metrics_result_model['MAPE'])



st.title("Daily")
# แสดงปุ่มเพื่อเลื่อนไปวันถัดไป
if st.button("Next Day"):
    increment_date()

# แสดงตัวเลือกวันที่
selected_date = st.date_input("Select Date:", st.session_state.selected_date, min_value=start_date, max_value=end_date)
# อัปเดต state วันที่หากมีการเลือกจาก date input
st.session_state.selected_date = selected_date

date_str = selected_date

        
metrics_result = calculate_metrics(df, date_str.strftime('%Y-%m-%d'))
# พล็อตกราฟด้วย Plotly บน Streamlit
# พล็อตกราฟโดยใช้ Plotly
daily_data = df[df['sensor date'] == pd.to_datetime(date_str)]

hours = pd.date_range(start=daily_data['sensor time'].min(), end=daily_data['sensor time'].max(), freq='H')

fig = go.Figure()

fig.add_trace(go.Scatter(x=daily_data['sensor time'], y=daily_data['Actual Y'], mode='lines', name='Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=daily_data['sensor time'], y=daily_data['Predicted Y'], mode='lines', name='Predicted', line=dict(color='pink')))

fig.update_layout(title='Actual vs. Predicted Congestion',
                xaxis_title='Time',
                yaxis_title='Congestion',
                xaxis=dict(tickmode='array', tickvals=hours, tickformat='%H:%M'),
                yaxis=dict(range=[-100, 14000]),
                legend=dict(x=0, y=1),
                width=1000, height=500)
st.plotly_chart(fig)
    
# แสดงผลลัพธ์บน Streamlit
st.markdown("## Metrics for Date:")
st.write("Metrics for Date:", date_str)
st.write("Confusion Matrix:")
st.write(metrics_result['Confusion Matrix'])
st.write("Precision:", metrics_result['Precision'])
st.write("Recall:", metrics_result['Recall'])
st.write("F1 Score:", metrics_result['F1 Score'])
st.write("MAE:", metrics_result['MAE'])
st.write("MAPE:", metrics_result['MAPE'])
    
    

