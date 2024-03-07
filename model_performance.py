import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error,accuracy_score
import numpy as np
import plotly.graph_objects as go 
import streamlit as st
from datetime import timedelta , datetime

def calculate_metrics_model(df):


    # Classification Metrics
    confusion = confusion_matrix(df['Actual Congestion Class'], df['Predicted Congestion Class'])
    accuracy = accuracy_score(df['Actual Congestion Class'], df['Predicted Congestion Class'])
    precision = precision_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)
    recall = recall_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)
    f1 = f1_score(df['Actual Congestion Class'], df['Predicted Congestion Class'], zero_division=1)
    
    
    df_date_non_zero = df[df['Actual Y'] != 0]
    df_date_zero = df[df['Actual Y'] == 0]

    if not df_date_non_zero.empty:
        mae_non_zero = mean_absolute_error(df_date_non_zero['Actual Y'], df_date_non_zero['Predicted Y'])
        mape_non_zero = np.mean(np.abs((df_date_non_zero['Actual Y'] - df_date_non_zero['Predicted Y']) / df_date_non_zero['Actual Y'])) * 100
        count_non_zero = len(df_date_non_zero)
    else:
        mae_non_zero = 0
        mape_non_zero = 0
        count_non_zero = 0

    if not df_date_zero.empty:
        mae_zero = 0
        mape_zero = 0
        count_zero = len(df_date_zero)
    else:
        mae_zero = 0
        mape_zero = 0
        count_zero = 0

    # Calculate weighted average MAPE
    total_count = len(df)
    avg_mape = ((mape_non_zero * count_non_zero) + (mape_zero * count_zero)) / total_count


    metrics_model = {
        'Confusion Matrix': confusion,
        'accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MAE': mae_non_zero,
        'MAPE': mape_non_zero,
        'avg_mape': avg_mape
    }

    return metrics_model

def calculate_metrics(df, date_str):
    # แปลงคอลัมน์ 'sensor date' เป็น datetime

    # แยกข้อมูลเป็นรายวัน
    daily_data = df[df['sensor date'] == date_str]

    # Classification Metrics
    confusion = confusion_matrix(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'])
    accuracy = accuracy_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'])
    precision = precision_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)
    recall = recall_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)
    f1 = f1_score(daily_data['Actual Congestion Class'], daily_data['Predicted Congestion Class'], zero_division=1)

        
    df_date_non_zero = daily_data[daily_data['Actual Y'] != 0]
    df_date_zero = daily_data[daily_data['Actual Y'] == 0]

    if not df_date_non_zero.empty:
        mae_non_zero = mean_absolute_error(df_date_non_zero['Actual Y'], df_date_non_zero['Predicted Y'])
        mape_non_zero = np.mean(np.abs((df_date_non_zero['Actual Y'] - df_date_non_zero['Predicted Y']) / df_date_non_zero['Actual Y'])) * 100
        count_non_zero = len(df_date_non_zero)
    else:
        mae_non_zero = 0
        mape_non_zero = 0
        count_non_zero = 0

    if not df_date_zero.empty:
        mae_zero = 0
        mape_zero = 0
        count_zero = len(df_date_zero)
    else:
        mae_zero = 0
        mape_zero = 0
        count_zero = 0

    # Calculate weighted average MAPE
    total_count = len(df)
    avg_mape = ((mape_non_zero * count_non_zero) + (mape_zero * count_zero)) / total_count

    metrics = {
        'Date': date_str,
        'Confusion Matrix': confusion,
        'accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MAE': mae_non_zero,
        'MAPE': mape_non_zero,
        'avg_mape': avg_mape
    }

    return metrics

# ---------------------------------------------------------------------------------------------------------------------------------------
# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('Randomparameterclass.csv')  # แทนชื่อไฟล์ CSV ที่คุณมีอยู่

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[0:25]
st.title("Class")
st.title("n_estimators")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['n_estimators'], y=selected_row['accuracy'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    title='Accuracy vs. n_estimators',
    xaxis_title='n_estimators',
    yaxis_title='Accuracy',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[0.98, 1]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[31:41]

st.title("max_depth")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['max_depth'], y=selected_row['accuracy'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='max_depth',
    yaxis_title='Accuracy',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[0.98, 1]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[41:60]

st.title("min_samples_split")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['min_samples_split'], y=selected_row['accuracy'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='min_samples_split',
    yaxis_title='Accuracy',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[0.98, 1]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[60:82]

st.title("min_samples_leaf")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['min_samples_leaf'], y=selected_row['accuracy'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='min_samples_leaf',
    yaxis_title='Accuracy',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[0.98, 1]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)


# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv('Randomparameterregression.csv')  # แทนชื่อไฟล์ CSV ที่คุณมีอยู่

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[0:25]
st.title("Regression")
st.title("n_estimators")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['n_estimators'], y=selected_row['MAPE'], mode='lines', name='MAPE', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    title='MAPE vs. n_estimators',
    xaxis_title='n_estimators',
    yaxis_title='MAPE',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[50, 90]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[30:41]

st.title("max_depth")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['max_depth'], y=selected_row['MAPE'], mode='lines', name='MAPE', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='max_depth',
    yaxis_title='MAPE',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[65, 70]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[41:60]

st.title("min_samples_split")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['min_samples_split'], y=selected_row['MAPE'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='min_samples_split',
    yaxis_title='MAPE',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[65, 70]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)

# กำหนดแถวที่ต้องการพล็อต (เช่นแถวที่ 1-31)
selected_row = df.iloc[60:82]

st.title("min_samples_leaf")
# สร้างกราฟ
fig = go.Figure()

# เพิ่มข้อมูล n_estimators
fig.add_trace(go.Scatter(x=selected_row['min_samples_leaf'], y=selected_row['MAPE'], mode='lines', name='Accuracy', line=dict(color='blue')))

# อัพเดตเลเอาท์ของกราฟ
fig.update_layout(
    xaxis_title='min_samples_leaf',
    yaxis_title='MAPE',
    legend=dict(x=0, y=1),
    yaxis=dict(range=[65, 70]),
    width=1000, height=500
)

# แสดงกราฟบน Streamlit
st.plotly_chart(fig)


# ---------------------------------------------------------------------------------------------------------------------------------------
# อ่านข้อมูลจาก DataFrames
df = pd.read_csv('randomforest.csv')  # เปลี่ยน 'data.csv' เป็นชื่อไฟล์ข้อมูลของคุณ
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
st.write("accuracy:", metrics_result_model['accuracy'])
st.write("Precision:", metrics_result_model['Precision'])
st.write("Recall:", metrics_result_model['Recall'])
st.write("F1 Score:", metrics_result_model['F1 Score'])
st.write("MAE:", metrics_result_model['MAE'])
st.write("MAPE:", metrics_result_model['MAPE'])
st.write("avg_mape:", metrics_result_model['avg_mape'])



st.title("Daily")
# แสดงปุ่มเพื่อเลื่อนไปวันถัดไป
if st.button("Next Day", key='next_day_button_1'):
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
st.write("accuracy:", metrics_result['accuracy'])
st.write("Precision:", metrics_result['Precision'])
st.write("Recall:", metrics_result['Recall'])
st.write("F1 Score:", metrics_result['F1 Score'])
st.write("MAE:", metrics_result['MAE'])
st.write("MAPE:", metrics_result['MAPE'])
st.write("avg_mape:", metrics_result['avg_mape'])
    
# --------------------------------------------------------------------------------------------------------------------------------------------   
# อ่านข้อมูลจาก DataFrames
df = pd.read_csv('randomforest_config.csv')  # เปลี่ยน 'data.csv' เป็นชื่อไฟล์ข้อมูลของคุณ
df['sensor time'] = pd.to_datetime(df['sensor time'])
df['sensor date'] = pd.to_datetime(df['sensor date'], format='%d/%m/%Y')

start_date = pd.to_datetime('2023-05-01')
end_date = pd.to_datetime('2023-06-30')
# ส่วนของการเรียกใช้ function ใน Streamlit app
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = start_date

        

# fig.show()
metrics_result_model = calculate_metrics_model(df)
st.title("Model Test Config")
# แสดงผลลัพธ์บน Streamlit
st.write("Confusion Matrix:")
st.write(metrics_result_model['Confusion Matrix'])
st.write("accuracy:", metrics_result_model['accuracy'])
st.write("Precision:", metrics_result_model['Precision'])
st.write("Recall:", metrics_result_model['Recall'])
st.write("F1 Score:", metrics_result_model['F1 Score'])
st.write("MAE:", metrics_result_model['MAE'])
st.write("MAPE:", metrics_result_model['MAPE'])
st.write("avg_mape:", metrics_result_model['avg_mape'])



st.title("Daily")
# แสดงปุ่มเพื่อเลื่อนไปวันถัดไป
if st.button("Next Day", key='next_day_button_2'):
    increment_date()

# แสดงตัวเลือกวันที่
selected_date = st.date_input("Select Date:", st.session_state.selected_date, min_value=start_date, max_value=end_date, key='unique_date_input_key')
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
st.write("accuracy:", metrics_result['accuracy'])
st.write("Precision:", metrics_result['Precision'])
st.write("Recall:", metrics_result['Recall'])
st.write("F1 Score:", metrics_result['F1 Score'])
st.write("MAE:", metrics_result['MAE'])
st.write("MAPE:", metrics_result['MAPE'])
st.write("avg_mape:", metrics_result['avg_mape'])


# --------------------------------------------------------------------------------------------------------------------------------------------   
# อ่านข้อมูลจาก DataFrames
df = pd.read_csv('randomforest_ship.csv')  # เปลี่ยน 'data.csv' เป็นชื่อไฟล์ข้อมูลของคุณ
df['sensor time'] = pd.to_datetime(df['sensor time'])
df['sensor date'] = pd.to_datetime(df['sensor date'], format='%d/%m/%Y')

start_date = pd.to_datetime('2023-05-01')
end_date = pd.to_datetime('2023-06-30')
# ส่วนของการเรียกใช้ function ใน Streamlit app
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = start_date

        

# fig.show()
metrics_result_model = calculate_metrics_model(df)
st.title("Model Test ship")
# แสดงผลลัพธ์บน Streamlit
st.write("Confusion Matrix:")
st.write(metrics_result_model['Confusion Matrix'])
st.write("accuracy:", metrics_result_model['accuracy'])
st.write("Precision:", metrics_result_model['Precision'])
st.write("Recall:", metrics_result_model['Recall'])
st.write("F1 Score:", metrics_result_model['F1 Score'])
st.write("MAE:", metrics_result_model['MAE'])
st.write("MAPE:", metrics_result_model['MAPE'])
st.write("avg_mape:", metrics_result_model['avg_mape'])



st.title("Daily")
# แสดงปุ่มเพื่อเลื่อนไปวันถัดไป
if st.button("Next Day", key='next_day_button_3'):
    increment_date()

# แสดงตัวเลือกวันที่
selected_date = st.date_input("Select Date:", st.session_state.selected_date, min_value=start_date, max_value=end_date, key='unique_date_input_key1')
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
st.write("accuracy:", metrics_result['accuracy'])
st.write("Precision:", metrics_result['Precision'])
st.write("Recall:", metrics_result['Recall'])
st.write("F1 Score:", metrics_result['F1 Score'])
st.write("MAE:", metrics_result['MAE'])
st.write("MAPE:", metrics_result['MAPE'])
st.write("avg_mape:", metrics_result['avg_mape'])
