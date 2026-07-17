# Pandas and Data Science Basics

This document covers official best practices for loading, cleaning, transforming, and visualizing data in Python using Pandas, NumPy, Matplotlib, Seaborn, and Plotly.

## File Loading
Always check the file extension and use the appropriate loading function.
* **CSV Files:**
  ```python
  df = pd.read_csv('filename.csv')
  ```
* **Excel Files:**
  Use `openpyxl` engine if needed.
  ```python
  df = pd.read_excel('filename.xlsx')
  # For specific sheets:
  df = pd.read_excel('filename.xlsx', sheet_name='Sheet1')
  ```
* **JSON Files:**
  ```python
  df = pd.read_json('filename.json')
  ```

## Data Inspection
Always inspect column names and data types before executing transformations.
```python
df.info()
df.head()
df.shape
df.columns
```

## Data Cleaning
* **Handling Missing Values:**
  ```python
  # Drop rows with any missing values
  df_clean = df.dropna()
  # Drop rows missing in specific columns
  df_clean = df.dropna(subset=['Salary', 'Age'])
  # Fill missing values
  df['Age'] = df['Age'].fillna(df['Age'].mean())
  df['Category'] = df['Category'].fillna('Unknown')
  ```
* **Type Conversion:**
  Ensure columns have the correct types before arithmetic or time-series operations.
  ```python
  df['Date'] = pd.to_datetime(df['Date'])
  df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
  ```
* **Handling Duplicates:**
  ```python
  df = df.drop_duplicates()
  ```

## Aggregation & Grouping
Use `groupby` to calculate statistics by categories.
```python
# Group by and aggregate
summary = df.groupby('Region')['Revenue'].sum().reset_index()
# Multiple aggregations
summary = df.groupby('Category').agg({
    'Price': 'mean',
    'Units': 'sum'
}).reset_index()
```

## Visualisation Guidelines
* **Matplotlib / Seaborn:**
  Always clear the active figure before creating a new plot to avoid overlapping lines. Save the figure as a static PNG file named `output_plot.png`.
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  plt.figure(figsize=(10, 6))
  sns.set_theme(style="darkgrid")
  
  sns.barplot(data=df_summary, x='Region', y='Revenue', hue='Region', palette='viridis')
  plt.title('Revenue by Region')
  plt.xlabel('Region')
  plt.ylabel('Revenue ($)')
  plt.tight_layout()
  
  # Save the plot
  plt.savefig('output_plot.png', dpi=300)
  plt.close()
  ```
* **Plotly:**
  If using Plotly, write the interactive figure to an HTML file named `output_plot.html`. Streamlit can render it using `st.components.v1.html`.
  ```python
  import plotly.express as px
  
  fig = px.bar(df_summary, x='Region', y='Revenue', color='Region', title='Revenue by Region')
  fig.write_html('output_plot.html')
  ```
