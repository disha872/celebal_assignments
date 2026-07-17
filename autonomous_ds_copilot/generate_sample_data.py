import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def make_sample_data():
    os.makedirs("./sample_data", exist_ok=True)
    print("Generating sample datasets...")

    # 1. Sales Dashboard Dataset (CSV)
    np.random.seed(42)
    regions = ['North', 'East', 'South', 'West']
    reps = ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan']
    products = ['Laptop', 'Tablet', 'Monitor', 'Keyboard', 'Mouse']
    prices = [1200, 450, 300, 80, 30]
    
    dates = [datetime(2026, 1, 1) + timedelta(days=int(i)) for i in range(100)]
    
    sales_list = []
    for i in range(200):
        rep = np.random.choice(reps)
        region = np.random.choice(regions)
        prod_idx = np.random.randint(0, len(products))
        product = products[prod_idx]
        unit_price = prices[prod_idx]
        units = np.random.randint(1, 10)
        revenue = units * unit_price
        date = np.random.choice(dates).strftime('%Y-%m-%d')
        
        sales_list.append({
            'Date': date,
            'Region': region,
            'Representative': rep,
            'Product': product,
            'Units': units,
            'UnitPrice': unit_price,
            'Revenue': revenue
        })
        
    df_sales = pd.DataFrame(sales_list)
    df_sales.to_csv('./sample_data/sales_data.csv', index=False)
    print("- Saved sales_data.csv")

    # 2. Data Quality Audit Dataset (CSV with intentional anomalies)
    dirty_list = [
        {"ID": 101, "Name": "Aarav Sharma", "Age": 28, "Salary": "85000", "Department": "Engineering", "JoinDate": "2024-01-15"},
        {"ID": 102, "Name": "Diya Patel", "Age": 32, "Salary": "92000", "Department": "Marketing", "JoinDate": "2023-05-10"},
        {"ID": 103, "Name": "Vivaan Sen", "Age": None, "Salary": "78000", "Department": "Engineering", "JoinDate": "2024-03-01"}, # Missing Age
        {"ID": 104, "Name": "Ananya Rao", "Age": 45, "Salary": "120000", "Department": "Finance", "JoinDate": "2022-11-20"},
        {"ID": 105, "Name": "Kabir Singh", "Age": 29, "Salary": None, "Department": "HR", "JoinDate": "2024-06-01"}, # Missing Salary
        {"ID": 102, "Name": "Diya Patel", "Age": 32, "Salary": "92000", "Department": "Marketing", "JoinDate": "2023-05-10"}, # Duplicate Row
        {"ID": 106, "Name": "Sai Prasad", "Age": 38, "Salary": "$110,000", "Department": "Engineering", "JoinDate": "2021-08-14"}, # Bad formatting (string with $ and comma)
        {"ID": 107, "Name": "Riya Gupta", "Age": 150, "Salary": "65000", "Department": "Sales", "JoinDate": "2025-01-05"}, # Outlier Age
        {"ID": 108, "Name": "Ishaan Nair", "Age": -5, "Salary": "50000", "Department": "Sales", "JoinDate": "2025-02-10"}, # Negative Age
        {"ID": 109, "Name": "Meera Joshi", "Age": 24, "Salary": "N/A", "Department": "HR", "JoinDate": "2024-10-15"}, # Text Salary
        {"ID": 110, "Name": "Aditya Verma", "Age": 31, "Salary": "88000", "Department": "Engineering", "JoinDate": "2024-07-22"}
    ]
    df_dirty = pd.DataFrame(dirty_list)
    df_dirty.to_csv('./sample_data/dirty_data.csv', index=False)
    print("- Saved dirty_data.csv")

    # 3. Trend Analysis Dataset (JSON time series)
    traffic_list = []
    base_date = datetime(2026, 6, 1)
    for i in range(30):
        date = (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
        # Add a growth trend and weekend drops
        is_weekend = (base_date + timedelta(days=i)).weekday() >= 5
        base_visitors = 1000 + i * 25
        if is_weekend:
            visitors = int(base_visitors * 0.6 + np.random.randint(-50, 50))
        else:
            visitors = int(base_visitors + np.random.randint(-100, 100))
            
        pageviews = int(visitors * np.random.uniform(2.1, 2.8))
        bounce_rate = round(np.random.uniform(40.0, 55.0), 2)
        
        traffic_list.append({
            "Date": date,
            "Visitors": visitors,
            "PageViews": pageviews,
            "BounceRate": bounce_rate
        })
        
    with open('./sample_data/traffic_data.json', 'w', encoding='utf-8') as f:
        json.dump(traffic_list, f, indent=2)
    print("- Saved traffic_data.json")

    # 4. Cohort Analysis / Customer Segmentation (CSV)
    customer_list = []
    for cid in range(1001, 1101):
        gender = np.random.choice(['Male', 'Female'])
        age = int(np.random.randint(18, 70))
        income = int(np.random.randint(15, 140)) # Annual income in thousands
        # Create structured scores based on age and income (3 segments)
        if income < 40:
            score = int(np.random.randint(60, 99)) if age < 35 else int(np.random.randint(5, 40))
        elif income > 80:
            score = int(np.random.randint(70, 99)) if age < 40 else int(np.random.randint(10, 45))
        else:
            score = int(np.random.randint(40, 60))
            
        customer_list.append({
            "CustomerID": cid,
            "Gender": gender,
            "Age": age,
            "Annual_Income_k": income,
            "Spending_Score": score
        })
    df_customers = pd.DataFrame(customer_list)
    df_customers.to_csv('./sample_data/customer_segments.csv', index=False)
    print("- Saved customer_segments.csv")

    # 5. Ad-hoc Queries Budget Dataset (Excel - Multi-Sheet)
    budget_data = {
        'Department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations'],
        'Budget': [500000, 300000, 450000, 120000, 150000, 200000],
        'Manager': ['Siddharth Singh', 'Neha Sharma', 'Vikram Malhotra', 'Pooja Iyer', 'Rahul Verma', 'Amit Goel']
    }
    
    expense_data = {
        'Department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations'],
        'Actual_Spent': [480000, 320000, 410000, 115000, 148000, 210000],
        'Q1_Spent': [110000, 75000, 95000, 28000, 35000, 48000],
        'Q2_Spent': [120000, 85000, 105000, 29000, 37000, 52000],
        'Q3_Spent': [125000, 80000, 100000, 28000, 36000, 50000],
        'Q4_Spent': [125000, 80000, 110000, 30000, 40000, 60000]
    }
    
    df_budget = pd.DataFrame(budget_data)
    df_expense = pd.DataFrame(expense_data)
    
    with pd.ExcelWriter('./sample_data/ad_hoc_queries.xlsx', engine='openpyxl') as writer:
        df_budget.to_excel(writer, sheet_name='Budgets', index=False)
        df_expense.to_excel(writer, sheet_name='Expenses', index=False)
    print("- Saved ad_hoc_queries.xlsx")
    print("All sample datasets generated successfully!")

if __name__ == "__main__":
    make_sample_data()
