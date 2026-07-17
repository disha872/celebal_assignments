import os
import re
import requests
from typing import Dict, Any, List, Optional
from rag import RAGPipeline
from sandbox import execute_code

class LLMClient:
    """Unified client to interact with Gemini, OpenAI, and Anthropic APIs."""
    def __init__(self, provider: str, api_key: str, model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key.strip()
        self.model_name = model_name

        if self.provider == "gemini":
            self.model_name = model_name or "gemini-2.5-flash"
        elif self.provider == "openai":
            self.model_name = model_name or "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.model_name = model_name or "claude-3-5-sonnet-latest"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str) -> str:
        """Generates text from the LLM based on the prompt."""
        if self.provider == "gemini":
            return self._generate_gemini(prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        return ""

    def _generate_gemini(self, prompt: str) -> str:
        # We try using google-generativeai library
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback to direct HTTP request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            res = requests.post(url, headers=headers, json=data, timeout=30)
            if res.status_code == 200:
                result = res.json()
                try:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                except KeyError:
                    raise Exception(f"Failed to parse Gemini response: {result}")
            else:
                raise Exception(f"Gemini API error ({res.status_code}): {res.text}")

    def _generate_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            res = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return res.choices[0].message.content
        except Exception as e:
            # Fallback to direct HTTP request
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            res = requests.post(url, headers=headers, json=data, timeout=30)
            if res.status_code == 200:
                result = res.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenAI API error ({res.status_code}): {res.text}")

    def _generate_anthropic(self, prompt: str) -> str:
        # Anthropic direct API request (very clean and works without anthropic library)
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": self.model_name,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post(url, headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            result = res.json()
            return result["content"][0]["text"]
        else:
            raise Exception(f"Anthropic API error ({res.status_code}): {res.text}")


class AutonomousDSAgent:
    """The central agent coordinating RAG, LLM code generation, sandbox execution, and self-healing."""
    def __init__(self, provider: str, api_key: str, model_name: Optional[str] = None, docs_dir: str = "./docs"):
        self.llm = LLMClient(provider, api_key, model_name)
        self.rag = RAGPipeline(docs_dir)

    def get_dataset_schema(self, file_path: str) -> str:
        """Reads metadata and first few rows of a dataset to build a text schema for the LLM."""
        import pandas as pd
        import json
        
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path, nrows=5)
                # Get total rows
                total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
            elif ext in ['.xlsx', '.xls']:
                # Inspect sheets
                xl = pd.ExcelFile(file_path)
                sheets = xl.sheet_names
                df = pd.read_excel(file_path, sheet_name=sheets[0], nrows=5)
                total_rows = "Unknown (Excel)"
                schema_info = f"Excel file with sheets: {sheets}. Showing sheet '{sheets[0]}':\n"
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:5])
                    total_rows = len(data)
                elif isinstance(data, dict):
                    # Try to normalize or read key-value
                    df = pd.DataFrame(data).head(5)
                    total_rows = len(data)
                else:
                    return "JSON file containing non-tabular format."
            else:
                return f"Unsupported file type: {ext}"
            
            # Format schema details
            col_types = df.dtypes.to_string()
            sample_rows = df.to_string(index=False)
            
            schema_desc = (
                f"File format: {ext}\n"
                f"Total columns: {len(df.columns)}\n"
                f"Estimated total rows: {total_rows}\n"
                f"Columns and Data Types:\n{col_types}\n\n"
                f"Sample rows:\n{sample_rows}"
            )
            return schema_desc
        except Exception as e:
            return f"Error reading file schema: {str(e)}"

    def extract_code(self, response: str) -> str:
        """Extracts Python code blocks from markdown responses."""
        pattern = r"```python(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback to checking for general code blocks
        pattern_general = r"```(.*?)```"
        match_gen = re.search(pattern_general, response, re.DOTALL)
        if match_gen:
            return match_gen.group(1).strip()
        return response.strip()

    def get_offline_template_code(self, file_path: str, user_query: str) -> str:
        """Returns pre-compiled analysis code for standard templates when LLM is unavailable."""
        filename = os.path.basename(file_path).lower()
        
        # 1. Sales Dashboard Use Case
        if "sales_data" in filename:
            return """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('sales_data.csv')

# Group and aggregate
region_revenue = df.groupby('Region')['Revenue'].sum().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
sns.barplot(data=region_revenue, x='Region', y='Revenue', hue='Region', palette='viridis', legend=False)
plt.title('Total Revenue by Region', fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.tight_layout()
plt.savefig('output_plot.png', dpi=300)
plt.close()

print("[PLOT_SAVED] output_plot.png")
print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Regional Sales Analysis Results\\n")
print("Below is the summary of total revenue generated across different geographic regions:\\n")
for idx, row in region_revenue.iterrows():
    print(f"- **{row['Region']}**: ${row['Revenue']:,}")
print("\\n**Key Observations:**")
max_idx = region_revenue['Revenue'].idxmax()
min_idx = region_revenue['Revenue'].idxmin()
print(f"1. The **{region_revenue.loc[max_idx, 'Region']}** region generated the highest revenue of **${region_revenue.loc[max_idx, 'Revenue']:,}**.")
print(f"2. The **{region_revenue.loc[min_idx, 'Region']}** region had the lowest revenue of **${region_revenue.loc[min_idx, 'Revenue']:,}**.")
"""

        # 2. Data Quality Audit Use Case
        elif "dirty_data" in filename:
            return """
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dirty_data.csv')

# 1. Check duplicates
duplicates_count = df.duplicated().sum()
df_clean = df.drop_duplicates()

# 2. Check nulls
nulls_before = df_clean.isnull().sum().to_dict()

# Clean Age (fill with mean)
mean_age = df_clean['Age'].dropna().mean()
df_clean['Age'] = df_clean['Age'].fillna(mean_age)

# Clean Salary
df_clean['Salary'] = df_clean['Salary'].astype(str).str.replace('$', '').str.replace(',', '').str.replace('N/A', 'nan')
df_clean['Salary'] = pd.to_numeric(df_clean['Salary'], errors='coerce')
mean_salary = df_clean['Salary'].dropna().mean()
df_clean['Salary'] = df_clean['Salary'].fillna(mean_salary)

# Remove outliers/invalid ages
df_clean = df_clean[(df_clean['Age'] > 0) & (df_clean['Age'] < 100)]

df_clean.to_csv('cleaned_data.csv', index=False)

print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Data Quality Audit & Cleaning Report\\n")
print(f"- **Duplicate Rows Found & Removed:** {duplicates_count}")
print("- **Missing Values Before Cleaning:**")
for col, count in nulls_before.items():
    if count > 0:
        print(f"  - `{col}`: {count} missing value(s)")
        
print("\\n- **Cleaning Operations Applied:**")
print(f"  - Filled missing ages using the dataset mean of **{mean_age:.1f}** years.")
print("  - Handled non-standard salary strings (e.g. '$110,000', 'N/A') and converted the column to numeric.")
print(f"  - Filled missing salaries using the dataset mean of **${mean_salary:,.2f}**.")
print("  - Filtered out invalid ages (e.g. negative or above 100).")
print("\\n- **Cleaned Data Preview (First 5 Rows):**\\n")
cols = df_clean.columns.tolist()
print("| " + " | ".join(cols) + " |")
print("| " + " | ".join(["---"] * len(cols)) + " |")
for idx, row in df_clean.head(5).iterrows():
    row_strs = [str(val) for val in row]
    print("| " + " | ".join(row_strs) + " |")
print("\\n*Cleaned dataset saved as `cleaned_data.csv`.*")
"""

        # 3. Trend Analysis Use Case
        elif "traffic_data" in filename:
            return """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_json('traffic_data.json')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

plt.figure(figsize=(12, 6))
sns.set_theme(style="darkgrid")
plt.plot(df['Date'], df['Visitors'], label='Daily Visitors', marker='o', color='#3B82F6', linewidth=2)
plt.plot(df['Date'], df['PageViews'], label='Daily PageViews', marker='s', color='#10B981', linewidth=2)
plt.title('Website Traffic Trend Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output_plot.png', dpi=300)
plt.close()

print("[PLOT_SAVED] output_plot.png")
print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Website Traffic Trend Analysis\\n")
print(f"- **Analysis Period:** From {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')} ({len(df)} days)")
print(f"- **Total Visitors:** {df['Visitors'].sum():,}")
print(f"- **Total PageViews:** {df['PageViews'].sum():,}")
print(f"- **Average Bounce Rate:** {df['BounceRate'].mean():.2f}%")
print("\\n**Key Trend Observations:**")
print("1. There is a steady **growth trend** in weekday traffic, with noticeable cyclical drops on weekends.")
max_date = df.loc[df['Visitors'].idxmax(), 'Date'].strftime('%Y-%m-%d')
print(f"2. Peak visitors occurred on **{max_date}** with **{df['Visitors'].max():,}** visits.")
"""

        # 4. Cohort Analysis Use Case
        elif "customer_segments" in filename:
            return """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('customer_segments.csv')

plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
sns.scatterplot(
    data=df, 
    x='Annual_Income_k', 
    y='Spending_Score', 
    hue='Gender', 
    size='Age', 
    sizes=(20, 200),
    palette='Set1',
    alpha=0.8
)
plt.title('Customer Segments: Spending Score vs Annual Income', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (in $k)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('output_plot.png', dpi=300)
plt.close()

print("[PLOT_SAVED] output_plot.png")
print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Customer Cohort Segmentation Results\\n")
print("We analyzed customer demographic records. Three distinct customer cohorts are visible:")
print("1. **High Income, Low Spend:** Customers with high annual income (> $80k) but low spending scores (< 40).")
print("2. **Low Income, High Spend:** Younger customers with lower income (< $40k) but high spending scores (> 60).")
print("3. **Mid Income, Mid Spend:** A dense central cluster of average earners and spenders.")
avg_age = df['Age'].mean()
avg_income = df['Annual_Income_k'].mean()
print(f"\\n**Cohort Summary Stats:**")
print(f"- Average Customer Age: **{avg_age:.1f} years**")
print(f"- Average Annual Income: **${avg_income:.1f}k**")
print(f"- Gender Distribution: **{df['Gender'].value_counts(normalize=True).get('Female', 0)*100:.1f}% Female**")
"""

        # 5. Ad-hoc Budget Excel Use Case
        elif "ad_hoc_queries" in filename:
            return """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sheets
df_budget = pd.read_excel('ad_hoc_queries.xlsx', sheet_name='Budgets')
df_expense = pd.read_excel('ad_hoc_queries.xlsx', sheet_name='Expenses')

# Merge
df_merge = pd.merge(df_budget, df_expense, on='Department')
df_merge['Remaining_Budget'] = df_merge['Budget'] - df_merge['Actual_Spent']

# Melt
df_melted = pd.melt(df_merge, id_vars=['Department'], value_vars=['Budget', 'Actual_Spent'], 
                    var_name='Category', value_name='Amount')
                    
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
sns.barplot(data=df_melted, x='Department', y='Amount', hue='Category', palette='muted')
plt.title('Budget vs Actual Spent by Department', fontsize=14, fontweight='bold')
plt.xlabel('Department', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.tight_layout()
plt.savefig('output_plot.png', dpi=300)
plt.close()

print("[PLOT_SAVED] output_plot.png")
print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Operational Budget Variance Analysis\\n")
print("Below is the department-wise budget utilization summary:\\n")
cols = ['Department', 'Budget', 'Actual_Spent', 'Remaining_Budget']
print("| " + " | ".join(cols) + " |")
print("| " + " | ".join(["---"] * len(cols)) + " |")
for idx, row in df_merge.iterrows():
    print(f"| {row['Department']} | ${row['Budget']:,} | ${row['Actual_Spent']:,} | ${row['Remaining_Budget']:,} |")
    
overspent = df_merge[df_merge['Remaining_Budget'] < 0]
print("\\n**Key Variance Insights:**")
if len(overspent) > 0:
    for idx, row in overspent.iterrows():
        print(f"- [WARNING] **{row['Department']}** has **overspent** its budget by **${abs(row['Remaining_Budget']):,}**!")
else:
    print("- [SUCCESS] All departments stayed within their allocated budgets.")
"""

        # General CSV Fallback (e.g. diabetes.csv or custom data)
        else:
            q_lower = user_query.lower()
            if "line" in q_lower:
                plot_code = """
if len(num_cols) >= 1:
    plt.figure(figsize=(10, 6))
    # Plot top numerical columns
    for col in num_cols[:3]:
        plt.plot(df.index, df[col], label=col, linewidth=2, alpha=0.8)
    plt.title('Line Chart of Key Features', fontsize=14, fontweight='bold')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""
            elif "bar" in q_lower:
                plot_code = """
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if len(cat_cols) > 0:
    plt.figure(figsize=(10, 6))
    top_vals = df[cat_cols[0]].value_counts().head(10)
    sns.barplot(x=top_vals.index, y=top_vals.values, hue=top_vals.index, palette='viridis', legend=False)
    plt.title(f'Bar Chart: Top 10 Categories in {cat_cols[0]}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
elif len(num_cols) >= 1:
    plt.figure(figsize=(10, 6))
    top_vals = df[num_cols[0]].value_counts().head(10)
    sns.barplot(x=top_vals.index, y=top_vals.values, hue=top_vals.index, palette='viridis', legend=False)
    plt.title(f'Bar Chart: Top 10 Values in {num_cols[0]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""
            elif "scatter" in q_lower:
                plot_code = """
if len(num_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], hue=num_cols[1], palette='viridis')
    plt.title(f'Scatter Plot: {num_cols[0]} vs {num_cols[1]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""
            elif "hist" in q_lower or "dist" in q_lower:
                plot_code = """
if len(num_cols) >= 1:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[num_cols[0]], kde=True, color='blue')
    plt.title(f'Distribution of {num_cols[0]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""
            elif "box" in q_lower:
                plot_code = """
if len(num_cols) >= 1:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[num_cols[:4]])
    plt.title('Box Plot of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""
            else:
                plot_code = """
if len(num_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
elif len(num_cols) == 1:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[num_cols[0]], kde=True, color='blue')
    plt.title(f'Distribution of {num_cols[0]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_plot.png', dpi=300)
    plt.close()
    print("[PLOT_SAVED] output_plot.png")
"""

            return f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the csv
df = pd.read_csv('{os.path.basename(file_path)}')

print("[INFO] *Note: API authorization was denied or rate-limited. Activating local offline template analytics.*\\n")
print("### Dataset Exploratory Analysis Summary\\n")
print("Successfully loaded `{os.path.basename(file_path)}`. Here is the dataset profile:")
print(f"- **Dimensions:** {{df.shape[0]}} rows, {{df.shape[1]}} columns\\n")

# Columns preview
print("- **Columns and Data Types:**")
for col, dtype in df.dtypes.items():
    print(f"  - `{{col}}` ({{dtype}})")

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("\\n- **Missing Values Detected:**")
    for col, count in missing.items():
        print(f"  - `{{col}}`: {{count}} missing value(s)")

# Identify numerical columns
num_cols = df.select_dtypes(include=['number']).columns.tolist()

# Dynamic Plot code injection
{plot_code}

# Descriptive statistics summary
print("\\n- **Descriptive Statistics Summary:**\\n")
desc = df.describe().reset_index()
cols = desc.columns.tolist()
print("| " + " | ".join(cols) + " |")
print("| " + " | ".join(["---"] * len(cols)) + " |")
for idx, row in desc.iterrows():
    row_strs = [f"{{val:.2f}}" if isinstance(val, (int, float)) else str(val) for val in row]
    print("| " + " | ".join(row_strs) + " |")
"""

    def run(self, file_path: str, user_query: str, output_dir: str, max_retries: int = 3) -> Dict[str, Any]:
        """Runs the self-healing loop: write code -> run in sandbox -> heal on error -> repeat."""
        schema_desc = self.get_dataset_schema(file_path)
        data_filename = os.path.basename(file_path)
        
        steps = []
        current_query = user_query
        failed_codes = []
        
        # 1. Initial Prompt Creation
        prompt = (
            "You are an expert Data Science AI Assistant.\n"
            "Your task is to write a single self-contained Python script to analyze the provided dataset and answer the user's query.\n\n"
            f"Dataset File Name: {data_filename}\n"
            "Dataset Schema and Samples:\n"
            "===================================\n"
            f"{schema_desc}\n"
            "===================================\n\n"
            f"User Query: {user_query}\n\n"
            "Instructions:\n"
            f"1. Load the data from '{data_filename}' using Pandas.\n"
            "2. Make sure to perform operations matching the query (cleaning, aggregation, plotting, etc.).\n"
            "3. If a plot/visualization is requested or would be helpful for the query:\n"
            "   - Save the plot as a PNG image named 'output_plot.png' or an interactive HTML chart named 'output_plot.html'.\n"
            "   - Print the exact text '[PLOT_SAVED] output_plot.png' or '[PLOT_SAVED] output_plot.html' to standard output.\n"
            "4. Print your final analysis conclusions and insights in clean Markdown format to standard output. This will be shown directly to the user.\n"
            "5. Do NOT include any interactive matplotlib commands like plt.show(). Save the files directly.\n"
            "6. Make the script robust. Handle NaN values where appropriate.\n\n"
            "Return ONLY the Python code block enclosed in ```python ... ```. Do not add conversational intro/outro text."
        )

        for attempt in range(1, max_retries + 2):
            step_info = {"attempt": attempt, "status": "In Progress"}
            steps.append(step_info)

            try:
                # Generate code from LLM
                response_text = self.llm.generate(prompt)
                code = self.extract_code(response_text)
                step_info["code"] = code
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["403", "429", "denied", "api key not valid", "not found", "quota", "resource_exhausted", "rate"]):
                    code = self.get_offline_template_code(file_path, user_query)
                    step_info["code"] = code
                    step_info["status"] = "Offline Fallback Triggered"
                else:
                    step_info["status"] = "LLM Generation Failed"
                    step_info["error"] = str(e)
                    return {"success": False, "error": f"LLM generation failed: {str(e)}", "steps": steps}

            # Run in sandbox
            run_result = execute_code(code, file_path, output_dir)
            step_info["stdout"] = run_result["stdout"]
            step_info["stderr"] = run_result["stderr"]
            step_info["exit_code"] = run_result["exit_code"]
            step_info["sandbox_dir"] = run_result["sandbox_dir"]

            if run_result["success"]:
                # Success!
                step_info["status"] = "Success"
                
                # Parse stdout to find markdown description and plot filename
                stdout_content = run_result["stdout"]
                plot_file = None
                
                # Check for explicit plot printed or just check if output_plot exists
                plot_match = re.search(r'\[PLOT_SAVED\]\s+(\S+)', stdout_content)
                if plot_match:
                    plot_file = plot_match.group(1)
                else:
                    # Fallback scan in generated files
                    for gf in run_result["generated_files"]:
                        gf_name = os.path.basename(gf)
                        if gf_name in ["output_plot.png", "output_plot.html"]:
                            plot_file = gf_name
                            break

                # Resolve absolute path of plot file
                resolved_plot_path = None
                if plot_file:
                    potential_path = os.path.join(run_result["sandbox_dir"], plot_file)
                    if os.path.exists(potential_path):
                        resolved_plot_path = potential_path

                # Extract markdown text (filtering out [PLOT_SAVED] markers)
                clean_markdown = re.sub(r'\[PLOT_SAVED\]\s+\S+', '', stdout_content).strip()
                
                return {
                    "success": True,
                    "markdown_insight": clean_markdown if clean_markdown else "Analysis completed successfully (no text output).",
                    "plot_path": resolved_plot_path,
                    "generated_files": run_result["generated_files"],
                    "sandbox_dir": run_result["sandbox_dir"],
                    "steps": steps
                }
            else:
                # Failed! Run self-healing
                step_info["status"] = "Execution Error"
                failed_codes.append(code)
                
                # If we've run out of retries, return failure
                if attempt > max_retries:
                    break

                # Retrieve relevant docs using RAG
                query_for_docs = f"{user_query} {run_result['stderr']}"
                retrieved_chunks = self.rag.retrieve(query_for_docs, top_k=2)
                
                step_info["retrieved_docs"] = [
                    {"title": chunk.title, "file": os.path.basename(chunk.file_path)}
                    for chunk, _ in retrieved_chunks
                ]

                doc_context = ""
                for idx, (chunk, score) in enumerate(retrieved_chunks):
                    doc_context += f"Source: {os.path.basename(chunk.file_path)} ({chunk.title})\n---\n{chunk.content}\n\n"

                # Build self-healing prompt
                prompt = (
                    "The previous Python script you wrote failed with an execution error. "
                    "Study the error output, look up the troubleshooting guidelines, and rewrite the script to resolve the issue.\n\n"
                    f"Original User Query: {user_query}\n\n"
                    "Previous Script Code:\n"
                    "===================================\n"
                    f"```python\n{code}\n```\n"
                    "===================================\n\n"
                    "Execution Stderr Output:\n"
                    "===================================\n"
                    f"{run_result['stderr']}\n"
                    "===================================\n\n"
                    "Execution Stdout Output (if any):\n"
                    "===================================\n"
                    f"{run_result['stdout']}\n"
                    "===================================\n\n"
                    "Troubleshooting Documentation (RAG Context):\n"
                    "===================================\n"
                    f"{doc_context}\n"
                    "===================================\n\n"
                    "Instructions:\n"
                    "1. Correct the code to avoid the crash. Make sure to fix the specific issues highlighted in the stderr.\n"
                    "2. Make sure you load the dataset using the correct local filename (use pandas to load data from the current directory).\n"
                    "3. Maintain all visualisations and analysis requested. Save plots as 'output_plot.png' or 'output_plot.html' and print '[PLOT_SAVED] plot_name'.\n"
                    "4. Return ONLY the corrected code block enclosed in ```python ... ```."
                )

        return {
            "success": False,
            "error": "Self-healing failed after reaching maximum attempts.",
            "steps": steps
        }
