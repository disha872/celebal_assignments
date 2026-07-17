# Common Errors and Solutions in Pandas & Python

This document details common exceptions encountered during data science scripting and explains how to debug and fix them.

## 1. KeyError
* **Symptom:** `KeyError: 'column_name'`
* **Cause:** Trying to access a column that does not exist in the DataFrame. This is often caused by case mismatch, leading/trailing whitespace, or incorrect assumptions about columns after a merge/groupby.
* **Troubleshooting:**
  - Print columns: `print(df.columns.tolist())`
  - Strip whitespace from column names: `df.columns = df.columns.str.strip()`
  - Verify that the column was not dropped or renamed in a previous step.
  - If selecting columns after `.groupby()`, make sure you use `.reset_index()` or access the group key correctly.

## 2. AttributeError: 'DataFrame' object has no attribute 'X'
* **Symptom:** `AttributeError: 'DataFrame' object has no attribute 'append'`
* **Cause:** Pandas 2.0+ removed the `.append()` method for DataFrames.
* **Solution:** Use `pd.concat()` instead.
  ```python
  # Old syntax (errors in Pandas 2.0+):
  # df = df1.append(df2)
  # New syntax:
  df = pd.concat([df1, df2], ignore_index=True)
  ```

* **Symptom:** `AttributeError: 'Series' object has no attribute 'dt'`
* **Cause:** Calling date/time attributes on a column that is not of datetime type.
* **Solution:** Convert the column to datetime first:
  ```python
  df['Date'] = pd.to_datetime(df['Date'])
  years = df['Date'].dt.year
  ```

## 3. ValueError
* **Symptom:** `ValueError: Cannot convert float NaN to integer`
* **Cause:** Attempting to convert a column with missing values (`NaN`) to `int`.
* **Solution:** Handle missing values first (`df.fillna()`) or use the nullable integer type (`'Int64'`).
  ```python
  df['Age'] = df['Age'].fillna(0).astype(int)
  # OR use pandas nullable integer:
  df['Age'] = df['Age'].astype('Int64')
  ```

* **Symptom:** `ValueError: could not convert string to float: 'abc'`
* **Cause:** Parsing non-numeric strings to numeric.
* **Solution:** Use `pd.to_numeric` with `errors='coerce'` to turn invalid values into `NaN`, then fill or drop them.
  ```python
  df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
  ```

## 4. TypeError
* **Symptom:** `TypeError: unsupported operand type(s) for +: 'float' and 'str'`
* **Cause:** Mixing numeric types and string types in arithmetic operations.
* **Solution:** Cast columns to the same type.
  ```python
  df['Price'] = df['Price'].astype(float)
  df['Tax'] = df['Tax'].astype(float)
  df['Total'] = df['Price'] + df['Tax']
  ```

## 5. SettingWithCopyWarning
* **Symptom:** `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.`
* **Cause:** Modifying a slice of a DataFrame directly instead of explicitly copying it.
* **Solution:** Use `.copy()` when slicing, or use `.loc[row_indexer, col_indexer] = value`.
  ```python
  # Bad:
  df_sub = df[df['Age'] > 30]
  df_sub['Category'] = 'Senior'
  # Good:
  df_sub = df[df['Age'] > 30].copy()
  df_sub['Category'] = 'Senior'
  ```
