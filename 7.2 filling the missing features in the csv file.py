import pandas as pd

# Load the CSV file (update the path)
file_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_vedio_dataset.csv"
df = pd.read_csv(file_path)

# Fill missing values (NaN) with 0
df.fillna(0, inplace=True)

# Save the cleaned dataset
cleaned_file_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_vedio_dataset_version2.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Cleaned CSV saved as {cleaned_file_path}")
