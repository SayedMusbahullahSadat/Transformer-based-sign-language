
import pandas as pd

# Load the dataset
file_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\Datasets\WLASL_dataset_video_landmarks.csv"
df = pd.read_csv(file_path)

# Get the correct number of columns (assuming first row is the correct structure)
expected_columns = df.columns.tolist()

# Function to clean each row
def clean_row(row):
    # If a row has more columns than expected, trim it
    row = row[:len(expected_columns)]
    # If a row has fewer columns, fill missing values with 0
    row += [0] * (len(expected_columns) - len(row))
    return row

# Apply row cleaning
df_cleaned = pd.DataFrame([clean_row(row.tolist()) for _, row in df.iterrows()], columns=expected_columns)

# Save the cleaned dataset
cleaned_file_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_vedio_dataset.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned CSV saved as {cleaned_file_path}")
