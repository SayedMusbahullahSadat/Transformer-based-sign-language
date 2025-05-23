import csv

def clean_landmarks_csv(
    input_csv_path,
    output_csv_path,
    expected_num_columns=64  # 64 features + 1 label
):
    """
    Removes rows which:
      - Do not match the expected number of columns.
      - Have all-zero feature values.
    Then writes the cleaned rows to a new CSV file.
    """

    with open(input_csv_path, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        header = next(reader)  # First row is the header

        # Prepare a list to hold valid rows
        valid_rows = [header]  # Keep the header as is

        for row in reader:
            # Check if the row has the right number of columns
            if len(row) != expected_num_columns:
                # Skip this row if the number of columns is incorrect
                continue

            # Separate features (first 126 columns) from label (last column)
            features = row[:-1]  # all but last
            label = row[-1]

            # Safely convert features to floats; skip row if there's an error
            try:
                features_float = [float(x) for x in features]
            except ValueError:
                # If conversion fails for any feature, skip this row
                continue

            # Check if all feature values are zero
            if all(value == 0.0 for value in features_float):
                # If all zeros, skip this row
                continue

            # If we reach here, the row is valid - add it
            valid_rows.append(row)

    # Write valid rows to a new CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(valid_rows)


if __name__ == "__main__":
    input_csv = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\landmarks_dataset.csv"
    output_csv = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_landmarks_dataset.csv"

    clean_landmarks_csv(input_csv, output_csv)
    print(f"Cleaned CSV saved to: {output_csv}")
