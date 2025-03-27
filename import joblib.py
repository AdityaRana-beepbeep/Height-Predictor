import joblib
import pandas as pd

# Adjust file path for Windows if necessary
file_path = "C:\Users\adity\OneDrive\Desktop\Project\Simple Linear Regression Project\package_predictor.joblib"  # Update this path

# Load the joblib file
data = joblib.load(file_path)

# Check the type of data
print(f"Data type: {type(data)}")

# If it's a dictionary or list-like, convert it to DataFrame
if isinstance(data, (list, dict)):
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)
    print("CSV file saved successfully.")
else:
    print("Loaded data is not in a suitable format for DataFrame conversion.")
