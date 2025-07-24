import pandas as pd
import numpy as np

# Create a dummy DataFrame
dummy_data = {
    'Time': [0, 0],  # Example timestamps
    **{f'V{i}': np.random.uniform(-3, 3, 2) for i in range(1, 29)},  # Random values for V1-V28
    'Amount': [150.0, 20.0]  # Example amounts
}

# Convert to DataFrame and save
df = pd.DataFrame(dummy_data)
df.to_csv("test_transactions.csv", index=False)
print("CSV file 'test_transactions.csv' created!")