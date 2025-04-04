import pandas as pd

diseases_6 = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other lesion', 'No finding']

rad_distribution = {
    'R1': 1995,
    'R2': 3119,
    'R3': 2285,
    'R4': 1513,
    'R5': 2783,
    'R6': 2041,
    'R7': 1733,
    'R8': 6582,
    'R9': 6125,
    'R10': 6467,
    'R11': 1526,
    'R12': 1671,
    'R13': 1629,
    'R14': 1440,
    'R15': 1639,
    'R16': 1676,
    'R17': 776
}

# Read the input CSV file
input_file = '/scratch/jjin43/ark/VinDr-CXR/vindrcxr/image_labels_train.csv'
df = pd.read_csv(input_file)

# Filter the dataframe to include only the specified diseases
df = df[['image_id'] + diseases_6]

# Shuffle the dataframe to ensure random distribution
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into subsets based on the distribution
start_idx = 0
for rad, count in rad_distribution.items():
    subset = df.iloc[start_idx:start_idx + count].copy()  # create a copy here
    start_idx += count

    # Add the prefix "train_jpeg/" to the image ids using .loc
    subset.loc[:, 'image_id'] = "train_jpeg/" + subset['image_id'].astype(str)

    # Save the subset to a text file with columns separated by a space
    output_file = f'{rad}.txt'
    subset.to_csv(output_file, index=False, header=False, sep=" ")

# Verify that all entries are included and no duplicates exist
total_entries = sum(rad_distribution.values())
assert len(df) == total_entries, "The total number of entries does not match the distribution."
print("Text files have been successfully created.")
