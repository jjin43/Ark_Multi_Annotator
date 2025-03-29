import pandas as pd

diseases_6 = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other lesion', 'No finding']

# Read the input CSV file
input_file = 'image_labels_train.csv'
df = pd.read_csv(input_file)

# Filter the dataframe to include only the specified diseases
df = df[['image_id', 'rad_id'] + diseases_6]

# Iterate over unique rad_id values and save separate text files
for rad_id in df['rad_id'].unique():
    # Filter the dataframe by rad_id
    df_filtered = df[df['rad_id'] == rad_id]
    
    # Drop the rad_id column
    df_filtered = df_filtered.drop(columns=['rad_id'])
    
    # Add the prefix "train_jpeg/" to the image_id column
    df_filtered['image_id'] = 'train_jpeg/' + df_filtered['image_id'].astype(str)
    
    # Create a filename based on rad_id
    output_file = f'{rad_id}.txt'
    
    # Save the filtered dataframe to a new text file with space-separated values
    df_filtered.to_csv(output_file, index=False, sep=' ', header=False)

print("Text files have been successfully created.")