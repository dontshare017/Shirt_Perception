import pandas as pd
import ast

df = pd.read_csv('/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/60-79/raw_annotation/Shirt_Annotation_T60-69_csv.csv')
df['region_shape_attributes'] = df['region_shape_attributes'].apply(ast.literal_eval)
df['region_attributes'] = df['region_attributes'].apply(ast.literal_eval)

reformatted_df = pd.DataFrame(columns=['filename', 'x_start', 'x_end', 'y_start', 'y_end'])

for filename in df['filename'].unique():

    file_df = df[df['filename'] == filename]
    coordinates = {'x_start': None, 'x_end': None, 'y_start': None, 'y_end': None}

    for _, row in file_df.iterrows():
        if row['region_attributes']['start'] == "1":  # Start point
            coordinates['x_start'] = row['region_shape_attributes']['cx']
            coordinates['y_start'] = row['region_shape_attributes']['cy']
        elif row['region_attributes']['end'] == "1":  # End point
            coordinates['x_end'] = row['region_shape_attributes']['cx']
            coordinates['y_end'] = row['region_shape_attributes']['cy']

    new_row = pd.DataFrame([{
        'filename': filename,
        **coordinates
    }])
    reformatted_df = pd.concat([reformatted_df, new_row], ignore_index=True)


reformatted_df.to_csv('/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/60-79/annotations/annotation_T60-T69.csv', index=False)
