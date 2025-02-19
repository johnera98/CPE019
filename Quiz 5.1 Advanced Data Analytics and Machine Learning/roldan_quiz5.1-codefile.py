import pandas as pd
import numpy as np

#  ataset
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Create DataFrame
df = pd.DataFrame(exam_data, index=labels)

# SET A: Count rows and columns
rows, columns = df.shape
print('SET A')
print("Number of rows:", rows)
print("Number of columns:", columns)
print('\n')

# SET B: Select 'attempts' and 'qualify' columns
selected_columns = df[['attempts', 'qualify']]
print('SET B')
print("Selected columns:\n", selected_columns)
print('\n')

# SET C: Select rows where score is between 15 and 20 (inclusive)
filtered_rows = df[(df['score'] >= 15) & (df['score'] <= 20)]
print('SET C')
print("Rows with score between 15 and 20:\n", filtered_rows)
