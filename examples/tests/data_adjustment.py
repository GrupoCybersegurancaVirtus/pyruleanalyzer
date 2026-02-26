import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_path = "examples/data/mushrooms_train.csv"
test_path = "examples/data/mushrooms_test.csv"

# Move the "class" column (target) to the last position in both train and test datasets
def move_class_col_to_last(csv_path):
    df = pd.read_csv(csv_path)
    if 'class' in df.columns:
        cols = [col for col in df.columns if col != 'class'] + ['class']
        df = df[cols]
        df.to_csv(csv_path, index=False)

move_class_col_to_last(train_path)
move_class_col_to_last(test_path)

# Load the train and test datasets
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Apply label encoding to all object (string/categorical) columns using the train set fit
label_encoders = {}
for col in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

# Save the encoded datasets to new CSV files
encoded_train_path = "examples/data/mushrooms_encoded_train.csv"
encoded_test_path = "examples/data/mushrooms_encoded_test.csv"
df_train.to_csv(encoded_train_path, index=False)
df_test.to_csv(encoded_test_path, index=False)

