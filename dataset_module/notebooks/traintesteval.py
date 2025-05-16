import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
R_TRAIN = 0.7
R_TEST  = 0.2
R_EVAL  = 0.1
assert R_TRAIN + R_EVAL + R_TEST == 1.0


df = pd.read_csv('/work/com-304/SAGA/clean_vggsound.csv')

train_df, temp_df = train_test_split(
    df,
    test_size=R_EVAL + R_TEST,
    stratify=df['class'],
    random_state=SEED
)

eval_df, test_df = train_test_split(
    temp_df,
    test_size= R_TEST / (R_TEST + R_EVAL) ,
    stratify=temp_df['class'],
    random_state=SEED
)

train_df['group_name'] = 'train'
eval_df['group_name']  = 'eval'
test_df['group_name']  = 'test'

stacked_df = pd.concat([train_df, test_df, eval_df], ignore_index=True)

stacked_df.to_csv('dataset_module/data/processed_data/vggsound_valid.csv', index=False)


