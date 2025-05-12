import re
from pathlib import Path
import pandas as pd
import os

log_dir = Path("dataset_module/downloads/log_files")
pattern = re.compile(r"ERROR: \[youtube\]\s+([^\s:]+):")
SEED = 42
RATIO_TEST = 0.2


results_ids = set()

file_column = 'video_clip_name'
# names = ['video_clip_name','timestamp','class','group_name']
csv_file = '/work/com-304/SAGA/vggsound.csv'
test = 0

for i in range(1, 21):
    filename = f"{i:02d}.log"           # "01.log", "02.log", ..., "20.log"
    file_path = log_dir / filename      
    if not file_path.is_file():
        print(f"error wrong file location : {file_path}")  

    with file_path.open('r', encoding='utf-8') as f:
        ids = []
        for line in f:
            m = pattern.search(line)
            if m:
                ids.append(m.group(1))
                results_ids.add(m.group(1))
        test += len(ids)

df = pd.read_csv(csv_file, header = 0)
df = df[~df[file_column].isin(results_ids)].reset_index(drop=True)
# TODO : Missing files (error but video exist)



# AT THIS PART OUR DATASET IS CLEANED FROM ERRORS
count = df['class'].value_counts()
min_v = count.min()
test_size = int(min_v * RATIO_TEST)

balanced_df = (
    df
    .groupby('class', group_keys=False)               
    .sample(n=min_v, random_state=SEED)                           
)

train_idx = balanced_df.index
balanced_df = balanced_df.reset_index(drop=True)


remaining = df[~df.index.isin(train_idx)].reset_index(drop=True)
test_df = (
    remaining
    .groupby('class', group_keys=False)               
    .apply(lambda grp: grp.sample(
        n=min(len(grp), test_size),
        random_state=SEED
    ))
    .reset_index(drop=True)                           
)

balanced_df['group_name'] = 'train'
test_df['group_name'] = 'test'

print(balanced_df.shape, balanced_df['class'].value_counts())
print(test_df.shape, test_df['class'].value_counts())

# Save balanced dataset
output_path = 'dataset_module/data/processed_data/balanced_vggsound.csv'
balanced_df.to_csv(output_path, index=False)
print(f"Saved balanced dataset to: {output_path}")

# Save test dataset
output_path = 'dataset_module/data/processed_data/test_vggsound.csv'
test_df.to_csv(output_path, index=False)
print(f"Saved test dataset to: {output_path}")
