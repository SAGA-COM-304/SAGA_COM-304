import re
from pathlib import Path
import pandas as pd
import os
import cv2


log_dir = Path("dataset_module/downloads/log_files")
pattern = re.compile(r"ERROR: \[youtube\]\s+([^\s:]+):")
SEED = 42

RATIO_EVAL = 0.1
RATIO_TEST = 0.2
RATIO_TRAIN = 1 - RATIO_EVAL - RATIO_TEST

def _has_readable_frames(path: str) -> bool:
        """
        True  → OpenCV can decode at least one frame.
        False → the file is empty / truncated / wrong codec / unreadable.
        """
        cap = cv2.VideoCapture(path)
        ok, _ = cap.read()
        cap.release()
        return ok

results_ids = set()

file_column = 'video_clip_name'
ts_column = 'timestamp'
# names = ['video_clip_name','timestamp','class','group_name']
csv_file = '/work/com-304/SAGA/vggsound.csv'
path_to_videos = '/work/com-304/SAGA/raw/videos'
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
    
print(os.path.join(path_to_videos, df[file_column][0]))
mask = df.apply(lambda row: _has_readable_frames(os.path.join(path_to_videos, f"{row[file_column]}_{row[ts_column]}.mp4")), axis=1)
df = df[mask].reset_index(drop=True)

# AT THIS PART OUR DATASET IS CLEANED FROM ERRORS
count = df['class'].value_counts()
min_v = count.min()
combined_size = int(min_v / RATIO_TRAIN * (RATIO_TEST + RATIO_EVAL) )

balanced_df = (
    df
    .groupby('class', group_keys=False)               
    .sample(n=min_v, random_state=SEED)                           
)
train_idx = balanced_df.index
balanced_df = balanced_df.reset_index(drop=True)


remaining = df[~df.index.isin(train_idx)].reset_index(drop=True)

n_eval = int(combined_size * (RATIO_EVAL / (RATIO_EVAL + RATIO_TEST)))
n_test = combined_size - n_eval

# Create evaluation set
eval_df = (
    remaining
    .groupby('class', group_keys=False)
    .apply(lambda grp: grp.sample(n=min(len(grp), n_eval), random_state=SEED))
)

eval_idx = eval_df.index
eval_df = eval_df.reset_index(drop=True)

# Drop evaluation samples and create test set
remaining_after_eval = remaining.drop(eval_idx).reset_index(drop=True)
test_df = (
    remaining_after_eval
    .groupby('class', group_keys=False)
    .apply(lambda grp: grp.sample(n=min(len(grp), n_test), random_state=SEED))
    .reset_index(drop=True)
)

balanced_df['group_name'] = 'train'
eval_df['group_name'] = 'eval'
test_df['group_name'] = 'test'

stacked_df = pd.concat([balanced_df, eval_df, test_df], ignore_index=True)



print(balanced_df.shape, balanced_df['class'].value_counts())
print(eval_df.shape, eval_df['class'].value_counts())
print(test_df.shape, test_df['class'].value_counts())
print(stacked_df.shape, stacked_df['class'].value_counts())




# Save balanced dataset
# output_path = 'dataset_module/data/processed_data/balanced_vggsound.csv'
# balanced_df.to_csv(output_path, index=False)
# print(f"Saved balanced dataset to: {output_path}")    

# Save test dataset
# output_path = 'dataset_module/data/processed_data/test_vggsound.csv'
# test_df.to_csv(output_path, index=False)
# print(f"Saved test dataset to: {output_path}")

# Save eval dataset
# output_path = 'dataset_module/data/processed_data/eval_vggsound.csv'
# eval_df.to_csv(output_path, index=False)
# print(f"Saved eval dataset to: {output_path}")

#Save clean dataset
output_clean = 'dataset_module/data/processed_data/clean_vggsound.csv'
df.to_csv(output_clean, index=False)


# Save stacked dataset
output_path = 'dataset_module/data/processed_data/stacked_balanced_vggsound.csv'
stacked_df.to_csv(output_path, index=False)
print(f"Saved stacked dataset to: {output_path}")


