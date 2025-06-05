import os
import pandas as pd
import numpy as np
from pathlib import Path

class Maplabel:
    def __init__(self, path_csv):
        """
        Args:
            path_csv (str): Path to a CSV file (no header) with two columns [label, count].
        """
        self.path_csv = path_csv
        self.label2id  = self._build_map()
        self.nb_label  = len(self.label2id)

    def _build_map(self):
        df = pd.read_csv(self.path_csv, header=None, names=['label','count'])
        df = df.sort_values(['count','label'], ascending=[False,True])
        return {lab: idx for idx, lab in enumerate(df['label'].tolist())}

    def add_special_char(self, char:str):
        """
        Add a special character to the label mapping if not already present.
        Args:
            char (str): The special character to add.
        """
        if char not in self.label2id:
            self.label2id[char] = self.nb_label
            self.nb_label += 1

    def tokenize_csv_to_npy(self,
                            data_csv: str,
                            output_root: str,
                            id_cols:     list   = ['video_clip_name','timestamp'],
                            group_col:   str    = 'group_name',
                            label_col:   str    = 'class',
                            sep:         str    = None,
                            modality:    str    = 'tok_label'):
        """
        For each row in data_csv:
         - Build sample_id = "_".join(id_cols)
         - Map class → ids
         - Save to {output_root}/{group_name}/{modality}/{sample_id}.npy
        Args:
            data_csv (str): Path to the CSV file containing the data to tokenize.
            output_root (str): Root directory to save the .npy files.
            id_cols (list): List of columns to build the sample_id.
            group_col (str): Column name for grouping.
            label_col (str): Column name for the label.
            sep (str): Separator for splitting the label (if multiple classes per sample).
            modality (str): Name of the modality (used as subfolder name).
        """
        df        = pd.read_csv(data_csv)
        out_root  = Path(output_root)

        for _, row in df.iterrows():
            sample_id = "_".join(str(row[c]) for c in id_cols)
            raw = row[label_col]
            toks = raw.split(sep) if sep else [str(raw)]
            ids = [self.label2id[t] for t in toks]
            arr = np.array(ids, dtype=np.uint32)
            grp_dir = out_root / str(row[group_col]) / modality
            grp_dir.mkdir(parents=True, exist_ok=True)
            np.save(grp_dir / f"{sample_id}.npy", arr)

        print(f"✔ {len(df)} .npy files created under {output_root}/*/{modality}/")

if __name__ == "__main__":
    dict_csv    = "/home/godey/SAGA_COM-304/dataset_module/data/processed_data/label_counts.csv"
    data_csv    = "/work/com-304/SAGA/vggsound_valid.csv"   # with columns video_clip_name,timestamp,class,group_name
    output_root = "/work/com-304/SAGA/tokens_16_05"

    mapper = Maplabel(dict_csv)
    mapper.tokenize_csv_to_npy(
        data_csv     = data_csv,
        output_root  = output_root,
        id_cols      = ['video_clip_name'],
        group_col    = 'group_name',
        label_col    = 'class',
        sep          = None,        # or ' ' if multiple classes
        modality     = 'tok_label'  # desired folder name
    )
