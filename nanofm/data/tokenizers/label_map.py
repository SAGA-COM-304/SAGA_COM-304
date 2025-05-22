import os
import pandas as pd
import numpy as np
from pathlib import Path

class Maplabel:
    def __init__(self, path_csv):
        """
        path_csv: CSV (sans header) à deux colonnes [label, count]
        """
        self.path_csv = path_csv
        self.label2id  = self._build_map()
        self.nb_label  = len(self.label2id)

    def _build_map(self):
        df = pd.read_csv(self.path_csv, header=None, names=['label','count'])
        df = df.sort_values(['count','label'], ascending=[False,True])
        return {lab: idx for idx, lab in enumerate(df['label'].tolist())}

    def add_special_char(self, char:str):
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
        Pour chaque ligne de data_csv:
         - construit sample_id = "_".join(id_cols)
         - mappe class → [SOF] + ids + [EOF]
         - sauve dans {output_root}/{group_name}/{modality}/{sample_id}.npy
        """
        # 1) ajoute les tokens spéciaux
        # for s in ('SOF','EOF','PAD'):
            # self.add_special_char(s)

        # 2) prépare dataframe et dossier racine
        df        = pd.read_csv(data_csv)
        out_root  = Path(output_root)

        # 3) itère échantillons
        for _, row in df.iterrows():
            # a) ID unique
            sample_id = "_".join(str(row[c]) for c in id_cols)

            # b) extract tokens
            raw = row[label_col]
            toks = raw.split(sep) if sep else [str(raw)]

            # c) map to IDs
            # ids = [self.label2id['SOF']] + [self.label2id[t] for t in toks] + [self.label2id['EOF']]
            ids = [self.label2id[t] for t in toks]
            arr = np.array(ids, dtype=np.uint32)

            # d) chemin de sortie
            grp_dir = out_root / str(row[group_col]) / modality
            grp_dir.mkdir(parents=True, exist_ok=True)

            # e) save
            np.save(grp_dir / f"{sample_id}.npy", arr)
            # print(grp_dir / f"{sample_id}.npy")
            # print(arr)

        print(f"✔ {len(df)} fichiers .npy créés sous {output_root}/*/{modality}/")

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
        sep          = None,        # ou ' ' si plusieurs classes
        modality     = 'tok_label'  # nom du dossier souhaité
    )
