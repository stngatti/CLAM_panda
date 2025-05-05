import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification using slide_id')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'panda_isup'], default='panda_isup',
                    help='Specific task for splitting')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--csv_path', type=str, default='dataset_csv/train.csv',
                    help='Path to the dataset CSV file (default: dataset_csv/train.csv)')
parser.add_argument('--label_col', type=str, default='isup_grade',
                    help='Column name for labels (default: isup_grade)')
parser.add_argument('--split_dir_suffix', type=str, default='',
                    help='Optional suffix for the split directory name')

args = parser.parse_args()

print("[!] Modifica: Forzatura della stratificazione a livello di slide (patient_strat=False) poiché si usa slide_id come ID primario.")
patient_strat = False #each slide is a patient
patient_voting = None

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    label_dict = {'normal_tissue':0, 'tumor_tissue':1}
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2}
elif args.task == 'panda_isup':
    args.n_classes = 6
    label_dict = {i: i for i in range(args.n_classes)}
else:
    raise NotImplementedError(f"Task '{args.task}' not recognized.")

print(f"[*] Initializing Dataset for task: {args.task}")
dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                        shuffle = False,
                        seed = args.seed,
                        print_info = True,
                        label_dict = label_dict,
                        label_col = args.label_col,
                        patient_strat= patient_strat,
                        patient_voting= patient_voting,
                        ignore=[])

if patient_strat:
    num_samples_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    id_type = "patients"
else:
    num_samples_cls = np.array([len(cls_ids) for cls_ids in dataset.slide_cls_ids])
    id_type = "slides"

min_samples_needed = args.k if args.k > 1 else 2
if np.any(num_samples_cls < min_samples_needed):
     print(f"Warning: Some classes have fewer samples than the number of folds (or minimum required).")
     print(f"Number of {id_type} per class: {num_samples_cls}")

val_num = np.round(num_samples_cls * args.val_frac).astype(int)
test_num = np.round(num_samples_cls * args.test_frac).astype(int)

for i in range(args.n_classes):
    if num_samples_cls[i] > 1 and args.val_frac > 0:
        val_num[i] = max(1, val_num[i])
    else:
        val_num[i] = 0

    if num_samples_cls[i] > 1 and args.test_frac > 0:
        test_num[i] = max(1, test_num[i])
    else:
        test_num[i] = 0

    if val_num[i] + test_num[i] >= num_samples_cls[i]:
        print(f"Warning: Too few samples in class {i} ({num_samples_cls[i]} slides) to create non-empty train split with val_frac={args.val_frac}, test_frac={args.test_frac}.")
        needed_for_train = 1
        total_val_test = num_samples_cls[i] - needed_for_train
        if total_val_test < 0: total_val_test = 0

        if test_num[i] > 0:
             test_num[i] = min(test_num[i], total_val_test)
             val_num[i] = min(val_num[i], total_val_test - test_num[i])
        else:
             val_num[i] = min(val_num[i], total_val_test)

        val_num[i] = max(0, val_num[i])

        print(f"Adjusted for class {i}: val_num={val_num[i]}, test_num={test_num[i]}")

print(f"Calculated validation samples per class ({id_type}): {val_num}")
print(f"Calculated test samples per class ({id_type}): {test_num}")

if __name__ == '__main__':
    label_fracs = [args.label_frac]

    for lf in label_fracs:
        split_dir = f'splits/{args.task}_{int(lf * 100)}{args.split_dir_suffix}'
        os.makedirs(split_dir, exist_ok=True)

        print(f"[*] Creating {args.k} splits for label fraction: {lf}")
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)

        for i in range(args.k):
            print(f"\n--- Processing Fold {i} ---")
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)

            if splits[0] is None or len(splits[0]) == 0:
                 print(f"Warning: Skipping fold {i} due to empty training set.")
                 continue
            if splits[1] is None or (len(splits[1]) == 0 and args.val_frac > 0 and np.sum(val_num) > 0):
                 print(f"Warning: Fold {i} has an empty validation set (val_frac={args.val_frac}, calculated val_num={val_num}).")
            if splits[2] is None or (len(splits[2]) == 0 and args.test_frac > 0 and np.sum(test_num) > 0):
                 print(f"Warning: Fold {i} has an empty test set (test_frac={args.test_frac}, calculated test_num={test_num}).")

            split_base_name = os.path.join(split_dir, f'splits_{i}')
            save_splits(splits, ['train', 'val', 'test'], f'{split_base_name}.csv')
            descriptor_df.to_csv(f'{split_base_name}_descriptor.csv')
            print(f"[*] Saved splits (slide IDs) and descriptor for fold {i} in {split_dir}")

    print("\n[*] Split creation complete (using slide_id).")