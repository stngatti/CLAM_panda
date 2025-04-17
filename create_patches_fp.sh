#!/bin/bash

# Imposta le variabili per i percorsi base
DATA_SLIDE_DIR="/mnt/e/prostate-cancer-grade-assessment/train_images"
SAVE_DIR="/mnt/e/prostate-cancer-grade-assessment/processed_images3"
MASK_DIR="/mnt/e/prostate-cancer-grade-assessment/train_label_masks"

# Imposta le variabili per i parametri di patching
PATCH_SIZE=256
STEP_SIZE=256
PATCH_LEVEL=0
CUSTOM_DOWNSAMPLE=1

# Commentato per mantenere i file esistenti nella cartella "patches"
# rm -f "${SAVE_DIR}/patches/"*.h5

# Esegui lo script Python create_patches_fp.py con i parametri specificati
python \
  --source "$DATA_SLIDE_DIR" \
  --save_dir "$SAVE_DIR" \
  --patch_size "$PATCH_SIZE" \
  --step_size "$STEP_SIZE" \
  --patch_level "$PATCH_LEVEL" \
  --seg \
  --patch \
  --mask_source "$MASK_DIR" \

echo "Patching FP completato!"
