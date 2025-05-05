#!/bin/bash

# --- Configurazione ---

# Script Paths (assumendo che siano nella stessa directory di questo script .sh)
SPLIT_SCRIPT="create_splits_seq.py"
TRAIN_SCRIPT="main.py"

# Data & Output Paths
WSI_CSV_PATH="dataset_csv/train.csv" 
FEATURE_DIR="./output/"             
RESULTS_BASE_DIR="./results"

TASK_NAME="panda_isup"              # Nome del task (deve corrispondere a quello usato in main.py)
LABEL_COL="isup_grade"              # Nome della colonna label nel CSV
NUM_FOLDS=5                         # Numero di fold per la cross-validation
VAL_FRAC=0.15                       # Frazione per il set di validazione
TEST_FRAC=0.15                      # Frazione per il set di test
SPLIT_SEED=42                       # Seed per la riproducibilità della creazione degli split
SPLIT_SUFFIX=""

MODEL_TYPE="clam_sb"                # Tipo di modello CLAM ('clam_sb' o 'clam_mb')
EXP_CODE="${TASK_NAME}_k${NUM_FOLDS}_seed${SPLIT_SEED}${SPLIT_SUFFIX}" # Codice univoco per questo esperimento
MAX_EPOCHS=50                       # Numero massimo di epoche
LEARNING_RATE=2e-4                 
BAG_LOSS="ce"         
INST_LOSS="svm"    
BAG_WEIGHT=0.7                      # Peso per la bag loss (instance loss weight = 1 - bag_weight)
B_INSTANCES=8                       # Numero di istanze +/- da campionare per instance loss
TRAIN_SEED=${SPLIT_SEED}            # Seed per l'addestramento (spesso uguale a SPLIT_SEED per coerenza)

# --- Fase 1: Creazione degli Split ---

echo "--- [FASE 1/2] Avvio Creazione Split (${NUM_FOLDS}-fold) ---"

python "$SPLIT_SCRIPT" \
  --task "$TASK_NAME" \
  --csv_path "$WSI_CSV_PATH" \
  --label_col "$LABEL_COL" \
  --k "$NUM_FOLDS" \
  --val_frac "$VAL_FRAC" \
  --test_frac "$TEST_FRAC" \
  --seed "$SPLIT_SEED" \
  --split_dir_suffix "$SPLIT_SUFFIX"

# Controlla se la creazione degli split è fallita
if [ $? -ne 0 ]; then
  echo "Errore: La creazione degli split è fallita. Script interrotto."
  exit 1
fi

echo "--- [FASE 1/2] Creazione Split Completata ---"
echo "" 

# --- Fase 2: Addestramento CLAM ---
echo "--- [FASE 2/2] Avvio Addestramento CLAM (${EXP_CODE}) ---"

echo "tensorboard --logdir ${RESULTS_BASE_DIR}/${EXP_CODE}_s${TRAIN_SEED}/logs"

python "$TRAIN_SCRIPT" \
  --task "$TASK_NAME" \
  --data_root_dir "$FEATURE_DIR" \
  --results_dir "$RESULTS_BASE_DIR" \
  --model_type "$MODEL_TYPE" \
  --exp_code "$EXP_CODE" \
  --k "$NUM_FOLDS" \
  --max_epochs "$MAX_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --bag_loss "$BAG_LOSS" \
  --inst_loss "$INST_LOSS" \
  --bag_weight "$BAG_WEIGHT" \
  --B "$B_INSTANCES" \
  --seed "$TRAIN_SEED" \
  --log_data \
  --early_stopping \
  --weighted_samples

# Controlla se l'addestramento è fallito
if [ $? -ne 0 ]; then
  echo "Errore: L'addestramento è fallito. Script interrotto."
  exit 1
fi

echo "--- [FASE 2/2] Addestramento CLAM Completato ---"
echo ""
echo "Pipeline completata con successo! Risultati in: ${RESULTS_BASE_DIR}/${EXP_CODE}_s${TRAIN_SEED}"
echo "Per visualizzare i log di TensorBoard, esegui:"
echo "tensorboard --logdir ${RESULTS_BASE_DIR}/${EXP_CODE}_s${TRAIN_SEED}/logs"

exit 0