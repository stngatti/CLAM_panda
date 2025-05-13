#!/bin/bash

# --- Configurazione ---

# Percorso dello script Python
PYTHON_SCRIPT_PATH="/home/fausto/clam_panda/CLAM_panda/create_heatmaps_tia.py"

# Percorso del file di configurazione YAML
# Assicurati che questo percorso sia corretto e che il file esista.
CONFIG_FILE="heatmaps/configs/config.yaml" 

# (Opzionale) Specifica quali GPU utilizzare (es. "0" per la prima GPU, "0,1" per le prime due)
# Se non vuoi specificare, puoi commentare questa riga e lo script userà le GPU visibili di default.
CUDA_VISIBLE_DEVICES="1" 

# (Opzionale) Altri argomenti da passare allo script Python, se necessario.
# Esempio: OVERLAP_VALUE="0.25"
# Esempio: SAVE_EXP_CODE_VALUE="MyExperiment"

# --- Controllo Esistenza File ---
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Errore: Lo script Python '$PYTHON_SCRIPT_PATH' non è stato trovato."
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Errore: Il file di configurazione '$CONFIG_FILE' non è stato trovato."
    echo "Assicurati che il percorso sia corretto e relativo alla directory da cui esegui questo script .sh, oppure usa un percorso assoluto."
    exit 1
fi

# --- Costruzione del Comando ---
COMMAND="python $PYTHON_SCRIPT_PATH --config_file $CONFIG_FILE"

# Aggiungi CUDA_VISIBLE_DEVICES se definito
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $COMMAND"
fi

# Aggiungi altri argomenti opzionali se definiti
# Esempio:
# if [ ! -z "$OVERLAP_VALUE" ]; then
#    COMMAND="$COMMAND --overlap $OVERLAP_VALUE"
# fi
# if [ ! -z "$SAVE_EXP_CODE_VALUE" ]; then
#    COMMAND="$COMMAND --save_exp_code $SAVE_EXP_CODE_VALUE"
# fi

# --- Esecuzione ---
echo "Esecuzione del comando:"
echo "$COMMAND"
echo "---"

eval $COMMAND

echo "---"
echo "Script terminato."