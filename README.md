# Cyrillic_Script_Papers_KD
Authors: 
- Johan Krause
- Igor Shapiro

## Data
The data used for model training can be downloaded here: `https://bwsyncandshare.kit.edu/s/e4X6PQtWqwow82b?path=%2FPDFs%20for%20ML%20Model%2F_final_selection_15553`

- Meta data: `final_items_15553.csv`
- Final selection of PDF papers: `PDFs_15553.zip/.rar`

## Training of the Model (including Data Pre-processing)

## Evaluation of our BILSTM Model
- `BILSTM_Evaluation.ipynb`: Read the test data & evaluates the predicitons of our BILSTM model with respect to the chosen performance measures (Accuracy, Macro-F1-Score, Macro-Jaccard-Score)
## Evaluation of GROBID
- `GROBID_Prediction.ipynb`: extracts the header meta data of the papers with the GROBID service and saves them in XML-files.
- `GROBID_Evaluation.ipynb`: Read the XML-files & converts them into a data frame and saves it in a CSV-file (`grobid_16467.csv`). The code also evaluates the predicitons of GROBID with respect to the chosen performance measures (Accuracy, Macro-F1-Score, Macro-Jaccard-Score)

