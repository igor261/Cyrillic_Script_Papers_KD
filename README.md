# Cyrillic_Script_Papers_KD
Authors: 
- Johan Krause
- Igor Shapiro
## Task
- Build an information extraction (IE) system for the retrieval of title and authors from Cyrillic script academic works
- Establish this system using an approach based on machine learning

## Data
- Scholarly works from the CORE open access database were used (https://core.ac.uk/) 
- The data used for model training can be downloaded here: https://bwsyncandshare.kit.edu/s/e4X6PQtWqwow82b?path=%2FPDFs%20for%20ML%20Model%2F_final_selection_15553
- Meta data file: `final_items_15553.csv`
- Final selection of PDF papers: `PDFs_15553.zip/.rar`
- The script used for PDF download incorporates the CORE API (https://core.ac.uk/services/api/) 
- Download procedure was carried out as in .ipynb file `PDF_downloads_API_from_CSVs_.ipynb` 

## Training of the Model (including Data Pre-processing)
- Python file `KDDM_LSTM_VM_100dim_Recall_5_Folds_25_Epochs_clean.py`
- Includes the read in of all utilized PDFs with the PDFminer
- Includes the pre-processing steps for token processing, text vectorization and additional feature computation
- Includes the building of the Bi-LSTM sequential Keras model
- Includes the fitting (training) of the model with a 5-fold cross validation and 25 epochs each 
- Was deployed on a Google Compute Engine for increased computing power and RAM 

## Evaluation of our Bi-LSTM Model
- `BILSTM_Evaluation.ipynb`: Read the test data & evaluates the predicitons of our BILSTM model with respect to the chosen performance measures (Accuracy, Macro-F1-Score, Macro-Jaccard-Score)

## Evaluation of GROBID
- `GROBID_Prediction.ipynb`: extracts the header meta data of the papers with the GROBID service and saves them in XML-files.
- `GROBID_Evaluation.ipynb`: Read the XML-files & converts them into a data frame and saves it in a CSV-file (`grobid_16467.csv`). The code also evaluates the predicitons of GROBID with respect to the chosen performance measures (Accuracy, Macro-F1-Score, Macro-Jaccard-Score)

