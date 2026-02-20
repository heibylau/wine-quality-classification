# Wine Quality Classification

Classify the quality of wine into `Poor` or `Good` based on its physicochemical properties.  

## Model Performance
Based on accuracy
- SVM (rbf): 0.753  
- SVM (polynomial): 0.740
- Logistic Regression/LDA: 0.739
- SVM (linear): 0.736
- Classification Tree: 0.729

## Files
`wine-quality-classification.ipynb`: Contains the main flow of this data analysis project  
`data-preprocessing.ipynb`: For applying preprocessing steps to the raw data  
`model.py`: Model definitions and training utilities  
`visualization.py`: For generating plots and visual analysis  

All figures are contained in the `images/` folder.

## Data
The dataset used is called Wine Quality, found on the UCI Machine Learning Repository. The link to the dataset is [here](https://archive.ics.uci.edu/dataset/186/wine+quality).  

The raw and cleaned datasets are in `data/` folder.