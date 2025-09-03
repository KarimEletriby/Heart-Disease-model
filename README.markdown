# Heart Disease Prediction

## Project Overview
This project uses Machine Learning to predict the presence of heart disease based on a dataset containing patient attributes. It explores the data, performs preprocessing, trains multiple classification models, evaluates their performance using accuracy, and visualizes the results in a bar chart for comparison.

## Dataset
The dataset is the **Heart Failure Prediction Dataset** (heart.csv), available from sources like [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It includes features such as:
- `Age`: Age of the patient
- `Sex`: Gender (M/F)
- `ChestPainType`: Type of chest pain (ATA, NAP, ASY, TA)
- `RestingBP`: Resting blood pressure
- `Cholesterol`: Serum cholesterol level
- `FastingBS`: Fasting blood sugar (1 = >120 mg/dl, 0 = otherwise)
- `RestingECG`: Resting electrocardiogram results (Normal, ST, LVH)
- `MaxHR`: Maximum heart rate achieved
- `ExerciseAngina`: Exercise-induced angina (Y/N)
- `Oldpeak`: ST depression induced by exercise
- `ST_Slope`: Slope of the peak exercise ST segment (Up, Flat, Down)
- Target: `HeartDisease` (1 = heart disease, 0 = no heart disease)

Note: The dataset file (`heart.csv`) is not included in the repository. Download it and place it in the project directory.

## Project Structure
- `Project 1 (Heart disease).ipynb`: The main Jupyter notebook with data exploration, preprocessing, model training, evaluation, and visualization.
- `README.md`: This file.
- (Optional) `requirements.txt`: List of dependencies.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

## Data Preprocessing
- Check for missing values and duplicates (none found in this dataset).
- Explore data: Summary statistics, unique values, histograms for numerical features, and correlation heatmap.
- Encode categorical variables using `LabelEncoder` for features like `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`.
- Scale numerical features using `StandardScaler`.
- Split data into features (X) and target (y), then into train/test sets (80/20 split).

## Models and Evaluation
Multiple classification models were trained and evaluated using accuracy on the test set:

| Model                  | Accuracy |
|------------------------|----------|
| K-Nearest Neighbors   | 0.86    |
| Decision Tree         | 0.80    |
| Naive Bayes           | 0.88    |
| Support Vector Machine| 0.72    |
| Random Forest         | 0.87    |
| Gradient Boosting     | 0.86    |

The best model is **Naive Bayes** with 88% accuracy.

## Usage
1. Download `heart.csv` and place it in the project root (or update the path in the notebook).
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Project 1 (Heart disease).ipynb"
   ```
3. The notebook will:
   - Load and explore the data.
   - Preprocess the data.
   - Train and evaluate the models.
   - Display a bar chart comparing model accuracies.

## Results
- Best model: Naive Bayes (Accuracy: 0.88).
- A bar chart is generated to visualize the accuracy comparison across models.

## Future Improvements
- Feature engineering (e.g., create new features from existing ones like BMI if additional data is available).
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Handle class imbalance if present (e.g., using SMOTE).
- Try advanced models like XGBoost or Neural Networks.
- Cross-validation for more reliable performance metrics.

## Contributing
Contributions are welcome! Fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions, open an issue on GitHub.