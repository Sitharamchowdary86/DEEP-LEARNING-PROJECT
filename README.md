.
├── data/
│   ├── raw_data.csv         # Input CSV file (auto-generated if missing)
│   └── processed_data.csv   # Output CSV file after transformation
├── etl_pipeline.py          # Main ETL script
├── README.md                # Project documentation

Features

Extract: Reads raw CSV data.

Transform:

Imputes missing numerical values with mean.

Imputes missing categorical values with most frequent value.

Scales numerical columns using StandardScaler.

Encodes categorical columns using OneHotEncoder.

Load: Saves the transformed data into a new CSV file.

🚀 Getting Started
1. Clone the repository
git clone https://github.com/yourusername/etl-pipeline.git
cd etl-pipeline

2. Install dependencies

Make sure you have Python 3.7+ and install required libraries:

pip install pandas numpy scikit-learn

3. Run the ETL Pipeline
python etl_pipeline.py


If data/raw_data.csv does not exist, a sample dataset will be created automatically.

📊 Sample Data

Input (raw_data.csv)

age,salary,city,gender
25,50000,Delhi,M
30,54000,Mumbai,F
NaN,58000,Delhi,F
40,,Chennai,
35,62000,,M


Output (processed_data.csv): Transformed version with scaled and encoded features.

🛠️ Customization

You can modify the pipeline logic in etl_pipeline.py:

Change the imputation strategy (e.g., median, constant).

Use different encoders (e.g., OrdinalEncoder).

Add new preprocessing steps.

📌 License

This project is licensed under the MIT License.

🙌 Acknowledgements

Built with 💙 using:

Pandas

Scikit-Learn

visualization:
