# Uber Data Analysis Project

This project analyzes Uber ride data using Python, pandas, seaborn, and matplotlib. The workflow includes data cleaning, feature engineering, distance calculation, descriptive statistics, and visualizations.

## Files
- `uber.csv`: Raw Uber dataset.
- `uber_cleaned.csv`: Cleaned dataset (rows with missing values removed).
- `uber_enhanced.csv`: Enhanced dataset with additional features (hour, day, month, weekday, distance).
- `uber_analysis.py`: Main analysis script.
- `Screenshot/`: Contains images and Power BI files related to the analysis.

## Steps Performed
1. **Data Loading**: Load the raw Uber dataset.
2. **Data Cleaning**: Remove rows with missing values.
3. **Feature Engineering**:
   - Convert pickup datetime to datetime object.
   - Extract hour, day, month, and weekday.
4. **Distance Calculation**: Compute trip distance using the Haversine formula.
5. **Descriptive Statistics**: Display summary statistics of the cleaned data.
6. **Visualizations**:
   - Fare amount distribution (histogram)
   - Fare amount outlier detection (boxplot)
   - Fare vs. distance (scatterplot)
   - Average fare by hour of day (line plot)
7. **Save Enhanced Dataset**: Save the final enhanced dataset as `uber_enhanced.csv`.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Usage
Run the analysis script:
```bash
python uber_analysis.py
```

## Outputs
- Cleaned and enhanced CSV files
- Plots visualizing fare distributions and relationships
- Screenshots and Power BI files in the `Screenshot/` folder

---
**Author:** [Akize israel]
**Date:** July 2025
