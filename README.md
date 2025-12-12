# Simple Linear Regression Tutorial

An interactive web application built with Streamlit to learn and understand Simple Linear Regression using the Boston Housing dataset.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This application provides a comprehensive learning platform for understanding Simple Linear Regression, from mathematical foundations to practical implementation. Users can explore the Boston Housing dataset, visualize relationships, and make real-time predictions through an intuitive interface.

## Features

- **7 Interactive Sections**: Home, Theory & Concepts, Problem Statement, Dataset Exploration, Visualizations, Model & Prediction, and Results Analysis
- **Real-time Predictions**: Interactive slider to input values and see instant predictions
- **Comprehensive Visualizations**: Scatter plots, distribution plots, residual plots, and Q-Q plots
- **Detailed Metrics**: R-squared Score, RMSE, MAE with clear interpretations
- **Educational Content**: Step-by-step explanations of linear regression concepts

## Technologies Used

- **Streamlit** - Interactive web framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **SciPy** - Scientific computing

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simple-linear-regression-app.git
cd simple-linear-regression-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure model files are present:
   - `slr_model.pkl` - Trained regression model
   - `model_data.pkl` - Dataset and predictions

## Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
simple-linear-regression-app/
|
├── app.py                 # Main application file
├── slr_model.pkl         # Trained model
├── model_data.pkl        # Dataset
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## Model Information

- **Algorithm**: Simple Linear Regression (Ordinary Least Squares)
- **Dataset**: Boston Housing Dataset
- **Feature**: RM (Average number of rooms per dwelling)
- **Target**: MEDV (Median house value in $1000s)

## Model Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| R-squared Score | 0.5051 | 0.3708 |
| RMSE | $6.55k | $6.79k |
| MAE | $4.43k | $4.47k |

The model explains approximately 48% of the variance in house prices, with an average prediction error of $6,560.

## Screenshots

### Home Page
[![home.png](https://i.postimg.cc/fWc6D9yg/home.png)](https://postimg.cc/z3fdk3nn)

### Interactive Predictions
[![interactivepred.png](https://i.postimg.cc/g2shWNF4/interactivepred.png)](https://postimg.cc/XrZqKKvy)

### Visualizations
[![visualization.png](https://i.postimg.cc/vH97prZK/visualization.png)](https://postimg.cc/crx838Y7)

## Learning Objectives

After using this application, users will understand:
- Mathematical foundations of linear regression
- How to interpret coefficients and model parameters
- Evaluation metrics and their significance
- Residual analysis and model assumptions
- Making predictions with trained models

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Future Enhancements

- Multiple Linear Regression comparison
- Feature importance visualization
- Additional datasets (California Housing, etc.)
- Model comparison with other algorithms
- Downloadable reports
- Cross-validation implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Sahil Rajpure**
- GitHub: [@sahilrajpure](https://github.com/sahilrajpure)
- LinkedIn: [Sahil Rajpure](https://www.linkedin.com/in/sahil-rajpure-b2891924b/)
- Email: rajpuresahilcs222335@gmail.com

## Acknowledgments

- Boston Housing Dataset from UCI Machine Learning Repository
- Streamlit community for excellent documentation
- Scikit-learn for machine learning tools

## Contact

For questions or feedback:
- Email: rajpuresahilcs222335@gmail.com
- LinkedIn: [Sahil Rajpure](https://www.linkedin.com/in/sahil-rajpure-b2891924b/)

---

If you find this project helpful, please give it a star!
## Screenshots

![App Screenshot](C:\Users\Sahil\Downloads\home.png)


## home
