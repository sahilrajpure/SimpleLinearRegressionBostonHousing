import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 60px !important;
        font-weight: 800 !important;
        color: #1f77b4 !important;
        text-align: center !important;
        line-height: 1.1 !important;
        margin-top: -10px !important;
    }

    @media (max-width: 768px) {
        .main-header {
            font-size: 38px !important;
            margin-top: 0px !important;
        }
    }

    .sub-header {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #2c3e50 !important;
    }

    @media (max-width: 768px) {
        .sub-header {
            font-size: 22px !important;
        }
    }

    .theory-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }

    .highlight {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }

    .metric-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    try:
        with open('slr_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'slr_model.pkl' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_data
def load_model_data():
    try:
        with open('model_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("Data file 'model_data.pkl' not found. Please ensure the file is in the same directory.")
        return None

# Load the model and data
model = load_model()
model_data = load_model_data()

# Sidebar
st.sidebar.markdown("# Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Theory",
    "Dataset Exploration",
    "Model & Results"
])

# Main title
st.markdown('<p class="main-header">Simple Linear Regression Tutorial</p>', unsafe_allow_html=True)
st.markdown("---")

if page == "Home":
    st.markdown("## Welcome to Simple Linear Regression Learning Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What You'll Learn
        - Basic concepts of Simple Linear Regression
        - Mathematical foundation behind the algorithm
        - Practical implementation using Python
        - Real-world application with Boston Housing dataset
        - Model evaluation and interpretation
        """)
        
        st.markdown("""
        ### Model Status
        """)
        if model is not None and model_data is not None:
            st.success("Pre-trained model loaded successfully!")
            st.info(f"Dataset contains {len(model_data.get('X_train', []))} training samples")
        else:
            st.error("Model or data files not found")
        
    with col2:
        st.markdown("""
        ### Technologies Used
        - **Streamlit** - Interactive web app
        - **Scikit-learn** - Machine learning
        - **Pandas** - Data manipulation
        - **Matplotlib & Seaborn** - Visualization
        - **NumPy** - Numerical computing
        """)
        
        st.markdown("""
        ### Required Files
        - `slr_model.pkl` - Trained regression model
        - `model_data.pkl` - Dataset and predictions
        """)
    
    st.info("NOTE: Use the sidebar to navigate through different sections!")

elif page == "Theory":
    st.markdown("## Theory & Concepts")
    
    st.markdown('<div class="theory-box">', unsafe_allow_html=True)
    st.markdown("""
    ### What is Simple Linear Regression?
    
    Simple Linear Regression is a statistical method that studies relationships between 
    two continuous variables. One variable (X) predicts another variable (Y). The goal 
    is to find the best-fitting straight line through the data points.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### The Linear Equation
        
        **Y = b0 + b1*X + e**
        
        Where:
        - Y = Dependent variable (what we predict)
        - X = Independent variable (predictor)
        - b0 = Intercept (Y value when X=0)
        - b1 = Slope (change in Y per unit X)
        - e = Error term
        """)
        
        st.markdown("""
        ### Key Assumptions
        
        1. Linear relationship between X and Y
        2. Observations are independent
        3. Constant variance of errors
        4. Errors follow normal distribution
        """)
        
    with col2:
        st.markdown("""
        ### How It Works
        
        Uses Ordinary Least Squares (OLS):
        - Minimizes sum of squared errors
        - Error = Actual value - Predicted value
        - Goal: Minimize total squared errors
        
        Formulas:
        - b1 = sum((Xi - Xmean)*(Yi - Ymean)) / sum((Xi - Xmean)^2)
        - b0 = Ymean - b1*Xmean
        """)
        
        st.markdown("""
        ### Evaluation Metrics
        
        - **R-squared (R2)**: 0 to 1, higher is better
        - **RMSE**: Root Mean Squared Error
        - **MAE**: Mean Absolute Error
        """)
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    ### Problem Statement: Boston Housing Price Prediction
    
    **Objective**: Predict median home values in Boston suburbs using the average number 
    of rooms per dwelling (RM feature).
    
    **Business Context**: Real estate agencies and banks need to estimate property values 
    for buying and selling decisions and mortgage assessments.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Dataset Exploration":
    st.markdown("## Dataset Exploration")
    
    if model_data is not None:
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        st.markdown("### Variables in the Model")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Independent Variable (X): RM**
            - Average number of rooms per dwelling
            - Used to predict house prices
            """)
        
        with col2:
            st.markdown("""
            **Dependent Variable (y): MEDV**
            - Median value of owner-occupied homes
            - Measured in thousands of dollars
            """)
        
        st.markdown("---")
        
        # Create full dataset
        train_df = pd.DataFrame({
            feature_name: X_train.flatten() if len(X_train.shape) > 1 else X_train,
            'MEDV': y_train
        })
        train_df['Dataset'] = 'Train'
        
        test_df = pd.DataFrame({
            feature_name: X_test.flatten() if len(X_test.shape) > 1 else X_test,
            'MEDV': y_test
        })
        test_df['Dataset'] = 'Test'
        
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        st.markdown("### Boston Housing Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Total Samples", len(full_df))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Training Samples", len(train_df))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Test Samples", len(test_df))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Features Used", 1)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset view options
        st.markdown("### View Dataset")
        view_option = st.radio("Select view:", ["First 10 rows", "Statistics", "Full dataset"])
        
        if view_option == "First 10 rows":
            st.dataframe(full_df.head(10), use_container_width=True)
        elif view_option == "Statistics":
            st.dataframe(full_df.describe(), use_container_width=True)
        elif view_option == "Full dataset":
            st.dataframe(full_df, use_container_width=True)
        
        st.markdown("---")
        
        # Display statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RM Statistics")
            feature_data = full_df[feature_name]
            st.write(f"- Mean: {feature_data.mean():.2f}")
            st.write(f"- Median: {feature_data.median():.2f}")
            st.write(f"- Std Dev: {feature_data.std():.2f}")
            st.write(f"- Min: {feature_data.min():.2f}")
            st.write(f"- Max: {feature_data.max():.2f}")
            
        with col2:
            st.markdown("#### MEDV (Target) Statistics")
            st.write(f"- Mean: ${full_df['MEDV'].mean():.2f}k")
            st.write(f"- Median: ${full_df['MEDV'].median():.2f}k")
            st.write(f"- Std Dev: ${full_df['MEDV'].std():.2f}k")
            st.write(f"- Min: ${full_df['MEDV'].min():.2f}k")
            st.write(f"- Max: ${full_df['MEDV'].max():.2f}k")
        
    else:
        st.error("Data not loaded. Please ensure 'model_data.pkl' is available.")

elif page == "Model & Results":
    st.markdown("## Model Training, Prediction & Results")
    
    if model is not None and model_data is not None:
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        # Flatten arrays
        X_train_flat = X_train.flatten() if len(X_train.shape) > 1 else X_train
        X_test_flat = X_test.flatten() if len(X_test.shape) > 1 else X_test
        
        # Calculate predictions and metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Model Information
        
        Pre-trained Simple Linear Regression model using Ordinary Least Squares (OLS) method.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model parameters
        st.markdown("### Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"**Intercept (b0)**")
            st.markdown(f"### ${model.intercept_:.4f}k")
            st.markdown("*Base price when RM = 0*")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"**Coefficient (b1)**")
            st.markdown(f"### {model.coef_[0]:.4f}")
            st.markdown("*Price change per unit of RM*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        ### Model Equation
        
        **House Price = {model.intercept_:.4f} + {model.coef_[0]:.4f} * RM**
        
        For every unit increase in RM, house price changes by ${abs(model.coef_[0]):.4f}k
        """)
        
        st.markdown("---")
        
        # Make Predictions Section
        st.markdown("### Make Your Own Prediction")
        
        st.markdown(f"Enter a value for RM (Average number of rooms per dwelling) to predict house price:")
        
        # Get min and max for slider
        X_all = np.concatenate([X_train.flatten(), X_test.flatten()])
        min_val = float(X_all.min())
        max_val = float(X_all.max())
        mean_val = float(X_all.mean())
        
        user_input = st.slider(
            f"Select {feature_name} value:",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100
        )
        
        # Make prediction
        user_input_reshaped = np.array([[user_input]])
        prediction = model.predict(user_input_reshaped)[0]
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"""
        ### Prediction Result:
        
        **Input**: {feature_name} = {user_input:.2f}
        
        **Predicted House Price**: ${prediction:.2f}k (${prediction * 1000:.2f})
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Regression Plot", "Distribution Analysis", "Actual vs Predicted"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.scatter(X_train_flat, y_train, alpha=0.5, color='blue', label='Training Data', s=50)
            ax1.scatter(X_test_flat, y_test, alpha=0.5, color='green', label='Test Data', s=50)
            
            X_range = np.linspace(X_train_flat.min(), X_train_flat.max(), 100).reshape(-1, 1)
            y_pred_line = model.predict(X_range)
            ax1.plot(X_range, y_pred_line, color='red', linewidth=2, label='Regression Line')
            
            # Plot user prediction
            ax1.scatter(user_input, prediction, color='gold', s=300, marker='*', 
                       edgecolors='black', linewidth=2, label='Your Prediction', zorder=5)
            
            ax1.set_xlabel('RM (Avg rooms per dwelling)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('MEDV (House Price in $1000s)', fontsize=11, fontweight='bold')
            ax1.set_title('Simple Linear Regression: RM vs House Price', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            st.pyplot(fig1)
            
            st.markdown("""
            **Interpretation**: Blue/green points show actual data, red line is the best-fit regression line.
            Gold star shows your prediction. Points closer to the line indicate better predictions.
            """)
        
        with tab2:
            fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
            
            # Feature distribution
            axes[0, 0].hist(X_train_flat, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('RM', fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontweight='bold')
            axes[0, 0].set_title('Distribution of RM', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Target distribution
            axes[0, 1].hist(y_train, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].set_xlabel('MEDV (House Price)', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontweight='bold')
            axes[0, 1].set_title('Distribution of House Prices', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residual plot
            residuals_train = y_train - y_train_pred
            axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, color='purple')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted Values', fontweight='bold')
            axes[1, 0].set_ylabel('Residuals', fontweight='bold')
            axes[1, 0].set_title('Residual Plot (Training Data)', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals_train, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot of Residuals', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            st.markdown("""
            **Interpretation**:
            
            1. **Feature Distribution** (Top Left): Shows distribution of independent variable
            2. **House Price Distribution** (Top Right): Shows distribution of target variable
            3. **Residual Plot** (Bottom Left): Points should be randomly scattered around zero line
            4. **Q-Q Plot** (Bottom Right): Tests normality of residuals - points should follow diagonal line
            """)
        
        with tab3:
            fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Training set
            axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
            axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                        'r--', linewidth=2, label='Perfect Prediction')
            axes[0].set_xlabel('Actual Price ($1000s)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Predicted Price ($1000s)', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Training Set: Actual vs Predicted (R2 = {train_r2:.4f})', fontsize=11, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Test set
            axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
            axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'r--', linewidth=2, label='Perfect Prediction')
            axes[1].set_xlabel('Actual Price ($1000s)', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Predicted Price ($1000s)', fontsize=11, fontweight='bold')
            axes[1].set_title(f'Test Set: Actual vs Predicted (R2 = {test_r2:.4f})', fontsize=11, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig3)
            
            st.markdown("""
            **How to Read**: Red dashed line represents perfect predictions. Points closer to this line 
            indicate better model performance. Points above the line mean underprediction, below means overprediction.
            """)
        
        st.markdown("---")
        
        # Performance Metrics
        st.markdown("### Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Set Performance")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("R2 Score", f"{train_r2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("RMSE", f"${train_rmse:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("MAE", f"${train_mae:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### Test Set Performance")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("R2 Score", f"{test_r2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("RMSE", f"${test_rmse:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("MAE", f"${test_mae:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Metrics Interpretation
        st.markdown("### Metrics Interpretation")
        
        st.markdown(f"""
        **R2 Score (R-squared)**:
        - Training: {train_r2:.4f} ({train_r2*100:.2f}% of variance explained)
        - Test: {test_r2:.4f} ({test_r2*100:.2f}% of variance explained)
        - The model explains approximately {test_r2*100:.1f}% of house price variation using RM
        
        **RMSE (Root Mean Squared Error)**:
        - Test RMSE: ${test_rmse:.4f}k
        - On average, predictions are off by about ${test_rmse:.2f}k from actual prices
        
        **MAE (Mean Absolute Error)**:
        - Test MAE: ${test_mae:.4f}k
        - Average prediction deviation is ${test_mae:.2f}k from actual values
        - MAE is less sensitive to outliers compared to RMSE
        """)
        
        st.markdown("---")
        
        # Model Quality Assessment
        st.markdown("### Model Quality Assessment")
        
        if test_r2 > 0.7:
            quality = "Excellent"
            quality_msg = "The model shows excellent predictive performance"
        elif test_r2 > 0.5:
            quality = "Good"
            quality_msg = "The model shows good predictive performance"
        elif test_r2 > 0.3:
            quality = "Moderate"
            quality_msg = "The model shows moderate predictive performance"
        else:
            quality = "Poor"
            quality_msg = "The model shows poor predictive performance"
        
        st.markdown(f"""
        **Overall Model Quality**: **{quality}**
        
        - {quality_msg}
        - {'The feature has a strong linear relationship with house prices' if test_r2 > 0.5 else 'The feature has a weak to moderate linear relationship with house prices'}
        - {'Minimal overfitting detected' if abs(train_r2 - test_r2) < 0.1 else 'Some overfitting may be present' if train_r2 > test_r2 else 'Underfitting may be present'}
        """)
        
        st.markdown("---")
        
        # Sample Predictions
        st.markdown("### Sample Predictions Comparison")
        
        residuals_test = y_test - y_test_pred
        comparison_df = pd.DataFrame({
            feature_name: X_test.flatten()[:10] if len(X_test.shape) > 1 else X_test[:10],
            'Actual Price ($1000s)': y_test[:10],
            'Predicted Price ($1000s)': y_test_pred[:10],
            'Error ($1000s)': residuals_test[:10],
            'Error (%)': (residuals_test[:10] / y_test[:10] * 100)
        })
        
        st.dataframe(comparison_df.style.format({
            feature_name: '{:.2f}',
            'Actual Price ($1000s)': '{:.2f}',
            'Predicted Price ($1000s)': '{:.2f}',
            'Error ($1000s)': '{:.2f}',
            'Error (%)': '{:.2f}%'
        }).background_gradient(subset=['Error (%)'], cmap='RdYlGn_r'), use_container_width=True)
        
        st.markdown("---")
        
        # Key Findings
        st.markdown("### Key Findings & Conclusions")
        
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown(f"""
        #### Summary of Results:
        
        1. **Model Performance**:
           - R2 score of {test_r2:.4f} on test set
           - Explains {test_r2*100:.1f}% of variance in house prices
           - Average prediction error (RMSE) is ${test_rmse:.2f}k
        
        2. **Feature Importance**:
           - RM shows {'a strong' if abs(model.coef_[0]) > 5 else 'a moderate' if abs(model.coef_[0]) > 2 else 'a weak'} relationship with prices
           - Each unit increase in RM changes price by ${abs(model.coef_[0]):.4f}k
        
        3. **Model Reliability**:
           - {'Low overfitting - model generalizes well' if abs(train_r2 - test_r2) < 0.1 else 'Moderate overfitting detected' if train_r2 - test_r2 > 0.1 else 'Model is stable'}
           - Training R2 = {train_r2:.4f}, Test R2 = {test_r2:.4f}
        
        4. **Practical Implications**:
           - Model suitable for {'preliminary' if test_r2 < 0.6 else 'reliable'} price estimation
           - RM is {'a significant' if test_r2 > 0.5 else 'one of many'} predictor of house prices
           - For better accuracy, consider using multiple features
        
        5. **Limitations**:
           - Simple Linear Regression captures only linear relationships
           - Many other factors influence house prices
           - Real-world predictions should consider multiple features
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### Recommendations for Improvement")
        
        st.markdown("""
        1. **Feature Engineering**: 
           - Consider polynomial features for non-linear relationships
           - Create interaction terms between features
        
        2. **Multiple Features**:
           - Use Multiple Linear Regression with several features
           - This typically improves R2 score significantly
        
        3. **Advanced Models**:
           - Try Decision Trees, Random Forests, or Gradient Boosting
           - These can capture complex non-linear patterns
        
        4. **Data Quality**:
           - Handle outliers more carefully
           - Check for and address multicollinearity
           - Ensure data normalization if needed
        
        5. **Cross-Validation**:
           - Use k-fold cross-validation for robust evaluation
           - Helps detect overfitting more reliably
        """)
        
        st.markdown("---")
