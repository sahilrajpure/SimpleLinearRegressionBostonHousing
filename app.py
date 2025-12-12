import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="📊",
    layout="wide",
    font-size: 36px, 
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
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
st.sidebar.markdown("# 🧭 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📚 Theory & Concepts",
    "🎯 Problem Statement",
    "🔍 Dataset Exploration",
    "📈 Visualizations",
    "🤖 Model & Prediction",
    "📊 Results Analysis"
])

# Main title
st.markdown('<p class="main-header">📐 Simple Linear Regression Tutorial</p>', unsafe_allow_html=True)
st.markdown("---")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("## 👋 Welcome to Simple Linear Regression Learning Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 What You'll Learn
        - **Basic concepts** of Simple Linear Regression
        - **Mathematical foundation** behind the algorithm
        - **Practical implementation** using Python
        - **Real-world application** with Boston Housing dataset
        - **Model evaluation** and interpretation
        """)
        
        st.markdown("""
        ### 🔧 Model Status
        """)
        if model is not None and model_data is not None:
            st.success("✅ Pre-trained model loaded successfully!")
            st.info(f"📦 Dataset contains {len(model_data.get('X_train', []))} training samples")
        else:
            st.error("❌ Model or data files not found")
        
    with col2:
        st.markdown("""
        ### 🛠️ Technologies Used
        - **Streamlit** - Interactive web app
        - **Scikit-learn** - Machine learning
        - **Pandas** - Data manipulation
        - **Matplotlib & Seaborn** - Visualization
        - **NumPy** - Numerical computing
        """)
        
        st.markdown("""
        ### 📁 Required Files
        - `slr_model.pkl` - Trained regression model
        - `model_data.pkl` - Dataset and predictions
        """)
    
    st.info("💡 Use the sidebar to navigate through different sections!")

# THEORY & CONCEPTS
elif page == "📚 Theory & Concepts":
    st.markdown("## 📚 Understanding Simple Linear Regression")
    
    st.markdown('<div class="theory-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🤔 What is Simple Linear Regression?
    
    Simple Linear Regression is a **statistical method** that allows us to study relationships 
    between two continuous variables:
    - **Independent variable (X)**: The predictor or feature
    - **Dependent variable (Y)**: The target or outcome we want to predict
    
    The goal is to find the **best-fitting straight line** through the data points.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📐 The Linear Equation
        
        The relationship is modeled as:
        
        **Y = β₀ + β₁X + ε**
        
        Where:
        - **Y** = Dependent variable (what we predict)
        - **X** = Independent variable (what we use to predict)
        - **β₀** = Y-intercept (value of Y when X = 0)
        - **β₁** = Slope (change in Y for unit change in X)
        - **ε** = Error term (random variation)
        """)
        
    with col2:
        st.markdown("""
        ### ✅ Key Assumptions
        
        1. **Linearity**: Relationship between X and Y is linear
        2. **Independence**: Observations are independent
        3. **Homoscedasticity**: Constant variance of errors
        4. **Normality**: Errors follow normal distribution
        5. **No multicollinearity**: (for multiple regression)
        """)
    
    st.markdown('<div class="theory-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ⚙️ How Does It Work?
    
    The algorithm uses the **Ordinary Least Squares (OLS)** method:
    
    1. **Calculate the line** that minimizes the sum of squared residuals
    2. **Residual** = Actual value - Predicted value
    3. **Objective**: Minimize Σ(yᵢ - ŷᵢ)²
    
    The formulas for coefficients are:
    - **β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ(xᵢ - x̄)²**
    - **β₀ = ȳ - β₁x̄**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Evaluation Metrics
    
    - **R² Score (R-squared)**: Proportion of variance explained by the model (0 to 1, higher is better)
    - **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values
    - **RMSE (Root Mean Squared Error)**: Square root of MSE, same units as target variable
    - **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values
    """)

# PROBLEM STATEMENT
elif page == "🎯 Problem Statement":
    st.markdown("## 🎯 Problem Statement")
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    ### 🏘️ Boston Housing Price Prediction
    
    **Objective**: Predict the median value of owner-occupied homes (MEDV) in Boston suburbs 
    based on various features of the houses and neighborhoods.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Our Specific Goal
    
    In this tutorial, we'll use **Simple Linear Regression** to predict house prices based on 
    a **single feature**. We'll explore which feature has the strongest linear relationship 
    with the house prices.
    
    ### 💼 Business Context
    
    Real estate agencies, banks, and investors need to:
    - **Estimate property values** for buying/selling decisions
    - **Assess loan amounts** for mortgages
    - **Identify key factors** that influence property prices
    - **Make data-driven decisions** in the housing market
    
    ### 🎯 Expected Outcomes
    
    1. Understand which feature is most predictive of house prices
    2. Build a simple predictive model
    3. Evaluate model performance
    4. Make predictions for new properties
    """)
    
    if model is not None and model_data is not None:
        st.success("✅ Model is ready to use for predictions!")

# DATASET EXPLORATION
elif page == "🔍 Dataset Exploration":
    st.markdown("## 🔍 Dataset Exploration")
    
    if model_data is not None:
        # Extract data from model_data
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        # Display variable information
        st.markdown("### 📋 Variables in the Model")
        st.markdown(f"""
        **Independent Variable (X):** RM
        - Average number of rooms per dwelling
        - This feature is used to predict house prices
        
        **Dependent Variable (y):** MEDV
        - Median value of owner-occupied homes
        - Measured in thousands of dollars
        - This is the target variable we are trying to predict
        """)
        
        st.markdown("---")
        
        # Create full dataset for display
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
        
        st.markdown("""
        ### 🏠 Boston Housing Dataset
        
        The Boston Housing dataset contains information collected by the U.S Census Service 
        concerning housing in the area of Boston, Massachusetts.
        """)
        
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
        st.markdown("### 👀 View Dataset")
        view_option = st.radio("Select view:", ["First 10 rows", "Last 10 rows", "Random sample", "Full dataset", "Statistics"])
        
        if view_option == "First 10 rows":
            st.dataframe(full_df.head(10), use_container_width=True)
        elif view_option == "Last 10 rows":
            st.dataframe(full_df.tail(10), use_container_width=True)
        elif view_option == "Random sample":
            st.dataframe(full_df.sample(10), use_container_width=True)
        elif view_option == "Full dataset":
            st.dataframe(full_df, use_container_width=True)
        elif view_option == "Statistics":
            st.markdown("#### 📊 Statistical Summary")
            st.dataframe(full_df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Feature information
        st.markdown(f"""
        ### 📌 Feature Information
        
        **Selected Feature**: **RM** (Average number of rooms per dwelling)
        
        **Target Variable**: **MEDV** (Median value of owner-occupied homes in $1000s)
        """)
        
        # Display feature statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 📊 RM Statistics")
            feature_data = full_df[feature_name]
            st.write(f"- **Mean**: {feature_data.mean():.2f}")
            st.write(f"- **Median**: {feature_data.median():.2f}")
            st.write(f"- **Std Dev**: {feature_data.std():.2f}")
            st.write(f"- **Min**: {feature_data.min():.2f}")
            st.write(f"- **Max**: {feature_data.max():.2f}")
            
        with col2:
            st.markdown("#### 💰 MEDV (Target) Statistics")
            st.write(f"- **Mean**: ${full_df['MEDV'].mean():.2f}k")
            st.write(f"- **Median**: ${full_df['MEDV'].median():.2f}k")
            st.write(f"- **Std Dev**: ${full_df['MEDV'].std():.2f}k")
            st.write(f"- **Min**: ${full_df['MEDV'].min():.2f}k")
            st.write(f"- **Max**: ${full_df['MEDV'].max():.2f}k")
        
        # Common Boston Housing features explanation
        st.markdown("""
        ### 📝 Common Features in Boston Housing Dataset
        
        - **CRIM**: Per capita crime rate by town
        - **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
        - **INDUS**: Proportion of non-retail business acres per town
        - **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        - **NOX**: Nitric oxides concentration (parts per 10 million)
        - **RM**: Average number of rooms per dwelling
        - **AGE**: Proportion of owner-occupied units built prior to 1940
        - **DIS**: Weighted distances to five Boston employment centres
        - **RAD**: Index of accessibility to radial highways
        - **TAX**: Full-value property-tax rate per $10,000
        - **PTRATIO**: Pupil-teacher ratio by town
        - **B**: 1000(Bk - 0.63)² where Bk is the proportion of Black residents by town
        - **LSTAT**: Percentage of lower status of the population
        """)
    else:
        st.error("❌ Data not loaded. Please ensure 'model_data.pkl' is available.")

# VISUALIZATIONS
elif page == "📈 Visualizations":
    st.markdown("## 📈 Data Visualizations")
    
    if model_data is not None:
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        # Flatten arrays if needed
        X_train_flat = X_train.flatten() if len(X_train.shape) > 1 else X_train
        X_test_flat = X_test.flatten() if len(X_test.shape) > 1 else X_test
        
        st.markdown("""
        📊 These visualizations help us understand the relationship between the feature and target variable.
        """)
        
        # Visualization 1: Scatter Plot with Regression Line
        st.markdown("### 📍 Visualization 1: Scatter Plot with Regression Line")
        
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot training data
        ax1.scatter(X_train_flat, y_train, alpha=0.6, color='blue', label='Training Data', s=50)
        
        # Plot test data
        ax1.scatter(X_test_flat, y_test, alpha=0.6, color='green', label='Test Data', s=50)
        
        # Plot regression line if model exists
        if model is not None:
            X_range = np.linspace(X_train_flat.min(), X_train_flat.max(), 100).reshape(-1, 1)
            y_pred_line = model.predict(X_range)
            ax1.plot(X_range, y_pred_line, color='red', linewidth=2, label='Regression Line')
        
        ax1.set_xlabel('RM (Average number of rooms per dwelling)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MEDV (House Price in $1000s)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Simple Linear Regression: RM vs House Price', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
        
        st.markdown("""
        **💡 Interpretation**:
        - **Blue points** represent training data used to fit the model
        - **Green points** represent test data used to evaluate the model
        - **Red line** is the best-fit regression line that minimizes the squared errors
        - The line shows the predicted relationship between the feature and house prices
        - Points closer to the line indicate better predictions
        """)
        
        st.markdown("---")
        
        # Visualization 2: Distribution Plots
        st.markdown("### 📊 Visualization 2: Distribution Analysis")
        
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Feature distribution
        axes[0, 0].hist(X_train_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel(feature_name, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].set_title(f'Distribution of RM', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Target distribution
        axes[0, 1].hist(y_train, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('MEDV (House Price)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title('Distribution of House Prices', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual plot (if model exists)
        if model is not None:
            y_train_pred = model.predict(X_train)
            residuals_train = y_train - y_train_pred
            
            axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, color='purple')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted Values', fontweight='bold')
            axes[1, 0].set_ylabel('Residuals', fontweight='bold')
            axes[1, 0].set_title('Residual Plot (Training Data)', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for residuals
        if model is not None:
            from scipy import stats
            stats.probplot(residuals_train, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot of Residuals', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("""
        **💡 Interpretation**:
        
        1. **Feature Distribution** (Top Left):
           - Shows how the independent variable is distributed across the dataset
           - Helps identify if the data is skewed or normally distributed
        
        2. **House Price Distribution** (Top Right):
           - Shows the distribution of target variable (house prices)
           - Most Boston houses are in the $20k-$25k range
        
        3. **Residual Plot** (Bottom Left):
           - Shows the difference between actual and predicted values
           - Points should be randomly scattered around the horizontal line at 0
           - Patterns in residuals indicate model limitations
        
        4. **Q-Q Plot** (Bottom Right):
           - Tests if residuals follow a normal distribution
           - Points should fall along the diagonal line for normality assumption
           - Deviations suggest violations of normality assumption
        """)
        
    else:
        st.error("❌ Data not loaded. Please ensure 'model_data.pkl' is available.")

# MODEL & PREDICTION
elif page == "🤖 Model & Prediction":
    st.markdown("## 🤖 Model Training & Prediction")
    
    if model is not None and model_data is not None:
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ℹ️ Model Information
        
        Your pre-trained Simple Linear Regression model has been loaded successfully!
        
        **Model Type**: Linear Regression (Ordinary Least Squares)
        
        **Training Method**: The model was trained using the OLS method to minimize 
        the sum of squared residuals.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display model parameters
        st.markdown("### ⚙️ Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"**Intercept (β₀)**")
            st.markdown(f"### ${model.intercept_:.4f}k")
            st.markdown("*Base price when feature = 0*")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f"**Coefficient (β₁)**")
            st.markdown(f"### {model.coef_[0]:.4f}")
            st.markdown(f"*Price change per unit of RM(Average number of rooms per dwelling)*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        ### 📐 Model Equation
        
        **House Price = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {feature_name}**
        
        This equation means:
        - For every unit increase in RM, the house price changes by ${abs(model.coef_[0]):.4f}k
        - The base price (when RM = 0) is ${model.intercept_:.4f}k
        """)
        
        st.markdown("---")
        
        # Make Predictions
        st.markdown("### 🎯 Make Your Own Prediction")
        
        st.markdown(f"Enter a value for **RM(Average number of rooms per dwelling)** to predict the house price:")
        
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
        ### 🎯 Prediction Result:
        
        **Input**: {feature_name} = {user_input:.2f}
        
        **Predicted House Price**: ${prediction:.2f}k (${prediction * 1000:.2f})
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualize the prediction
        fig, ax = plt.subplots(figsize=(10, 6))
        
        X_train_flat = X_train.flatten() if len(X_train.shape) > 1 else X_train
        
        ax.scatter(X_train_flat, y_train, alpha=0.5, color='blue', label='Training Data')
        
        # Plot regression line
        X_range = np.linspace(X_train_flat.min(), X_train_flat.max(), 100).reshape(-1, 1)
        y_pred_line = model.predict(X_range)
        ax.plot(X_range, y_pred_line, color='red', linewidth=2, label='Regression Line')
        
        # Plot user's prediction
        ax.scatter(user_input, prediction, color='gold', s=300, marker='*', 
                  edgecolors='black', linewidth=2, label='Your Prediction', zorder=5)
        
        ax.set_xlabel('RM (Average number of rooms per dwelling)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MEDV (House Price in $1000s)', fontsize=12, fontweight='bold')
        ax.set_title('Your Prediction on the Regression Line', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    else:
        st.error("❌ Model or data not loaded. Please ensure both .pkl files are available.")

# RESULTS ANALYSIS
elif page == "📊 Results Analysis":
    st.markdown("## 📊 Model Performance Analysis")
    
    if model is not None and model_data is not None:
        X_train = model_data.get('X_train')
        y_train = model_data.get('y_train')
        X_test = model_data.get('X_test')
        y_test = model_data.get('y_test')
        feature_name = model_data.get('feature_name', 'Feature')
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Display metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Training Set Performance")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("R² Score", f"{train_r2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("RMSE", f"${train_rmse:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("MAE", f"${train_mae:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### 🧪 Test Set Performance")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("R² Score", f"{test_r2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("RMSE", f"${test_rmse:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("MAE", f"${test_mae:.4f}k")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interpretation
        st.markdown("### 💡 Metrics Interpretation")
        
        st.markdown(f"""
        **R² Score (Coefficient of Determination)**:
        - Training: {train_r2:.4f} ({train_r2*100:.2f}% of variance explained)
        - Test: {test_r2:.4f} ({test_r2*100:.2f}% of variance explained)
        - Interpretation: The model explains approximately {test_r2*100:.1f}% of the variation in house prices using {feature_name}
        
        **RMSE (Root Mean Squared Error)**:
        - Test RMSE: ${test_rmse:.4f}k
        - Interpretation: On average, predictions are off by about ${test_rmse:.2f}k from actual prices
        
        **MAE (Mean Absolute Error)**:
        - Test MAE: ${test_mae:.4f}k
        - Interpretation: On average, predictions deviate by ${test_mae:.2f}k from actual values
        - MAE is less sensitive to outliers compared to RMSE
        """)
        
        st.markdown("---")
        
        # Model Quality Assessment
        st.markdown("### ⭐ Model Quality Assessment")
        
        if test_r2 > 0.7:
            quality = "Excellent"
            color = "green"
            icon = "✅"
        elif test_r2 > 0.5:
            quality = "Good"
            color = "blue"
            icon = "👍"
        elif test_r2 > 0.3:
            quality = "Moderate"
            color = "orange"
            icon = "⚠️"
        else:
            quality = "Poor"
            color = "red"
            icon = "❌"
        
        st.markdown(f"""
        {icon} **Overall Model Quality**: **{quality}**
        
        - The model shows **{quality.lower()}** predictive performance
        - {'The feature has a strong linear relationship with house prices' if test_r2 > 0.5 else 'The feature has a weak to moderate linear relationship with house prices'}
        - {'Minimal overfitting detected' if abs(train_r2 - test_r2) < 0.1 else 'Some overfitting may be present' if train_r2 > test_r2 else 'Underfitting may be present'}
        """)
        
        st.markdown("---")
        
        # Visualization: Actual vs Predicted
        st.markdown("### 📈 Actual vs Predicted Values")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training set
        axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Price ($1000s)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Price ($1000s)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual Price ($1000s)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted Price ($1000s)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **💡 How to Read This Plot**:
        - Each point represents one house (actual price vs predicted price)
        - The **red dashed line** represents perfect predictions
        - Points **closer to the line** indicate more accurate predictions
        - Points **above the line** = model underpredicted (predicted lower than actual)
        - Points **below the line** = model overpredicted (predicted higher than actual)
        """)
        
        st.markdown("---")
        
        # Error Analysis
        st.markdown("### 🔍 Error Analysis")
        
        residuals_test = y_test - y_test_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Residual Statistics")
            st.write(f"- **Mean Residual**: ${np.mean(residuals_test):.4f}k")
            st.write(f"- **Std Dev of Residuals**: ${np.std(residuals_test):.4f}k")
            st.write(f"- **Min Residual**: ${np.min(residuals_test):.4f}k")
            st.write(f"- **Max Residual**: ${np.max(residuals_test):.4f}k")
        
        with col2:
            st.markdown("#### 📈 Error Distribution")
            within_1std = np.sum(np.abs(residuals_test) <= np.std(residuals_test)) / len(residuals_test) * 100
            within_2std = np.sum(np.abs(residuals_test) <= 2*np.std(residuals_test)) / len(residuals_test) * 100
            st.write(f"- **Within 1 Std Dev**: {within_1std:.1f}%")
            st.write(f"- **Within 2 Std Dev**: {within_2std:.1f}%")
            st.write(f"- **Outliers (>2 Std Dev)**: {100-within_2std:.1f}%")
        
        st.markdown("---")
        
        # Predictions Comparison Table
        st.markdown("### 📋 Sample Predictions Comparison")
        
        # Create comparison dataframe
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
        
        # Key Findings and Conclusions
        st.markdown("### 🎯 Key Findings & Conclusions")
        
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown(f"""
        #### 📝 Summary of Results:
        
        1. **Model Performance**:
           - The model achieves an R² score of **{test_r2:.4f}** on the test set
           - This means the model explains **{test_r2*100:.1f}%** of the variance in house prices
           - The average prediction error (RMSE) is **${test_rmse:.2f}k**
        
        2. **Feature Importance**:
           - **{feature_name}** {'shows a strong' if abs(model.coef_[0]) > 5 else 'shows a moderate' if abs(model.coef_[0]) > 2 else 'shows a weak'} relationship with house prices
           - Each unit increase in {feature_name} is associated with a **${abs(model.coef_[0]):.4f}k** {'increase' if model.coef_[0] > 0 else 'decrease'} in house price
        
        3. **Model Reliability**:
           - {'Low overfitting - model generalizes well' if abs(train_r2 - test_r2) < 0.1 else 'Moderate overfitting detected' if train_r2 - test_r2 > 0.1 else 'Model is stable'}
           - Training R² = {train_r2:.4f}, Test R² = {test_r2:.4f}
        
        4. **Practical Implications**:
           - This model can be used for **{'preliminary' if test_r2 < 0.6 else 'reliable'}** price estimation
           - {feature_name} is **{'a significant' if test_r2 > 0.5 else 'one of many'}** predictor{'s' if test_r2 < 0.5 else ''} of house prices
           - For better accuracy, consider using **multiple features** (Multiple Linear Regression)
        
        5. **Limitations**:
           - Simple Linear Regression captures only **linear relationships**
           - Many other factors influence house prices (location, condition, market trends)
           - Real-world predictions should consider multiple features simultaneously
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations for Improvement")
        
        st.markdown("""
        1. **Feature Engineering**: 
           - Consider polynomial features to capture non-linear relationships
           - Create interaction terms between features
        
        2. **Multiple Features**:
           - Use Multiple Linear Regression with several predictive features
           - This typically improves R² score significantly
        
        3. **Advanced Models**:
           - Try Decision Trees, Random Forests, or Gradient Boosting
           - These can capture complex non-linear patterns
        
        4. **Data Quality**:
           - Handle outliers more carefully
           - Check for and address multicollinearity
           - Ensure data normalization if needed
        
        5. **Cross-Validation**:
           - Use k-fold cross-validation for more robust evaluation
           - Helps detect overfitting more reliably
        """)
        
        st.markdown("---")
        
        # Footer
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 🎉 Congratulations!
        
        You've successfully explored Simple Linear Regression from theory to practice!
        
        **What you learned**:
        - ✅ Mathematical foundations of linear regression
        - ✅ How to interpret model coefficients and metrics
        - ✅ Visualization techniques for model evaluation
        - ✅ Making predictions with a trained model
        - ✅ Understanding model limitations and improvements
        
        **Next Steps**:
        - 🚀 Explore Multiple Linear Regression with more features
        - 🚀 Try different regression algorithms
        - 🚀 Apply these concepts to your own datasets
        - 🚀 Learn about regularization techniques (Ridge, Lasso)
        
        Thank you for using this tutorial!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.error("❌ Model or data not loaded. Please ensure both .pkl files are available.")
