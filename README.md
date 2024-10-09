# Bayesian-AirQuality-Forecasting

<img width="604" alt="image" src="https://github.com/user-attachments/assets/62a0c572-c88b-4338-958d-6c9201ef6983">

This project uses **Bayesian Inference** and **Gibbs Sampling** to forecast PM2.5 levels, a key indicator of air quality, in India. It focuses on capturing the complex temporal dynamics and uncertainties associated with PM2.5 levels using a Bayesian ARIMA model, leveraging data preprocessing, model specification, and evaluation techniques.

## Project Motivation

Air pollution, particularly from fine particulate matter like **PM2.5**, poses a serious health risk globally. In 2019, the **WHO** estimated that 37% of air pollution-related deaths were caused by heart disease and stroke, with significant percentages attributed to lung infections and cancer. This project aims to bridge the gap between environmental management and data science by using advanced Bayesian methods to forecast PM2.5 levels and provide insights for public health planning.

## Project Context
This project was developed as part of the **Bayesian Statistics** subject in the **Master's Program in Machine Learning at Georgia Tech**. It serves as a demonstration of the application of Bayesian Inference and Gibbs Sampling in time-series forecasting, specifically targeting PM2.5 levels in India's air quality data.


## Problem Statement

PM2.5 levels in cities like New Delhi, India, can often reach hazardous levels (as high as 600), posing significant health risks. Traditional statistical methods struggle to accurately capture the temporal dependencies and uncertainties in air quality data, which are critical for effective forecasting. This project addresses these challenges using **Bayesian Inference** with **Gibbs Sampling**, which provides a flexible, probabilistic framework for modeling these dependencies.

## Approach

### 1. **Data Preprocessing**
   - **Dataset**: The project uses air-quality data from Kaggle, which includes 37,000 records of PM2.5 measurements in India.
   - **Steps**:
     - Handled missing values.
     - Converted timestamps to appropriate formats.
     - Scaled PM2.5 values using `StandardScaler` to ensure better convergence during model training.
     - Visualized PM2.5 levels over time to observe trends.

### 2. **Bayesian Model Specification**
   - A **Bayesian ARIMA (1,1,1)** model is used to capture temporal dependencies in PM2.5 levels.
   - **Model Components**:
     - **ϕ (Auto-Regressive Coefficient)**: Represents dependency on past values.
     - **θ (Moving Average Coefficient)**: Accounts for past forecast errors.
     - **ε (Error Term)**: Represents noise or randomness in the data.
   - **Likelihood Function**: Formulated to incorporate autoregressive and moving average components using a normal distribution.

### 3. **Prior Specification**
   - Priors are set based on domain knowledge and the nature of PM2.5 data. Both informative and uninformative priors were explored to guide the Bayesian learning process.

### 4. **Gibbs Sampling Implementation**
   - **Gibbs Sampling**, a Markov Chain Monte Carlo (MCMC) method, is used to draw samples from the posterior distribution of model parameters.
   - **Steps**:
     1. Initialized parameters with an initial guess.
     2. Iterated over the parameters, sampling from conditional distributions to update parameter values.
     3. Repeated until convergence, discarding a burn-in period for more accurate predictions.

### 5. **Model Evaluation**
   - **Performance Metrics**:
     - **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and forecasted PM2.5 levels.
     - **Root Mean Squared Error (RMSE)**: Accounts for squared differences between actual and forecasted values, providing a more comprehensive accuracy measure.
   - **Visualizations**:
     - Plots comparing predicted PM2.5 levels to actual levels, helping to assess the model’s performance qualitatively.
   - **Implementation**: Code was developed using Python, leveraging libraries like `NumPy`, `pandas`, `matplotlib`, and `scikit-learn`.

## Results

- The Bayesian ARIMA model demonstrated a strong ability to capture temporal patterns in PM2.5 levels. The evaluation metrics and visualizations showed that the model could predict air quality trends effectively.
- A separate dataset of PM2.5 levels from 2023 was used to validate the model, and the predictions aligned closely with the actual values.

## Project Structure

- **DataProcessor.py**: Preprocessing and scaling of PM2.5 data.
- **BayesianModelTrainer.py**: Model specification and training using Gibbs Sampling.
- **GibbsSampler.py**: Implementation of Gibbs Sampling for posterior estimation.
- **TestRunner.ipynb**: Jupyter notebook for running tests and visualizations.
- **ProjectReport_AirQualityIndex.pdf**: Full project report detailing the methods and results.
- **air-quality-india.csv**: Dataset used for training and testing.

## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/megha102/Bayesian-AirQuality-Forecasting.git
   ```

2. **Install required dependencies by creating conda env**:
   ```bash
   conda env create -f env.yml
   conda activate bayesian_stats_01
   ```

3. **Run the preprocessing and model training scripts**:
   ```bash
   python DataProcessor.py
   python BayesianModelTrainer.py
   ```

4. **Evaluate the model and visualize results**:
   Open `TestRunner.ipynb` in Jupyter Notebook to run the tests and visualizations.


## Dataset

- **Source**: [Air Quality Data in India on Kaggle](https://www.kaggle.com/code/amankumar234/air-quality-analysis-in-india/input)



