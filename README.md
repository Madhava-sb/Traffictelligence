# Project Name : Traffictelligence

# Traffic Volume Estimation System

## Overview
The Traffic Volume Estimation System is a web-based application designed to predict hourly traffic volume based on user-inputted weather conditions, time, and holiday data. Built using Flask and a pre-trained RandomForestRegressor machine learning model, this project aims to assist urban commuters, traffic managers, and delivery drivers by providing accurate traffic predictions to reduce delays, improve safety, and optimize schedules. The system is deployed locally and offers an intuitive interface without requiring user registration.

## Features
- Predict traffic volume using weather (temperature, rain, snow), time, and holiday inputs.
- Display results with error handling for invalid inputs.
- Navigate back to the input page from results or error pages.
- Incorporate background images for enhanced user experience.
- Scalable design with potential for future real-time data integration.

## Technology Stack
- **Web Framework**: Flask
- **Machine Learning**: Scikit-learn, XGBoost, RandomForestRegressor
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Frontend**: HTML, CSS
- **Deployment**: Localhost (Python)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Madhava-sb/Traffictelligence.git
   cd TrafficTelligence/Flask
