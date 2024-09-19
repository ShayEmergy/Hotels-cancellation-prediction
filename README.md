# Hotel Reservation Cancellation Predictor

## Project Overview
This project uses machine learning to predict the likelihood of hotel reservation cancellations and calculates appropriate cancellation fees. By analyzing various factors related to bookings, the model helps hotels optimize their pricing strategies and manage reservation risks more effectively.

## Dataset Overview
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand?resource=download
Source: Kaggle
Size: 119,390 observations
Features: 32 variables
Time Period: 2015-2017

## Features
- Predicts cancellation probability for hotel reservations
- Calculates cancellation fees based on predicted probabilities
- Handles both complete and partial booking data
- Uses a neural network model trained on historical booking data

## Technologies Used
- Python
- PyTorch
- pandas
- scikit-learn
- NumPy
- Matplotlib

## Installation
To set up this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hotel-reservation-cancellation-predictor.git
   ```
2. Navigate to the project directory:
   ```
   cd hotel-reservation-cancellation-predictor
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the cancellation predictor:

1. Prepare your input data in the format specified in the notebook.
2. Load the trained model and scaler.
3. Use the `predict_and_calculate_fee` function to get cancellation probabilities and fees.

Example:
```python
cancel_prob, fee = predict_and_calculate_fee(reservation_data, booking_price, model, scaler)
print(f"Cancellation Probability: {cancel_prob:.2f}")
print(f"Calculated Fee: ${fee:.2f}")
```

## Model Performance
The current model achieves an overall accuracy of 83.84% on the test set, with varying performance across different cancellation categories.

## Future Improvements
- Address class imbalance for better prediction of rare events (e.g., no-shows)
- Experiment with different model architectures and hyperparameters
- Incorporate more features or external data sources to improve predictions
- Remove low-weight features to reduce noise and improve model efficiency

## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or feedback, please open an issue on this GitHub repository.

