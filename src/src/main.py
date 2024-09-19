import torch
from data_preparation import load_and_preprocess_data, prepare_features_and_target, split_and_scale_data
from model import HotelCancellationModel, train_model, evaluate_model
from utils import predict_and_calculate_fee

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('../data/hotel_bookings.csv')
    X, y = prepare_features_and_target(data)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Initialize and train the model
    model = HotelCancellationModel(X_train.shape[1])
    trained_model = train_model(model, X_train_tensor, y_train_tensor)

    # Evaluate the model
    accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
    print(f'Model Accuracy: {accuracy:.4f}')

    # Example prediction
    test_reservation = [30, 2, 3, 2, 0, 0, 0, 0, 1, 0, 150.0, 0, 1] + [0] * 61  # Simplified example
    test_booking_price = 750.0

    cancel_prob, fee = predict_and_calculate_fee(test_reservation, test_booking_price, trained_model, scaler)
    print(f"Cancellation Probability: {cancel_prob:.2f}")
    print(f"Calculated Fee: ${fee:.2f}")

if __name__ == "__main__":
    main()
