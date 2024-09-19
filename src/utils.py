import torch
import numpy as np

def predict_and_calculate_fee(features, booking_price, model, scaler, expected_features=74):
    processed_features = [0 if feature is None or feature == '' else float(feature) for feature in features]
    
    if len(processed_features) < expected_features:
        processed_features.extend([0] * (expected_features - len(processed_features)))
    elif len(processed_features) > expected_features:
        processed_features = processed_features[:expected_features]
    
    features_array = np.array(processed_features).reshape(1, -1)
    scaled_features = torch.tensor(scaler.transform(features_array), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output = model(scaled_features)
        probabilities = torch.softmax(output, dim=1)
    
    cancel_prob = probabilities[0, 1].item()
    
    if cancel_prob <= 0.25:
        fee = booking_price * 0.10
    elif cancel_prob <= 0.50:
        fee = booking_price * 0.25
    elif cancel_prob <= 0.75:
        fee = booking_price * 0.50
    else:
        fee = booking_price * 0.75
    
    return cancel_prob, fee
