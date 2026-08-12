import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# 1. Define the PyTorch Model
class LidarParkingModel(nn.Module):
    def __init__(self):
        super(LidarParkingModel, self).__init__()
        # Matches Keras Dense layers
        self.layer1 = nn.Linear(180, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

def main():
    # 2. Load and Prepare the Data
    df = pd.read_csv("parking_lidar_dataset.csv")
    
    # Extract features and labels
    x_train = df.drop("y", axis=1).to_numpy()
    y_train = df['y'].to_numpy()
    
    # Convert NumPy arrays to PyTorch Tensors
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    # y must be reshaped to (batch_size, 1) for Binary Crossentropy Loss
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader for batching
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = LidarParkingModel()
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train the Model (Explicit Training Loop)
    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            # Step A: Zero the gradients
            optimizer.zero_grad()
            
            # Step B: Forward pass
            predictions = model(batch_x)
            
            # Step C: Compute Loss
            loss = criterion(predictions, batch_y)
            
            # Step D: Backward pass (Compute gradients)
            loss.backward()
            
            # Step E: Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    # 5. Save the Model (Recommended PyTorch format is state_dict)
    torch.save(model.state_dict(), "lidar_parking_model.pth")
    print("Model saved to 'lidar_parking_model.pth'")

    # 6. Test the Model with your sample data
    model.eval() # Set model to evaluation mode (important for inference)
    
    test_data = np.array([
        10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,1.8548,10.0,10.0,10.0,1.2636,10.0,10.0,10.0,10.0,10.0,1.9504,10.0,10.0,10.0,10.0,10.0,10.0,0.7031,10.0,1.395,2.1056,10.0,10.0,10.0,1.8787,10.0,10.0,10.0,10.0,10.0,1.6123,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,1.2214,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,0.3115,0.3115,0.3141,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,0.6566,0.6293,0.6168,0.5941,0.5836,0.5788,0.5831,0.5921,0.6014,0.6061,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,7.8688,7.8363,7.8167,7.799,6.9789,6.9792,6.9815,6.9865,7.7474,7.7558,7.7431,7.7567,7.7467,7.7661,7.7592,7.7731,7.782,8.1814,10.0,10.0,1.4684,1.4605,1.4456,1.4385,1.4829,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0

    ]).reshape(1, 180) # Reshaped to (1, 180) as corrected earlier!
    
    # Convert test data to tensor
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    
    # Disable gradient tracking for inference to save memory/speed up
    with torch.no_grad():
        prediction = model(test_tensor)
        prob = prediction.item()
        
        print("\n--- Test Results ---")
        print(f"Raw Probability: {prob:.4f}")
        if prob > 0.5:
            print("Prediction: TURN LEFT (1)")
        else:
            print("Prediction: GO STRAIGHT (0)")

if __name__ == "__main__":
    main()