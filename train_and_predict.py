import torch
from cnn_model import SimpleGazeNet
import torch.nn as nn
import torch.optim as optim
import os
from global_info import GlobalInfo
from torch.utils.data import DataLoader


class GazeController:
    def __init__(self, ui_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGazeNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.ui_callback = ui_callback
        self.training = False
        self.is_predict = False

        # Load existing model if exists
        if os.path.exists("gaze_model.pth"):
            self.model.load_state_dict(torch.load("gaze_model.pth", map_location=self.device))

    def train_model(self):
        if self.training:
            return
        self.training = True
        if self.ui_callback:
            self.ui_callback("training_started")
        print("Starting model training...")
        dataset = GlobalInfo.train_data
        if len(dataset) < 16:
            print("Not enough 16 samples for training")
            self.training = False
            if self.ui_callback:
                self.ui_callback("training_finished")
            return

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        self.model.train()

        for epoch in range(10):
            total_loss = 0
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(dataloader):.4f}")
            if self.ui_callback:
                self.ui_callback("training_progress", epoch + 1, total_loss / len(dataloader))

        # Save trained model
        torch.save(self.model.state_dict(), "gaze_model.pth")
        self.model.eval()
        print("Model training completed.")
        self.training = False
        GlobalInfo.train_data.clear_samples()

    def predict_gaze(self, camera_frame):
        if self.is_predict:
            return
        self.is_predict = True
        with torch.no_grad():
            image = torch.from_numpy(camera_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            image = image.to(self.device)
            output = self.model(image)
            coords = output.cpu().numpy()[0]

            # Convert normalized coordinates to screen coordinates
            x = int(coords[0] * GlobalInfo.screen_width)
            y = int(coords[1] * GlobalInfo.screen_height)

            self.is_predict = False

            return x, y
