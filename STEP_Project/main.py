import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.step_model import STEP
from dataset import PedestrianDataset
from config import Config
from utils import ade_fde

def train():
    # 1. Setup
    model = STEP().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_dataset = PedestrianDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Loss Functions
    bce_loss = torch.nn.BCELoss() # For Endpoint (Eq. 17)
    mse_loss = torch.nn.MSELoss() # For Trajectory (Eq. 18)

    print("STARTING TRAINING...")
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (obs, pred_gt, scene, goal_gt) in enumerate(train_loader):
            obs = obs.to(Config.DEVICE)
            pred_gt = pred_gt.to(Config.DEVICE)
            scene = scene.to(Config.DEVICE)
            goal_gt = goal_gt.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            pred_traj, pred_map = model(obs, scene)
            
            # Calculate Loss
            # 1. Endpoint Loss
            l_ep = bce_loss(pred_map, goal_gt)
            
            # 2. Trajectory Loss
            l_traj = mse_loss(pred_traj, pred_gt)
            
            # 3. Total Loss (Eq. 19)
            loss = l_ep + Config.LAMBDA_MSE * l_traj
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] \t Loss: {total_loss / len(train_loader):.4f}")
            
    # Save Model
    torch.save(model.state_dict(), "step_final.pth")
    print("Training Completed.")

if __name__ == "__main__":
    train()