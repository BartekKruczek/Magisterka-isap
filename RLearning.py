import torch

class MyCustomRL:
    def __init__(self, model = None, optimizer = None):
        # here is model passed from peft
        self.model = model
        self.optimizer = self.get_optimizer(optimizer)

    def __repr__(self) -> str:
        return "Klasa do obsługi RL"
    
    def get_optimizer(self, optimizer) -> torch.optim:
        if optimizer:
            return optimizer
        else:
            return torch.optim.Adam(self.model.parameters(), lr = 1e-5)
        
    def step(self, reward: int):
        loss = -reward

        # back - propagation section
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()