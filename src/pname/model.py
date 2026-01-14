import torch
from torch import nn
from omegaconf import DictConfig


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, model_cfg: DictConfig = None) -> None:
        super().__init__()
        if model_cfg is None:
            # Default values for backward compatibility
            model_cfg = DictConfig({
                'conv1': {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1},
                'conv2': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1},
                'conv3': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1},
                'dropout': 0.5,
                'fc1': {'in_features': 128, 'out_features': 10},
                'max_pool': {'kernel_size': 2, 'stride': 2}
            })
        
        self.conv1 = nn.Conv2d(
            model_cfg.conv1.in_channels,
            model_cfg.conv1.out_channels,
            model_cfg.conv1.kernel_size,
            model_cfg.conv1.stride
        )
        self.conv2 = nn.Conv2d(
            model_cfg.conv2.in_channels,
            model_cfg.conv2.out_channels,
            model_cfg.conv2.kernel_size,
            model_cfg.conv2.stride
        )
        self.conv3 = nn.Conv2d(
            model_cfg.conv3.in_channels,
            model_cfg.conv3.out_channels,
            model_cfg.conv3.kernel_size,
            model_cfg.conv3.stride
        )
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.fc1 = nn.Linear(
            model_cfg.fc1.in_features,
            model_cfg.fc1.out_features
        )
        self.max_pool_kernel = model_cfg.max_pool.kernel_size
        self.max_pool_stride = model_cfg.max_pool.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
