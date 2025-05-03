import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, name='default'):
        super().__init__()
        self.name = name

        # 输入是单通道图像 (1, 65, 65)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # ResNet块
        self.layer1 = self._make_layer(8, 16, 1, stride=2)
        self.layer2 = self._make_layer(16, 32, 1, stride=2)
        self.layer3 = self._make_layer(32, 64, 1, stride=2) # Output: [B, 64, 5, 5]

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Output is the spatial feature map [B, 64, 5, 5]
        return x

class Predictor(nn.Module):
    def __init__(self, input_channels=64, action_dim=2, spatial_h=5, spatial_w=5):
        super().__init__()
        self.input_channels = input_channels
        self.action_dim = action_dim
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

        # Action projection: Project action and reshape to spatial dimensions
        # Match state channels (64) for concatenation
        self.action_proj_channels = input_channels
        self.action_proj_dim = self.action_proj_channels * spatial_h * spatial_w # 64 * 5 * 5 = 1600
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, 256), # Intermediate size
            nn.ReLU(),
            nn.Linear(256, self.action_proj_dim),
            nn.LayerNorm(self.action_proj_dim)
        )

        # Input channels for ResNet = state_channels + action_channels = 64 + 64 = 128
        conv_input_channels = self.input_channels + self.action_proj_channels
        internal_channels = 256 # Hyperparameter for internal conv layers

        # 三层ResNet网络
        self.res_net = nn.Sequential(
            # 第一层
            ResidualBlock(conv_input_channels, internal_channels),
            # 第二层
            ResidualBlock(internal_channels, internal_channels),
            # 第三层
            ResidualBlock(internal_channels, self.input_channels)
        )

    def forward(self, state_map, action):
        """
        Args:
            state_map: [B, input_channels, spatial_h, spatial_w] (e.g., [B, 64, 5, 5])
            action: [B, action_dim]
        Returns:
            next_state_map: [B, input_channels, spatial_h, spatial_w] (e.g., [B, 64, 5, 5])
        """
        B = state_map.shape[0]

        # 1. Project and reshape action
        action_projected = self.action_proj(action) # [B, action_proj_dim]
        action_reshaped = action_projected.view(B, self.action_proj_channels, self.spatial_h, self.spatial_w)

        # 2. Concatenate state map and reshaped action along the channel dimension
        x = torch.cat([state_map, action_reshaped], dim=1) # [B, 128, 5, 5]

        # 3. Pass through ResNet to predict the change
        predicted_change_map = self.res_net(x) # [B, 64, 5, 5]

        # 4. Add residual connection
        next_state_map = state_map + predicted_change_map

        return next_state_map

class JEPA(nn.Module):
    def __init__(self, hidden_dim=128, action_dim=2):
        super().__init__()
        # Store original hidden_dim for potential config compatibility/reference
        self.config_hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Encoders output fixed spatial size [B, 64, 5, 5]
        self.agent_encoder = Encoder()
        self.wall_encoder = Encoder()
        self.target_agent_encoder = Encoder()
        self.target_wall_encoder = Encoder()

        # Determine predictor input channels and spatial size from a dummy forward pass
        dummy_input = torch.randn(1, 1, 65, 65)
        with torch.no_grad():
            dummy_output_map = self.agent_encoder(dummy_input)
        _, self.predictor_input_channels, self.spatial_h, self.spatial_w = dummy_output_map.shape
        print(f"Encoder output spatial size: C={self.predictor_input_channels}, H={self.spatial_h}, W={self.spatial_w}")

        # Predictor takes spatial input and outputs spatial map
        self.predictor = Predictor(
            input_channels=self.predictor_input_channels,
            action_dim=action_dim,
            spatial_h=self.spatial_h,
            spatial_w=self.spatial_w
        )

        # Representation dimension for loss is the flattened spatial map size
        self.repr_dim = self.predictor_input_channels * self.spatial_h * self.spatial_w # 64 * 5 * 5 = 1600
        print(f"Representation dimension (flattened for loss): {self.repr_dim}")

        self.ema_decay = 0.99
        self._init_target_encoders()

    def _init_target_encoders(self):
        for param_q, param_k in zip(self.agent_encoder.parameters(),
                                  self.target_agent_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.wall_encoder.parameters(),
                                  self.target_wall_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self):
        for param_q, param_k in zip(self.agent_encoder.parameters(),
                                  self.target_agent_encoder.parameters()):
            param_k.data = param_k.data * self.ema_decay + \
                          param_q.data * (1 - self.ema_decay)
        
        for param_q, param_k in zip(self.wall_encoder.parameters(),
                                  self.target_wall_encoder.parameters()):
            param_k.data = param_k.data * self.ema_decay + \
                          param_q.data * (1 - self.ema_decay)

    def forward(self, states=None, actions=None, next_obs=None, teacher_forcing=False):
        """
        Args:
            states: 当前状态 [B, 1, C, H, W] or [B, C, H, W] (Training/Inference)
            actions: 动作 [B, T-1, A] (Training/Inference)
            next_obs: 下一个状态 [B, C, H, W] (Training only)
            teacher_forcing: 是否使用teacher forcing
        Returns:
            训练模式 (teacher_forcing=True):
                (pred_state_flat, target_state_flat) - Flattened tensors for loss [B, repr_dim]
            评估模式 (teacher_forcing=False):
                pred_encs_flat - Flattened representations [B, Seq+1, repr_dim]
        """
        if teacher_forcing and next_obs is not None:
            # Get current spatial representation
            current_state_map = self.get_representation(states) # [B, 64, 5, 5]
            # Predict next spatial representation
            pred_state_map = self.predictor(current_state_map, actions) # [B, 64, 5, 5]

            # Get target spatial representation (using target encoders)
            with torch.no_grad():
                target_state_map = self.get_target_representation(next_obs) # [B, 64, 5, 5]

            # Flatten for loss calculation
            B = pred_state_map.shape[0]
            pred_state_flat = pred_state_map.view(B, -1) # [B, 1600]
            target_state_flat = target_state_map.view(B, -1) # [B, 1600]

            return pred_state_flat, target_state_flat

        # --- Evaluation/Inference Mode --- (teacher_forcing=False)
        # Input states: [B, 1, C, H, W] (initial state)
        # Input actions: [B, T-1, A]
        B = states.shape[0]
        seq_len = actions.shape[1] # T-1

        # Output tensor for flattened representations
        pred_encs_flat = torch.zeros((B, seq_len + 1, self.repr_dim), device=states.device)

        # Get initial spatial state
        current_state_map = self.get_representation(states) # [B, 64, 5, 5]
        pred_encs_flat[:, 0] = current_state_map.view(B, -1) # Store flattened initial state

        # Rollout predictions step-by-step
        for t in range(seq_len):
            current_action = actions[:, t] # [B, A]
            # Predict next *spatial* state map
            next_state_map = self.predictor(current_state_map, current_action) # [B, 64, 5, 5]
            # Store *flattened* representation
            pred_encs_flat[:, t + 1] = next_state_map.view(B, -1) # [B, 1600]
            # Update internal state for next prediction step (use spatial map)
            current_state_map = next_state_map

        return pred_encs_flat

    def get_representation(self, obs):
        """获取单个观察的 *spatial* 表示
        Args:
            obs: [B, 1, C, H, W] 或 [B, C, H, W]
        Returns:
            representation: [B, 64, 5, 5]
        """
        # 确保输入是4D或5D
        if len(obs.shape) == 5:
            # [B, 1, C, H, W] -> [B, C, H, W]
            obs = obs.squeeze(1)
        
        # 分离智能体和墙壁的观察
        agent_obs = obs[:, 0:1]  # [B, 1, H, W]
        wall_obs = obs[:, 1:2]   # [B, 1, H, W]
        
        # 编码
        agent_repr = self.agent_encoder(agent_obs)
        wall_repr = self.wall_encoder(wall_obs)
        
        # 合并表示 (Addition)
        return agent_repr + wall_repr # Output: [B, 64, 5, 5]

    @torch.no_grad()
    def get_target_representation(self, obs):
        """获取单个观察的 *target spatial* 表示"""
        if len(obs.shape) == 5:
            obs = obs.squeeze(1)

        agent_obs = obs[:, 0:1]
        wall_obs = obs[:, 1:2]

        agent_repr = self.target_agent_encoder(agent_obs)
        wall_repr = self.target_wall_encoder(wall_obs)

        return agent_repr + wall_repr # Output: [B, 64, 5, 5]
