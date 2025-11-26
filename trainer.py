import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import warnings
from tqdm import tqdm
import torch.cuda.amp as amp
import gc
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MomentumFlowLoss(nn.Module):
    """
    自定义光流损失函数:
    包含两部分：
    1) 光流本身的 MSE (经典监督)
    2) 动量守恒约束 (u ∂u/∂x + v ∂u/∂y)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(MomentumFlowLoss, self).__init__()
        self.alpha = alpha  # 权重: 基础 MSE
        self.beta = beta    # 权重: 动量约束

    def forward(self, pred, target):
        """
        pred, target: [B, T, 2, H, W]
        """
        # ----- 1. 光流的基础 MSE -----
        mse_loss = torch.mean((pred - target) ** 2)

        # ----- 2. 动量守恒约束 -----
        u, v = pred[:, :, 0], pred[:, :, 1]           # [B, T, H, W]
        u_gt, v_gt = target[:, :, 0], target[:, :, 1]

        # 梯度 (一阶差分近似)
        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]      # [B, T, H, W-1]
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]      # [B, T, H-1, W]

        du_dx_gt = u_gt[:, :, :, 1:] - u_gt[:, :, :, :-1]
        dv_dy_gt = v_gt[:, :, 1:, :] - v_gt[:, :, :-1, :]

        # 对齐裁剪 (取交集区域)
        min_w = min(du_dx.shape[-1], dv_dy.shape[-1])
        min_h = min(du_dx.shape[-2], dv_dy.shape[-2])

        du_dx = du_dx[:, :, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_h, :min_w]
        du_dx_gt = du_dx_gt[:, :, :min_h, :min_w]
        dv_dy_gt = dv_dy_gt[:, :, :min_h, :min_w]

        # 动量残差: (∂u/∂x + ∂v/∂y)
        momentum_pred = du_dx + dv_dy
        momentum_gt = du_dx_gt + dv_dy_gt
        momentum_loss = torch.mean((momentum_pred - momentum_gt) ** 2)

        # ----- 总损失 -----
        return self.alpha * mse_loss + self.beta * momentum_loss


class FlowTrainer:
    """光流训练器（只处理光流输出）"""

    def __init__(
            self,
            model,
            train_loader,
            val_loader=None,
            test_loader=None,
            output_dir="save",
            learning_rate=1e-3,
            weight_decay=1e-5,
            early_stopping_patience=10,
            early_stopping_delta=0.001,
            save_model_every_epoch=10,
            initial_teacher_forcing=0.9,
            final_teacher_forcing=0.3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "history"), exist_ok=True)

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.save_model_every_epoch = save_model_every_epoch

        # Teacher forcing
        self.initial_tf = initial_teacher_forcing
        self.final_tf = final_teacher_forcing

        # AMP
        self.scaler = amp.GradScaler() if device.type == "cuda" else None

        # 光流损失
        self.criterion_flow = MomentumFlowLoss(alpha=1.0, beta=1.0)

        # 优化器 & 调度器
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=3
        )

        # 早停
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        # 历史记录
        self.train_history, self.val_history, self.lr_history = [], [], []
        self.final_test_loss = None

    def get_teacher_forcing_ratio(self, epoch, total_epochs):
        """计算教师强制比率"""
        if total_epochs == 1:
            return self.initial_tf
        return self.initial_tf - (self.initial_tf - self.final_tf) * (epoch - 1) / (total_epochs - 1)

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0

        bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

        # 线性衰减 teacher forcing
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(epoch, total_epochs)

        for inputs, targets in bar:
            # 将输入移动到设备
            inputs = inputs.to(device, non_blocking=True)

            # 将目标字典中的光流张量移动到设备
            flow_targets = targets['flow'].to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with amp.autocast():
                    # 模型调用不再需要亮温目标
                    outputs = self.model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)

                    # 计算光流损失
                    loss = self.criterion_flow(outputs['flow'], flow_targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = self.criterion_flow(outputs['flow'], flow_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.8f}")

        bar.close()

        return total_loss / len(self.train_loader)

    def _evaluate(self, loader, desc="Evaluating"):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=desc, leave=False):
                inputs = inputs.to(device)
                flow_targets = targets['flow'].to(device)
                outputs = self.model(inputs, teacher_forcing_ratio=0.0)

                loss = self.criterion_flow(outputs['flow'], flow_targets)
                total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self):
        if not self.val_loader:
            return None
        return self._evaluate(self.val_loader, desc="Validating")

    def test(self):
        if not self.test_loader:
            return None
        test_loss = self._evaluate(self.test_loader, desc="Testing")
        self.final_test_loss = test_loss

        with torch.no_grad():
            for inputs, _ in self.test_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs, teacher_forcing_ratio=0.0)
                print(f"✅ Test prediction flow shape: {outputs['flow'].shape}")
                break

        return test_loss

    def save_model(self, filename, epoch, loss):
        fname = filename.replace(".pth", f"_loss_{loss:.8f}.pth")
        path = os.path.join(self.output_dir, "models", fname)
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
        }
        if self.scaler:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        print(f"💾 Saved model: {path}")

    def save_history_to_npy(self):
        history_dir = os.path.join(self.output_dir, "history")
        os.makedirs(history_dir, exist_ok=True)

        np.save(os.path.join(history_dir, "train_history.npy"), np.array(self.train_history))
        np.save(os.path.join(history_dir, "val_history.npy"), np.array(self.val_history))
        np.save(os.path.join(history_dir, "lr_history.npy"), np.array(self.lr_history))

        if self.final_test_loss is not None:
            np.save(os.path.join(history_dir, "final_test_loss.npy"), np.array([self.final_test_loss]))

        print(f"📊 Saved training history to {history_dir}")

    def plot_loss_curves(self):
        plt.figure(figsize=(12, 6))

        epochs = range(1, len(self.train_history) + 1)
        plt.plot(epochs, self.train_history, label="Train Loss", linewidth=2)
        plt.plot(epochs, self.val_history, label="Validation Loss", linewidth=2)

        if self.final_test_loss is not None:
            plt.scatter(len(epochs), self.final_test_loss, s=150, color='red', marker='*',
                        label=f"Test Loss: {self.final_test_loss:.8f}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training, Validation and Test Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "models", "loss_curve.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved loss curve to {fig_path}")

    def train(self, num_epochs=50):
        print("🚀 Start training...\n")
        print(f"Training flow prediction model")

        log_file = os.path.join(self.output_dir, "training_log.txt")
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("Epoch\tTrain Loss\tVal Loss\tLearning Rate\n")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_epochs)

            val_loss = self.validate() if self.val_loader else train_loss

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            self.lr_history.append(lr)

            with open(log_file, "a") as f:
                f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t{lr:.8f}\n")

            print(f"Epoch {epoch:03d}/{num_epochs} | "
                  f"Train: {train_loss:.8f} | "
                  f"Val: {val_loss:.8f} | "
                  f"LR: {lr:.8f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"best_epoch_{epoch}.pth", epoch, val_loss)
                self.early_stopping_counter = 0
            elif self.save_model_every_epoch > 0 and epoch % self.save_model_every_epoch == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth", epoch, val_loss)

            if val_loss >= self.best_val_loss - self.early_stopping_delta:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"\n⏹ Early stopping at epoch {epoch}")
                    break
            else:
                self.early_stopping_counter = 0

            gc.collect()
            torch.cuda.empty_cache()

        if self.test_loader:
            test_loss = self.test()
            self.final_test_loss = test_loss
            self.save_model("final_model.pth", num_epochs, self.final_test_loss)

        self.save_history_to_npy()
        self.plot_loss_curves()
        print("\n🏁 Training finished.")
        return self.final_test_loss