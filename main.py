import torch
from torch.utils.data import DataLoader
from dataset import MultiSourceTimeIsolatedDataset
from model import FlowConvLSTM
from trainer import FlowTrainer
import os
import random
import numpy as np
import datetime
import shutil
import json
import time
import gc

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dir(base_output_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config, output_dir):
    config_path = os.path.join(output_dir, "config.json")
    config_dict = config.__dict__.copy()
    config_dict['MODEL_INPUT_CHANNELS'] = config.MODEL_INPUT_CHANNELS
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ 配置已保存到 {config_path}")


def save_script_snapshots(output_dir):
    script_names = ["main.py", "model.py", "trainer.py", "dataset.py"]
    dest_dir = os.path.join(output_dir, "scripts")
    os.makedirs(dest_dir, exist_ok=True)
    for script_name in script_names:
        try:
            script_path = os.path.join(os.path.dirname(__file__), script_name)
            dest_path = os.path.join(dest_dir, script_name)
            shutil.copy2(script_path, dest_path)
            print(f"✅ 脚本已保存: {script_name}")
        except Exception as e:
            print(f"⚠️ 无法保存脚本 {script_name}: {str(e)}")


# ---------------- 主函数 ----------------
def main():
    # 初始化配置
    config = Config()
    set_seed(config.SEED)
    output_dir = create_output_dir(config.BASE_OUTPUT_DIR)

    # 保存配置和脚本
    save_config(config, output_dir)
    save_script_snapshots(output_dir)

    # 设置设备
    device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"输出目录: {output_dir}")

    # 数据源
    data_paths = {name: os.path.abspath(path) for name, path in config.DATA_SOURCES.items()}
    print("\n使用的数据源:")
    for name, path in data_paths.items():
        print(f"• {name}: {path}")
    print(f"输入通道数: {config.MODEL_INPUT_CHANNELS}")

    # ---------------- 数据集 ----------------
    print("\n创建数据集...")

    # 训练集
    print("\n[1/3] 创建训练集...")
    train_dataset = MultiSourceTimeIsolatedDataset(
        data_paths=data_paths,
        split='train',
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        step=config.STEP,
        norm_stats=None,
        min_safe_buffer=config.MIN_SAFE_BUFFER,
        preload_to_memory=config.PRELOAD_MEMORY
    )
    norm_stats = train_dataset.get_norm_stats()

    # 验证集
    print("\n[2/3] 创建验证集...")
    val_dataset = MultiSourceTimeIsolatedDataset(
        data_paths=data_paths,
        split='val',
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        step=config.STEP,
        norm_stats=norm_stats,
        min_safe_buffer=config.MIN_SAFE_BUFFER,
        preload_to_memory=config.PRELOAD_MEMORY
    )

    # 测试集
    print("\n[3/3] 创建测试集...")
    test_dataset = MultiSourceTimeIsolatedDataset(
        data_paths=data_paths,
        split='test',
        input_steps=config.INPUT_STEPS,
        output_steps=config.OUTPUT_STEPS,
        step=config.STEP,
        norm_stats=norm_stats,
        min_safe_buffer=config.MIN_SAFE_BUFFER,
        preload_to_memory=config.PRELOAD_MEMORY
    )

    # 数据集统计
    total_sequences = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print("\n数据集统计:")
    print(f"• 训练集: {len(train_dataset)}序列 ({len(train_dataset)/total_sequences*100:.1f}%)")
    print(f"• 验证集: {len(val_dataset)}序列 ({len(val_dataset)/total_sequences*100:.1f}%)")
    print(f"• 测试集: {len(test_dataset)}序列 ({len(test_dataset)/total_sequences*100:.1f}%)")

    # ---------------- 数据加载器 ----------------
    print("\n创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )

    # ---------------- 模型 ----------------
    print("\n初始化模型...")

    # 动态获取 dataset 实际通道数
    actual_input_channels = train_dataset[0][0].shape[1]  # [T, C, H, W]

    # 添加检查
    if actual_input_channels <= 0:
        raise ValueError(f"无效的输入通道数: {actual_input_channels}。请检查数据集实现。")

    print(f"[模型架构]")
    print(f"• 输入通道数: {actual_input_channels}")
    print(f"• 隐藏层: {config.HIDDEN_CHANNELS[0]}-{config.HIDDEN_CHANNELS[1]}-{config.HIDDEN_CHANNELS[2]}")
    print(f"• 卷积核大小: {config.KERNEL_SIZE}x{config.KERNEL_SIZE}")

    # 使用新的模型类 FlowConvLSTM
    model = FlowConvLSTM(
        input_channels=actual_input_channels,
        hidden_channels=config.HIDDEN_CHANNELS,
        output_steps=config.OUTPUT_STEPS
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"• 参数数量: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # ---------------- 训练器 ----------------
    # 使用新的训练器类 FlowTrainer
    trainer = FlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        early_stopping_delta=config.EARLY_STOPPING_DELTA,
        save_model_every_epoch=config.SAVE_MODEL_EVERY_EPOCH,
        initial_teacher_forcing=0.9,
        final_teacher_forcing=0.3
    )

    # ---------------- 开始训练 ----------------
    print("\n开始训练...")
    print(f"[训练配置]")
    print(f"• 轮数: {config.EPOCHS}")
    print(f"• 学习率: {config.LEARNING_RATE}")
    print(f"• 权重衰减: {config.WEIGHT_DECAY}")
    print(f"• 早停: {config.EARLY_STOPPING_PATIENCE}轮, 阈值={config.EARLY_STOPPING_DELTA}")
    print(f"• 保存模型间隔: {config.SAVE_MODEL_EVERY_EPOCH}轮")

    start_time = time.time()
    trainer.train(num_epochs=config.EPOCHS)
    training_time = time.time() - start_time

    print(f"\n训练完成! 耗时: {training_time/3600:.2f}小时")
    print(f"所有输出已保存到: {output_dir}")

    # ---------------- 清理资源 ----------------
    del train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    gc.collect()


if __name__ == "__main__":

    main()
