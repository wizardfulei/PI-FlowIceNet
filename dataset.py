import numpy as np
import os
import torch
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
import gc
import datetime
import tempfile
import uuid
import matplotlib as mpl
import cv2
import random
import shutil

mpl.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings("ignore", category=FutureWarning)


class MultiSourceTimeIsolatedDataset(Dataset):
    """支持多源输入的时间隔离数据集"""

    def __init__(self, data_paths, split, input_steps, output_steps, step, norm_stats=None,
                 min_safe_buffer=5, preload_to_memory=False, seed=42):
        # 初始化参数
        self.data_paths = data_paths
        self.source_names = sorted(list(data_paths.keys()))
        self.split = split
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.total_steps = input_steps + output_steps
        self.step = step
        self.min_safe_buffer = min_safe_buffer
        self.preload_to_memory = preload_to_memory
        self.seed = seed

        # 初始化数据结构
        self.sequence_starts = []
        self.block_data = {}
        self.block_indices = {}
        self.block_safe_params = {}
        self.block_info = {}
        self.norm_params = {}
        self.temp_dir = None  # 用于存储临时文件

        # 主数据源（默认为第一个）
        self.main_source = self.source_names[0] if self.source_names else None

        # 首先扫描主文件获取总天数统计
        if self.main_source:
            main_path = data_paths[self.main_source]
            with h5py.File(main_path, 'r') as hf:
                if 'blocks' in hf:
                    blocks_group = hf['blocks']
                    self.block_names = sorted([k for k in blocks_group.keys() if k.startswith('block')])
                else:
                    self.block_names = sorted([k for k in hf.keys() if k.startswith('block')])

        # 提前识别有效区块
        self.block_names = self.identify_valid_blocks()

        # 创建区块索引映射
        self.block_indices = {}
        for idx, block_name in enumerate(self.block_names):
            self.block_indices[block_name] = idx

        # 加载区块数据并计算每个区块的安全参数
        self.load_data_and_compute_params()

        # 计算归一化参数（如果没有提供）
        self.compute_normalization_stats(norm_stats)

        # 创建序列
        self.create_sequences()

        # 输出数据集信息
        self.print_dataset_info()

    def identify_valid_blocks(self):
        """提前识别有足够数据的有效区块"""
        valid_blocks = []
        main_path = self.data_paths[self.main_source]
        with h5py.File(main_path, 'r') as hf:
            for block_name in tqdm(self.block_names, desc="筛选有效区块", leave=False):
                if self.block_exists(hf, block_name):
                    ds = self.get_block_dataset(hf, block_name)
                    total_days = len(ds)

                    safe_buffer = max(self.min_safe_buffer, int(self.total_steps * 0.1))
                    safe_start = safe_buffer
                    safe_end = max(safe_start, total_days - safe_buffer - self.total_steps)

                    if safe_end - safe_start >= self.total_steps:
                        self.block_safe_params[block_name] = {
                            'total_days': total_days,
                            'safe_start': safe_start,
                            'safe_end': safe_end
                        }
                        valid_blocks.append(block_name)
        return valid_blocks

    def block_exists(self, hf, block_name):
        """检查区块是否存在"""
        if 'blocks' in hf and block_name in hf['blocks']:
            return True
        return block_name in hf

    def get_block_dataset(self, hf, block_name):
        """获取区块数据集"""
        if 'blocks' in hf and block_name in hf['blocks']:
            return hf['blocks'][block_name]
        return hf[block_name]

    def print_dataset_info(self):
        """打印数据集信息摘要"""
        sources = ', '.join(self.source_names)
        print(f"数据集类型: {self.split.upper()}")
        print(f"数据源: {len(self.source_names)}个 ({sources})")
        print(f"区块数量: {len(self.block_names)}")
        print(f"总序列数: {len(self.sequence_starts)}")
        for source_name, (min_val, max_val) in self.norm_params.items():
            print(f" - {source_name}: min={min_val:.4f}, max={max_val:.4f}")

    def load_data_and_compute_params(self):
        """加载区块数据并计算每个区块的安全参数"""
        main_path = self.data_paths[self.main_source]
        main_days = None
        with h5py.File(main_path, 'r') as hf:
            for block_name in self.block_names:
                if self.block_exists(hf, block_name):
                    ds = self.get_block_dataset(hf, block_name)
                    if ds.ndim == 4:
                        self.block_info[block_name] = (ds.shape[1], ds.shape[2], ds.shape[3])
                    else:
                        self.block_info[block_name] = (ds.shape[1], ds.shape[2], 1)
                    if main_days is None:
                        main_days = len(ds)

        # 创建临时目录用于存储内存映射文件
        if not self.preload_to_memory:
            self.temp_dir = tempfile.mkdtemp(prefix=f"multisource_{os.getpid()}_")
            print(f"创建临时目录用于内存映射文件: {self.temp_dir}")

        for source_name, path in self.data_paths.items():
            source_data = {}
            with h5py.File(path, 'r') as hf:
                progress_bar = tqdm(self.block_names, desc=f"加载源: {source_name}", leave=False)
                for block_name in progress_bar:
                    if self.block_exists(hf, block_name):
                        ds = self.get_block_dataset(hf, block_name)
                        block_arr = ds[:]
                        current_days = len(block_arr)

                        if current_days != main_days:
                            if source_name == 'date_index':
                                if block_arr.ndim == 4:
                                    adjusted_arr = np.zeros((main_days, *block_arr.shape[1:]), dtype=np.float32)
                                else:
                                    adjusted_arr = np.zeros((main_days, *block_arr.shape[1:]), dtype=np.float32)
                                min_len = min(main_days, current_days)
                                adjusted_arr[:min_len] = block_arr[:min_len]
                                if min_len < main_days:
                                    last_date = block_arr[min_len - 1] if min_len > 0 else 0.0
                                    adjusted_arr[min_len:] = last_date
                                block_arr = adjusted_arr
                            else:
                                if current_days > main_days:
                                    block_arr = block_arr[:main_days]
                                else:
                                    if block_arr.ndim == 4:
                                        padded = np.zeros((main_days, *block_arr.shape[1:]), dtype=np.float32)
                                    else:
                                        padded = np.zeros((main_days, *block_arr.shape[1:]), dtype=np.float32)
                                    padded[:current_days] = block_arr
                                    block_arr = padded
                        if self.preload_to_memory:
                            source_data[block_name] = block_arr
                        else:
                            mmap_path = os.path.join(self.temp_dir, f"{source_name}_{block_name}.npy")
                            np.save(mmap_path, block_arr)
                            source_data[block_name] = np.load(mmap_path, mmap_mode='r')
                    else:
                        warnings.warn(f"数据源 '{source_name}' 或区块 '{block_name}' 不存在，使用零填充")
                        if block_name in self.block_info:
                            h, w, c = self.block_info[block_name]
                            if self.preload_to_memory:
                                source_data[block_name] = np.zeros((main_days, h, w, c), dtype=np.float32)
                            else:
                                mmap_path = os.path.join(self.temp_dir, f"{source_name}_{block_name}_zeros.npy")
                                np.save(mmap_path, np.zeros((main_days, h, w, c), dtype=np.float32))
                                source_data[block_name] = np.load(mmap_path, mmap_mode='r')
                        else:
                            # 根据数据源类型设置默认通道数
                            c = 2 if source_name == 'flow' else 3
                            if self.preload_to_memory:
                                source_data[block_name] = np.zeros((main_days, 32, 32, c), dtype=np.float32)
                            else:
                                mmap_path = os.path.join(self.temp_dir, f"{source_name}_{block_name}_zeros.npy")
                                np.save(mmap_path, np.zeros((main_days, 32, 32, c), dtype=np.float32))
                                source_data[block_name] = np.load(mmap_path, mmap_mode='r')
            self.block_data[source_name] = source_data

    def compute_normalization_stats(self, norm_stats):
        """计算归一化参数（仅训练集计算）"""
        if norm_stats:
            self.norm_params = norm_stats
            return

        self.norm_params = {}

        # 仅训练集计算归一化参数
        if self.split != 'train':
            raise ValueError("归一化参数必须从训练集获取或外部提供")

        for source_name in self.source_names:
            if source_name == 'date_index':
                self.norm_params[source_name] = (0.0, 1.0)
                continue

            min_val = float('inf')
            max_val = float('-inf')

            for block_name in self.block_names:
                if block_name in self.block_data[source_name]:
                    block_data = self.block_data[source_name][block_name]
                    if len(block_data) > 0:
                        if block_data.ndim == 4:
                            block_min = np.min(block_data)
                            block_max = np.max(block_data)
                        else:
                            block_min = np.min(block_data)
                            block_max = np.max(block_data)
                        min_val = min(min_val, block_min)
                        max_val = max(max_val, block_max)

            if min_val == float('inf'):
                min_val = 0.0
                max_val = 1.0
            self.norm_params[source_name] = (min_val, max_val)

        if self.main_source in self.norm_params:
            min_val, max_val = self.norm_params[self.main_source]
            print(f"主数据源归一化: min={min_val:.4f}, max={max_val:.4f}")

    def create_sequences(self):
        """为数据集创建序列（按序列划分）"""
        # 第一步：收集所有可能的序列
        all_sequences = []
        for block_name in self.block_names:
            block_params = self.block_safe_params[block_name]
            safe_start = block_params['safe_start']
            safe_end = block_params['safe_end']

            for day in range(safe_start, safe_end, self.step):
                if day + self.total_steps <= block_params['total_days']:
                    all_sequences.append((block_name, day))

        # 第二步：随机打乱所有序列（使用固定随机种子）
        rng = random.Random(self.seed)
        rng.shuffle(all_sequences)

        # 第三步：按比例划分数据集
        total_sequences = len(all_sequences)
        train_end = int(total_sequences * 0.7)
        val_end = train_end + int(total_sequences * 0.15)

        if self.split == 'train':
            self.sequence_starts = all_sequences[:train_end]
        elif self.split == 'val':
            self.sequence_starts = all_sequences[train_end:val_end]
        else:  # test
            self.sequence_starts = all_sequences[val_end:]

        print(f"创建{self.split}数据集: {len(self.sequence_starts)}个序列")
        print(f" - 总序列数: {total_sequences}")
        print(f" - 训练集序列数: {train_end}")
        print(f" - 验证集序列数: {val_end - train_end}")
        print(f" - 测试集序列数: {total_sequences - val_end}")

    def __len__(self):
        return len(self.sequence_starts)

    def _process_data(self, data, source_name, block_name, ref_height, ref_width):
        """通用数据处理函数"""
        # 归一化
        min_val, max_val = self.norm_params[source_name]
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val) * 2 - 1

        # 调整维度顺序
        if data.ndim == 4:
            # 已经是 (T, C, H, W) 格式
            pass
        elif data.ndim == 3:
            # 单通道数据，添加通道维度
            data = data[:, np.newaxis, :, :]
        else:
            warnings.warn(f"未知数据维度: {data.shape} for source {source_name}, block {block_name}")
            if block_name in self.block_info:
                h, w, c = self.block_info[block_name]
            else:
                # 根据数据源类型设置默认通道数
                c = 2 if source_name == 'flow' else 3
                h, w = 32, 32
            data = np.zeros((len(data), c, h, w), dtype=np.float32)

        # 调整尺寸（如果需要）
        if data.shape[2] != ref_height or data.shape[3] != ref_width:
            resized_data = np.zeros((data.shape[0], data.shape[1], ref_height, ref_width), dtype=data.dtype)
            for t in range(data.shape[0]):
                for c in range(data.shape[1]):
                    resized_data[t, c] = cv2.resize(
                        data[t, c],
                        (ref_width, ref_height),
                        interpolation=cv2.INTER_LINEAR
                    )
            data = resized_data

        return data

    def __getitem__(self, idx):
        block_name, start_day = self.sequence_starts[idx]
        end_day = start_day + self.total_steps

        input_sequences = []
        output_dict = {}

        if block_name in self.block_info:
            ref_height, ref_width = self.block_info[block_name][:2]
        else:
            ref_height, ref_width = 32, 32

        # 处理所有输入源（包括flow）
        for source_name in self.source_names:
            if source_name in self.block_data and block_name in self.block_data[source_name]:
                # 输入序列
                input_data = self.block_data[source_name][block_name][start_day:start_day + self.input_steps]
                processed_data = self._process_data(input_data, source_name, block_name, ref_height, ref_width)
                input_sequences.append(processed_data)

                # 如果是输出目标（flow），则处理输出序列
                if source_name == 'flow':
                    output_data = self.block_data[source_name][block_name][start_day + self.input_steps:end_day]
                    processed_output = self._process_data(output_data, source_name, block_name, ref_height, ref_width)
                    output_dict['flow'] = torch.tensor(processed_output, dtype=torch.float32)
            else:
                warnings.warn(f"数据源 '{source_name}' 或区块 '{block_name}' 不存在，使用零填充")
                if block_name in self.block_info:
                    h, w, c = self.block_info[block_name]
                else:
                    # 根据数据源类型设置默认通道数
                    c = 2 if source_name == 'flow' else 3
                    h, w = 32, 32

                # 创建输入序列的零填充
                input_data = np.zeros((self.input_steps, h, w, c), dtype=np.float32)
                processed_data = self._process_data(input_data, source_name, block_name, ref_height, ref_width)
                input_sequences.append(processed_data)

                # 如果是输出目标（flow），则创建输出序列的零填充
                if source_name == 'flow':
                    output_data = np.zeros((self.output_steps, h, w, c), dtype=np.float32)
                    processed_output = self._process_data(output_data, source_name, block_name, ref_height, ref_width)
                    output_dict['flow'] = torch.tensor(processed_output, dtype=torch.float32)

        # 确保输出目标存在
        if 'flow' not in output_dict:
            warnings.warn(f"光流数据在区块 '{block_name}' 不存在，使用零填充")
            # 光流默认2通道
            output_data = np.zeros((self.output_steps, ref_height, ref_width, 2), dtype=np.float32)
            processed_output = self._process_data(output_data, 'flow', block_name, ref_height, ref_width)
            output_dict['flow'] = torch.tensor(processed_output, dtype=torch.float32)

        if len(input_sequences) == 0:
            raise RuntimeError(f"没有可用的输入数据源用于区块 {block_name}")

        input_sequence = np.concatenate(input_sequences, axis=1)
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)

        # 打印调试信息 - 仅在需要时启用
        # print(f"输入序列形状: {input_sequence.shape}")
        # print(f"输出目标形状: {output_dict['flow'].shape}")

        return input_sequence, output_dict

    def get_norm_stats(self):
        """返回归一化参数 {数据源: (min, max)}"""
        return self.norm_params

    def __del__(self):
        """清理资源"""
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                print(f"清理临时目录失败: {e}")

        # 清理内存映射文件
        if not self.preload_to_memory:
            for source_name, source_data in self.block_data.items():
                for block_name, block_data in source_data.items():
                    if isinstance(block_data, np.memmap) and hasattr(block_data, 'filename'):
                        try:
                            if os.path.exists(block_data.filename):
                                os.remove(block_data.filename)
                        except Exception as e:
                            print(f"删除内存映射文件失败: {e}")

        # 清理其他资源
        del self.block_data
        gc.collect()
