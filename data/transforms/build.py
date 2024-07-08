# 时序扭曲（Time Warping）：模拟信号在时间维度上的非线性变化。时序扭曲是一种模拟信号在时间轴上的非线性变化的技术。这种方法通过对时间序列进行轻微的拉伸或压缩，来模拟信号传输过程中可能出现的速度变化。例如，由于环境因素的影响，信号的某些部分可能会比其他部分传播得更快或更慢。通过在训练数据中引入这种类型的变化，模型可以学会识别和处理时间轴上的微小变异，从而提高其对时间扭曲的鲁棒性。
# 噪声注入（Noise Injection）：向信号中添加随机噪声，以模拟环境干扰。噪声注入是向信号中添加随机噪声的过程，以模拟环境干扰、设备噪声或信号传输中的误差。这种增强方法可以帮助模型学习在存在背景噪声或其他形式的干扰时，仍然准确地识别和处理信号。通过训练模型以识别在噪声背景下的信号，可以显著提高其在实际应用中的鲁棒性和性能。
# 随机遮挡（Random Masking）：在信号的某些部分随机应用遮挡，模拟信号丢失情况。随机遮挡是在信号的随机部分中引入遮挡（例如，将信号值设置为零或其他中性值），以模拟信号丢失或传输错误的情况。这种方法可以增加模型对信号中缺失部分的鲁棒性，使其能够在部分数据缺失的情况下仍然进行有效的信号分析和识别。通过学习在不完整的数据上工作，模型可以更好地处理现实世界中的信号损失问题。
# 这些数据增强技术的共同目标是提高模型对各种真实世界情况的适应性，包括信号变化、干扰、噪声和数据缺失等。通过在训练过程中引入这些变化，可以显著提升模型的泛化能力和鲁棒性。

import numpy as np
import torch
import random

class TimeWarping:
    def __init__(self, max_warping_ratio=0.05, prob=0.5):
        self.max_warping_ratio = max_warping_ratio
        self.prob = prob

    def __call__(self, signal):
        if random.random() < self.prob:
            time_steps = np.arange(signal.shape[0])
            warping_steps = time_steps + np.random.uniform(-self.max_warping_ratio, self.max_warping_ratio, size=time_steps.shape) * signal.shape[0]
            warping_steps = np.clip(warping_steps, 0, signal.shape[0] - 1).astype(np.int32)
            warped_signal = signal[warping_steps]
            return warped_signal
        else:
            return signal

class NoiseInjection:
    def __init__(self, noise_level=0.02, prob=0.5):
        self.noise_level = noise_level
        self.prob = prob

    def __call__(self, signal):
        if random.random() < self.prob:
            noise = np.random.normal(0, self.noise_level, signal.shape)
            return signal + noise
        else:
            return signal

class RandomMasking:
    def __init__(self, masking_ratio=0.1, prob=0.5):
        self.masking_ratio = masking_ratio
        self.prob = prob

    def __call__(self, signal):
        if random.random() < self.prob:
            num_masks = int(signal.shape[0] * self.masking_ratio)
            mask_indices = np.random.choice(signal.shape[0], num_masks, replace=False)
            signal[mask_indices] = 0  # Assuming zero is a neutral value for masking
            return signal
        else:
            return signal

def build_transforms(cfg, is_train=True):
    transforms = []

    if is_train:
        # Example of adding some transforms based on configuration
        # if cfg.TIME_WARPING.ENABLED:
        #     transforms.append(TimeWarping(max_warping_ratio=cfg.TIME_WARPING.RATIO, prob=cfg.TIME_WARPING.PROB))
        # if cfg.NOISE_INJECTION.ENABLED:
        #     transforms.append(NoiseInjection(noise_level=cfg.NOISE_INJECTION.LEVEL, prob=cfg.NOISE_INJECTION.PROB))
        # if cfg.RANDOM_MASKING.ENABLED:
        #     transforms.append(RandomMasking(masking_ratio=cfg.RANDOM_MASKING.RATIO, prob=cfg.RANDOM_MASKING.PROB))
        transforms.append(TimeWarping())
        transforms.append(NoiseInjection())
        transforms.append(RandomMasking())
        pass
            
    # Convert to tensor at the end
    transforms.append(lambda x: torch.tensor(x, dtype=torch.float32))

    def apply_transforms(signal):
        for transform in transforms:
            signal = transform(signal)
        return signal

    return apply_transforms
