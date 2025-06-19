# =========== #
#             #
#   Импорты   #
#             #
# =========== #

# Базовые
import os
import gc
import sys
import glob
import json
import subprocess
import hashlib
import shutil
import math
import copy

# Прогрессбар
from tqdm import tqdm

# Логирование
import logging

# Рандомизация
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

# Типизация
from typing import List, Union, Dict, Any, cast
from pathlib import Path

# Работа с URL
from urllib.parse import urlparse

# Numpy
import numpy as np

# OpenCV
import cv2

# Torchvision
from torchvision import transforms

# PyTorch
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Dataset, Sampler 

# Оптимизатор / Accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs

# Diffusers - планировщики
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

# Diffusers - оптимизаторы
from diffusers.optimization import get_scheduler

# Diffusers - энкодеры
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

# Diffusers - трансформеры
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

# Diffusers - эмдеддинги
from diffusers.models.embeddings import apply_rotary_emb

from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline

# Diffusers - утилиты
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.loading_utils import load_image
from diffusers.utils.import_utils import is_torch_npu_available, is_torch_version
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.training_utils import cast_training_params
from diffusers.configuration_utils import register_to_config

# Diffusers - обработчики изображений
from diffusers.image_processor import VaeImageProcessor

# Diffusers - квантование
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

# scikit-learn
from sklearn.model_selection import train_test_split

# Трансформеры
from transformers import CLIPTokenizer, T5TokenizerFast
from transformers import CLIPTextModel, T5EncoderModel

# Трансформеры - конфигурации
from transformers.configuration_utils import PretrainedConfig

# PEFT
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

# TensorRT
import tensorrt.tensorrt as trt

# TensorBoard
import tensorboard

# ============== #
#                #
#   Константны   #
#                #
# ============== #

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8":  torch.float8_e4m3fn,
}

LORA_LAYERS = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

# =========== #
#             #
#  Настройки  #
#             #
# =========== #

# Общие настройки
PRETRAINED_MODEL_NAME           = "black-forest-labs/FLUX.1-Fill-dev"     # Название модели
SEED                            = 41                                      # Сид для генерации рандомных чисел. Значения: [10..99]
VAL_SEED                        = 42                                      # Сид для генерации рандомных чисел для валидации. Значения: [10..99]

# Работа с данными и кэширование
TRAIN_DATA_DIR                  = "dataset/train"                         # Папка с изображениями для обучающего набора данных
VALIDATION_RATIO                = 0.1                                     # Сколько % данных изъять для валидационного набора
RECREATE_CACHE                  = True                                    # Пересоздать все кэши датасета. Не забудьте сменить сид
METADATA_PREFIX                 = "flux_fill"                             # Префикс файлов с метаинформацией
MODEL_CACHE_DIR                 = "model_cache"                           # Кэш для моделей, куда они будут загружены
NUM_WORKERS                     = (os.cpu_count() or 2)-1                 # Количество потоков загрузки датасета

# Параметры изображения
WIDTH                           = 1280                                    # Ширина изображений
HEIGHT                          = 720                                     # Высота изображений

# Настройки обучения
LEARNING_RATE                   = 1e-4                                    # Начальная скорость обучения (после возможного периода разогрева), которая будет использоваться   
REPEATS                         = 2                                       # Количество повторов датасета в рамках одной эпохи
NUM_TRAIN_EPOCHS                = 5                                       # Количество тренировочных эпох
TRAIN_BATCH_SIZE                = 2                                       # Размер пакета (на одно устройство) для загрузчика обучающих данных
GRAD_ACCUMULATION_STEPS         = 1                                       # Количество шагов обновления, которые необходимо накопить перед выполнением обратного прохода и обновления параметров
GRADIENT_CHECKPOINTING          = True                                    # Использовать ли градиентное контрольное сохранение для экономии памяти за счёт замедления обратного прохода
MIXED_PRECISION                 = "bf16"                                  # Использовать ли смешанную точность. Выберите между fp16 и bf16 (bfloat16).   
ALLOW_TF32                      = True                                    # Разрешить использование TF32 на видеокартах Ampere. Может быть использовано для ускорения обучения

# Оптимизатор
OPTIMIZER_NAME                  = "adamw"                                 # Тип оптимизатора, который следует использовать. Выберите из ["adamw", "prodigy"]
USE_8BIT_ADAM                   = True                                    # Использовать ли 8-битный Adam из библиотеки bitsandbytes. Игнорируется, если оптимизатор не установлен как AdamW
ADAM_BETA1                      = 0.9                                     # Параметр β₁ для оптимизаторов
ADAM_BETA2                      = 0.999                                   # Параметр β₂ для оптимизаторов
ADAM_EPSILON                    = 1e-08                                   # Значение ε для оптимизаторов Adam и Prodigy
ADAM_WEIGHT_DECAY               = 1e-02                                   # Коэффициент регуляризации (weight decay) для параметров U-Net
ADAM_WEIGHT_DECAY_TEXT_ENCODER  = 1e-03                                   # Коэффициент регуляризации (weight decay) для параметров текстового энкодера. 
PRODIGY_BETA3                   = None                                    # Коэфф-ты для вычисл. шага обучения Prodigy с использ-ем скользящих средних. Если значение None, используется квадратный корень из β₂. Игнорируется, если оптимизатор установлен AdamW.
PRODIGY_DECOUPLE                = True                                    # Использовать ли стиль AdamW с раздельным уменьшением веса
PRODIGY_USE_BIAS_CORRECTION     = True                                    # Включить коррекцию смещения в Adam. Игнорируется, если оптимизатор не AdamW
PRODIGY_SAFEGUARD_WARMUP        = True                                    # Удалить ли скорость обучения из знаменателя оценки D, чтобы избежать проблем на этапе разогрева? Игнорируется, если оптимизатор не AdamW
PRODIGY_D_COEF                  = 2                                       # Размерность матриц обновления LoRA. Игнорируется, если оптимизатор не AdamW
MAX_GRAD_NORM                   = 1.0                                     # Максимальная норма градиента

# Планировщик скорости обучения
LR_SCHEDULER                    = "cosine_with_restarts"                  # Тип планировщика, который следует использовать. Выберите один из следующих вариантов: [linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup]
LR_WARMUP_STEPS                 = 1000                                    # Количество шагов разогрева в планировщике скорости обучения
COSINE_RESTARTS                 = 2                                       # Для планировщика LR_SCHEDULER cosine_with_restarts. Задает количество рестартов
MAX_TIME_STEPS                  = 0                                       # Ограничение максимального количества временных шагов. Обучение будет ограничено значением от 0 до max_time_steps. 0 - неограниченно
WEIGHTING_SCHEME                = "logit_normal"                          # Функция для определения весов потерь на разных временных шагах обучения. Значения: ["sigma_sqrt", "logit_normal", "mode", "cosmap", "logit_snr"]
LOGIT_MEAN                      = 0.0                                     # Среднее значение (mean) для схемы взвешивания 'logit_normal' определяет центральную тенденцию логит-нормального распределения
LOGIT_STD                       = 1.0                                     # Стандартное отклонение (std) для схемы взвешивания 'logit_normal' определяет степень рассеивания весов вокруг среднего значения
MODE_SCALE                      = 1.29                                    # Параметр scale в схеме взвешивания 'mode' определяет ширину области, в которой веса достигают своего максимального значения

# Параметры LoRA
LORA_RANK                       = 64                                      # Размерность матриц обновления LoRA
LORA_ALPHA                      = 64                                      # Масштабный коэффициент, управляющий влиянием адаптационных весов LoRA

# Регулировка потерь и шум
USE_DEBIAS                      = True                                    # Использовать ли функцию потерь с коррекцией смещения (debiased estimation loss)
SNR_GAMMA                       = 5                                       # Коэффициент γ для взвешивания по отношению сигнал/шум (SNR), используемый при ребалансировке функции потерь.
CAPTION_DROPOUT                 = 0.1                                     # Вероятность удаления подписи и обновления безусловного пространства
MASK_DROPOUT                    = 0.01                                    # Вероятность замены маски на нули
GUIDANCE_SCALE                  = 1                                       # Параметр управляет уровнем применения дистиллированного руководства (guidance distillation) при генерации изображений
NOISE_OFFSET                    = 0.01                                    # Смещение шума в исходном шуме (offset noise)

# Сохранение и контрольные точки
OUTPUT_DIR                      = "train"                                 # Каталог вывода, в который будут записаны предсказания модели и контрольные точки
SAVE_NAME                       = "flux_fill_"                            # Префикс имени для сохранения контрольных точек
SAVE_MODEL_EPOCHS               = 0                                       # Сохранять модель каждые N эпох
SAVE_MODEL_STEPS                = 250                                     # Сохранять модель каждые N шагов
RESUME_FROM_CHECKPOINT          = "latest"                                # Следует ли возобновлять обучение с предыдущей контрольной точки  
SKIP_EPOCH                      = 0                                       # Пропустить валидацию и только сохранять модель первые N эпох
SKIP_STEP                       = 0                                       # Пропустить валидацию и только сохранять модель первые N шагов

# Валидация
VALIDATION_EPOCHS               = 1                                       # Запускать валидацию каждые X эпох   

# Логирование и отчёты
LOGGING_DIR                     = "logs"                                  # Каталог логов для TensorBoard
REPORT_TO                       = "tensorboard"                           # Тип интеграции для отчётности результатов и логов. Поддерживаемые платформы: "tensorboard" 

# Регуляризация в трансферном обучении
REG_RATIO                       = 0.7                                     # Параметр регуляризации в контексте трансферного обучения служит для контроля степени отклонения модели от исходной задачи при обучении на новой.
REG_TIMESTEP                    = 900                                     # Определяет временной шаг, начиная с которого применяется регуляризация в процессе трансферного обучения

# Использование SageAttention
USE_SAGE_ATT                    = True                                   # Использование SageAttention

# Отдельная модель для трансформера
MODEL_PATH = r"https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/blob/main/flux1-fill-dev-Q6_K.gguf"

# ========== #
#            #
#   Классы   #
#            #
# ========== #
    
class CachedMaskedPairsDataset(Dataset):
    """
    Класс Dataset для загрузки предварительно закэшированных эмбеддингов текста и латентных представлений изображений.
    Также поддерживает выборочную подмену элементов с использованием списка "leftover_indices".

    Аргументы:
        datarows (list): Список словарей, содержащих пути к кэшированным данным.
        conditional_dropout_percent (float): Процент вероятности условного дропаута (в данной реализации пока не используется напрямую).
        has_redux (bool): Флаг, указывающий, содержат ли данные дополнительное поле 'redux_image'.
    """
    def __init__(self, datarows,conditional_dropout_percent=0.1, has_redux=False): 
        """
        Инициализация датасета CachedMaskedPairsDataset.

        Аргументы:
            datarows (list): Список словарей, содержащих пути к кэшированным эмбеддингам текста и латентным изображениям.
                            Каждый элемент должен включать как минимум:
                            - 'npz_path' (путь к файлу с текстовыми эмбеддингами)
                            - пути к латентным изображениям в подсловарах: ground_true, factual_image и т.д.
            
            conditional_dropout_percent (float): Процент вероятности дропаута (временное выключение или исключение информации),
                                                может быть использован для регулировки обучающей регуляризации. 
                                                В текущей реализации не используется, но может быть полезен в будущем.

            has_redux (bool): Флаг, указывающий, включать ли дополнительное поле 'redux_image' в выходной словарь.
                            Если True, датасет будет ожидать и загружать redux-представление.
        """
        # Устанавливаем флаг наличия redux-изображений
        self.has_redux = has_redux
        # Сохраняем список всех элементов датасета (метаданных)
        self.datarows = datarows
        # Список "остаточных" индексов, которые будут обработаны раньше обычных
        self.leftover_indices = []
        # Параметр для условного дропаута
        self.conditional_dropout_percent = conditional_dropout_percent
        
    def __len__(self):
        """
        Возвращает общее число записей в датасете.
        """
        return len(self.datarows)

    def __getitem__(self, index):
        """
        Возвращает один элемент датасета по индексу.
        
        Логика:
          1) Если в leftover_indices есть элементы, берём первый из них вместо index.
          2) Загружаем из .npz-файла эмбеддинги текста: prompt_embed, pooled_prompt_embed, txt_attention_mask.
          3) Формируем начальный словарь result с текстовыми эмбеддингами.
          4) Определяем список необходимых латентов (ground_true, factual_image, ...).
             При has_redux=True добавляется redux_image.
          5) Для каждого латента проверяем наличие ключа 'latent_path' в metadata
             и загружаем его через torch.load.
          6) Добавляем загруженные тензоры в result и возвращаем его.
        """
        # Выбор реального индекса: из leftover_indices или по порядку
        if self.leftover_indices:
            actual_index = self.leftover_indices.pop(0)
        else:
            actual_index = index

        # Словарь с путями к файлам
        metadata = self.datarows[actual_index] 

        # Загрузка текстовых эмбеддингов из .npz
        cached_npz = torch.load(metadata['npz_path'], weights_only=True)
        prompt_embed = cached_npz['prompt_embed']
        pooled_prompt_embed = cached_npz['pooled_prompt_embed']
        txt_attention_mask = cached_npz['txt_attention_mask']
        
        # Собираем базовый результат
        result = {
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embed,
            "txt_attention_mask": txt_attention_mask
        }

        # Формируем список названий латентных представлений
        cached_latent_names = [
            "ground_true",
            "factual_image",
            "factual_image_mask",
            "factual_image_masked_image"
        
        ]

         # Опциональное поле с «редуцированным» изображением
        if self.has_redux:
            cached_latent_names.append("redux_image")
            
         # Цикл по всем латентам: проверка и загрузка
        for cached_latent_name in cached_latent_names:
            if not "latent_path" in metadata[cached_latent_name]:
                raise ValueError(f"{cached_latent_name} is not in metadata")
            cached_latent = torch.load(metadata[cached_latent_name]['latent_path'], weights_only=True)
            result[cached_latent_name] = cached_latent
        return result
    
class BucketBatchSampler(Sampler):
    """
    Специальный сэмплер для DataLoader, который группирует элементы датасета по "бакетам" (bucket'ам)
    на основе отношения сторон изображений или других критериев, указанных в метаинформации (ключ 'bucket').
    
    Основная идея: элементы с похожим форм-фактором объединяются в бакеты, а затем в батчи одинакового размера.
    Это помогает уменьшить паддинг и ускоряет обучение, особенно при работе с изображениями или эмбеддингами.
    
    Если `drop_last=True`, батчи, не достигшие полного размера, игнорируются.
    Если `drop_last=False`, такие "остаточные" батчи обрабатываются, либо сохраняются для следующей эпохи.
    """
    def __init__(self, dataset, batch_size, drop_last=True):
        """
        Инициализация BucketBatchSampler.

        Параметры:
        - dataset: объект датасета, содержащий список `datarows` с метаинформацией (включая ключ 'bucket').
        - batch_size: размер батча (количество элементов в одной выборке).
        - drop_last: если True, неполные батчи (меньше batch_size) будут отброшены;
                    если False — будут сохранены/обработаны отдельно.
        """
        # Сохраняем датасет
        self.dataset = dataset
        # Извлекаем список метаинформации по всем элементам
        self.datarows = dataset.datarows
        # Устанавливаем размер батча
        self.batch_size = batch_size
        # Нужно ли отбрасывать неполные батчи
        self.drop_last = drop_last
        # Список "остатков" (элементов, не попавших в батчи в предыдущих эпохах)
        self.leftover_items = []
        # Создаем индексные бакеты по ключу 'bucket' (например, по соотношению сторон изображений)
        self.bucket_indices = self._bucket_indices_by_aspect_ratio() 

    def _bucket_indices_by_aspect_ratio(self):
        """
        Группирует индексы элементов по значению ключа 'bucket' в метаинформации.
        Например, изображения с одинаковым соотношением сторон будут в одном бакете.
        """
        # Словарь для хранения бакетов: ключ — bucket ID, значение — список индексов
        buckets = {}

        # Проходим по всем элементам датасета
        for idx in range(len(self.datarows)):
            # Получаем значение ключа 'bucket'
            closest_bucket_key = self.datarows[idx]['bucket']
            # Если такого бакета еще нет — создаём
            if closest_bucket_key not in buckets:
                buckets[closest_bucket_key] = []
            # Добавляем текущий индекс в соответствующий бакет
            buckets[closest_bucket_key].append(idx) #adds item to bucket

        # Перемешиваем элементы внутри каждого бакета для случайности при обучении
        for bucket in buckets.values():
            random.shuffle(bucket)

        # Возвращаем словарь бакетов с индексами    
        return buckets

    def __iter__(self):
        """
        Делает объект итерируемым: на каждой итерации возвращается батч из индексов одного бакета.
        """
        # Перегенерируем бакеты на случай обновления датасета или добавления остатков
        self.bucket_indices = self._bucket_indices_by_aspect_ratio()

        # Переносим остатки из предыдущей эпохи обратно в бакеты
        if self.leftover_items:
            for leftover_idx in self.leftover_items:
                closest_bucket_key = self.datarows[leftover_idx]['bucket']
                if closest_bucket_key in self.bucket_indices:
                    # Вставляем в начало, чтобы обработать первыми
                    self.bucket_indices[closest_bucket_key].insert(0, leftover_idx)
                else:
                    # Создаем новый бакет, если его ещё не было
                    self.bucket_indices[closest_bucket_key] = [leftover_idx]
            # Очищаем список остатков после переноса
            self.leftover_items = [] 
        
        # Получаем список и перемешиваем порядок самих бакетов
        all_buckets = list(self.bucket_indices.items())
        random.shuffle(all_buckets)

        # Формируем батчи по очереди из каждого бакета
        for _, bucket_indices in all_buckets: #iterate each bucket
            batch = []
            for idx in bucket_indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch     # Если набрали нужный размер — выдаем батч
                    batch = []      # Начинаем следующий
            # Обработка оставшихся элементов, если они есть после полного обхода бакета
            if not self.drop_last and batch:
                yield batch  # Отдаем даже неполный батч
            # Если неполные батчи запрещены — сохраняем для следующей эпохи
            elif batch:
                self.leftover_items.extend(batch)  

    def __len__(self):
        """
        Возвращает количество батчей на эпоху. Учитывает drop_last и возможные остатки.
        """
        # Вычисляем общее кол-во батчей
        total_batches = sum(len(indices) // self.batch_size for indices in self.bucket_indices.values())
        # Если есть неполные батчи
        if not self.drop_last:
            leftovers = sum(len(indices) % self.batch_size for indices in self.bucket_indices.values())
            total_batches += bool(leftovers)
        return total_batches

# ========= #
#           #
#  Датасет  #
#           #
# ========= #
def find_index_from_right(lst, value):
    """
    Ищет индекс последнего вхождения строки `value` в список строк `lst`.

    Алгоритм:
    1. Разворачивает список и строку, чтобы воспользоваться стандартным методом .index(), 
       который ищет только первое вхождение.
    2. Находит позицию первого вхождения перевёрнутого значения в перевёрнутом списке.
    3. Преобразует этот индекс обратно в индекс для исходного списка.

    Параметры:
        lst (list[str]): Список строк, в котором ищем значение.
        value (str): Строка, индекс последнего вхождения которой нужно найти.

    Возвращает:
        int: Индекс последнего вхождения `value` в `lst`. 
             Если `value` не найдено, возвращает -1.
    """
    try:
        reversed_index = lst[::-1].index(value[::-1])
        return len(lst) - 1 - reversed_index
    except:
        return -1

def collate_fn(examples):
    """
    Функция объединения (collate_fn) для DataLoader.

    Принимает список отдельных элементов выборки (примеров) и формирует из них один батч.
    Используется для объединения тензоров из разных примеров в батчевые тензоры.

    Предполагается, что каждый пример представляет собой словарь, содержащий текстовые эмбеддинги,
    маски внимания, а также несколько изображений в виде латентных представлений.

    ВАЖНО: потенциально чувствительна к перемешиванию разных соотношений сторон (aspect ratios),
    если такие данные представлены в батче. Предполагается, что BucketBatchSampler решает эту проблему.
    """    
    # Объединение текстовых эмбеддингов и масок внимания в батчевые тензоры
    prompt_embeds = torch.stack([example["prompt_embed"] for example in examples])
    pooled_prompt_embeds = torch.stack([example["pooled_prompt_embed"] for example in examples])
    txt_attention_masks = torch.stack([example["txt_attention_mask"] for example in examples])

    # Инициализация словаря с батчевыми данными    
    sample = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "txt_attention_masks": txt_attention_masks,
    }
        
    # Обработка изображений — загружаются латентные представления
    image_classes = [
        "ground_true",                  # Оригинальное (истинное) изображение
        "factual_image",                # Фактическое изображение
        "factual_image_mask",           # Маска фактического изображения
        "factual_image_masked_image"    # Маскированное изображение
    ]

     # Собираем латентные тензоры по каждому классу изображений и объединяем в батч
    for image_class in image_classes:
        sample[image_class] = torch.stack([example[image_class]["latent"] for example in examples])

    return sample
    
def create_empty_embedding(tokenizers,text_encoders,cache_path="cache/empty_embedding.npflux",recreate=False):
    """
    Создаёт и сохраняет в кэш «пустой» текстовый и маскирующий эмбеддинги.

    Если в кэше уже есть файл с заранее сгенерированными эмбеддингами и
    recreate=False, функция загружает их и возвращает. В противном случае
    она генерирует эмбеддинги пустой строки, сохраняет их на диск и возвращает.

    Args:
        tokenizers (list): Список токенизаторов (например, CLIP и T5).
        text_encoders (list): Список текстовых энкодеров, соответствующих токенизаторам.
        cache_path (str): Путь к файлу кэша эмбеддингов.
        recreate (bool): Если True — принудительно пересоздаёт кэш, удаляя старый файл.
        resolution (int): Разрешение (число шагов временного кода), используется внутренняя логика генерации.

    Returns:
        dict: Словарь с ключами:
            - "prompt_embed" (Tensor): эмбеддинг токенизированной пустой строки.
            - "pooled_prompt_embed" (Tensor): агрегированный эмбеддинг пустой строки.
            - "txt_attention_mask" (Tensor): маска внимания для пустой строки.
    """

    # Пересоздание кэша
    if recreate:
        os.remove(cache_path)

    # Загрузка существующего
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    # Кодируем промпт
    prompt_embeds, pooled_prompt_embeds, txt_attention_masks = encode_prompt(text_encoders, tokenizers, "")

    # Морозим эмбеддинги
    prompt_embed = prompt_embeds.squeeze(0)
    pooled_prompt_embed = pooled_prompt_embeds.squeeze(0)
    txt_attention_mask = txt_attention_masks.squeeze(0)

    # Получаем латенты
    latent = {
        "prompt_embed": prompt_embed.cpu(), 
        "pooled_prompt_embed": pooled_prompt_embed.cpu(),
        "txt_attention_mask": txt_attention_mask.cpu(),
    }
    
    # Сохраняем в кэш
    torch.save(latent, cache_path)
    return latent

# ========= #
#           #
#  Модель   #
#           #
# ========= #

def import_model_class_from_model_name_or_path(
    model_name_or_path: str, 
    subfolder: str = "text_encoder"
):
    """
    Возвращает класс текстового энкодера по имени модели или пути к сохранённой модели.

    Функция выполняет следующие шаги:
    1. Загружает конфигурацию энкодера из локальной директории или Hugging Face Hub с помощью
       `PretrainedConfig.from_pretrained(...)`.  
    2. Определяет строковое имя класса в поле `architectures` конфигурации.  
    3. Импортирует и возвращает реальный класс `transformers.CLIPTextModel` или `transformers.T5EncoderModel`.
       Если архитектура не поддерживается, выбрасывает `ValueError`.  

    Args:
        model_name_or_path (str): Название модели в Hugging Face Hub или путь к её директории.  
        subfolder (str, optional): Подкаталог внутри модели, где лежит конфигурация энкодера.
                                  По умолчанию `"text_encoder"`.  

    Returns:
        type: Класс энкодера из `transformers`, который можно использовать для создания экземпляра модели.

    Raises:
        ValueError: Если в конфиге найдена неподдерживаемая архитектура.
    """
    
    # Загружаем конфигурацию модели
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        cache_dir=MODEL_CACHE_DIR,
        subfolder=subfolder
    )

    # Получаем архитектуру токенайзера
    model_class = text_encoder_config.architectures[0]

    # Выбираем модель под токенайзер
    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def load_text_encoders(class_one, class_two):
    """
    Загружает два текстовых энкодера из предобученной модели.

    Функция выполняет следующие шаги:
    1. Использует метод `from_pretrained` первого класса энкодера (`class_one`),
       загружая его из директории или с Hugging Face Hub по пути `args.pretrained_model_name_or_path`
       из подкаталога "text_encoder".
    2. Повторяет тот же процесс для второго класса энкодера (`class_two`),
       но из подкаталога "text_encoder_2".
    3. Возвращает оба экземпляра энкодеров для дальнейшего использования.

    Args:
        class_one (type): Класс текстового энкодера (например, CLIPTextModel),
                          у которого будет вызван `from_pretrained`.
        class_two (type): Вторая версия текстового энкодера (например, T5EncoderModel),
                          у которого будет вызван `from_pretrained`.

    Returns:
        tuple: Кортеж из двух объектов энкодеров
               `(text_encoder_one, text_encoder_two)`.
    """
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        subfolder="text_encoder"
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
        cache_dir=MODEL_CACHE_DIR,
        subfolder="text_encoder_2"
    )

    return text_encoder_one, text_encoder_two

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: Union[str, List[str]],
    max_sequence_length=512,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    """
    Генерирует эмбеддинги текстового описания для модели заполнительного заполнения (fill-in).

    Функция обрабатывает входную строку prompt двумя энкодерами:
      1) CLIPTextModel для получения «пулед» (pooled) эмбеддингов всего описания.
      2) T5EncoderModel для получения пошаговых эмбеддингов (prompt_embeds) и масок внимания.

    Args:
        text_encoders (List[nn.Module]):
            Список из двух текстовых энкодеров:
              - text_encoders[0]: CLIPTextModel
              - text_encoders[1]: T5EncoderModel
        tokenizers (List[PreTrainedTokenizer]):
            Соответствующие токенизаторы:
              - tokenizers[0]: CLIPTokenizer
              - tokenizers[1]: T5TokenizerFast
        prompt (str или List[str]):
            Строка или список строк с описаниями, по которым нужно получить эмбеддинги.
        max_sequence_length (int, optional):
            Максимальная длина последовательности токенов для T5-энкодера. По умолчанию 512.
        device (torch.device или None, optional):
            Устройство, на котором выполнять энкодинг. Если None, берётся из первого энкодера.
        num_images_per_prompt (int, optional):
            Сколько вариантов масок/эмбеддингов генерировать на каждое описание. Например, для батча.
        text_input_ids_list (List[Tensor], optional):
            Если заранее вычислены ID токенов, можно передать их сюда:
              - text_input_ids_list[0] для CLIP
              - text_input_ids_list[1] для T5

    Returns:
        Tuple[
            torch.Tensor,  # prompt_embeds: тензор пошаговых эмбеддингов от T5 [B, L, D]
            torch.Tensor,  # pooled_prompt_embeds: «пулед» эмбеддинг от CLIP [B, D]
            torch.Tensor   # txt_attention_masks: маски внимания от T5 [B, L]
        ]
    """
    # Конвертация строк по типу
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Настройка CLIP
    clip_tokenizer = tokenizers[0]
    clip_text_encoder = text_encoders[0]

    # Кодировка в CLIP
    pooled_prompt_embeds = encode_prompt_with_clip(
        text_encoder=clip_text_encoder,
        tokenizer=clip_tokenizer,
        prompt=prompt,
        device=device if device is not None else clip_text_encoder.device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    # Кодировка в T5
    prompt_embeds, txt_attention_masks = encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    return prompt_embeds, pooled_prompt_embeds, txt_attention_masks

def encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    """
    Преобразует текстовый запрос в эмбеддинги с помощью CLIPTextModel.

    Функция выполняет следующие шаги:
      1. Принимает строку или список строк `prompt` и при необходимости упаковывает в список.
      2. Токенизирует текст командой CLIPTokenizer (padding до max_length=77, обрезка).
      3. Прогоняет тензор input_ids через CLIPTextModel, получая скрытые состояния.
      4. Извлекает pooled-выход (`pooler_output`) — глобальный вектор представления всего запроса.
      5. Приводит эмбеддинги к тому же dtype и устройству, что у модели.
      6. Дублирует каждое эмбеддинг-представление столько раз, сколько нужно изображений на запрос.
      7. Формирует итоговый тензор размера `[batch_size * num_images_per_prompt, embed_dim]`.

    Args:
        text_encoder (CLIPTextModel):
            Модель для преобразования токенов в векторы.
        tokenizer (CLIPTokenizer или None):
            Токенизатор для преобразования текста в input_ids.
            Если None, вместо токенизации должны быть переданы готовые text_input_ids.
        prompt (str или List[str]):
            Описание (или список описаний) для генерации эмбеддингов.
        device (torch.device, optional):
            Устройство для вычислений. Если None — используется устройство модели.
        text_input_ids (torch.Tensor, optional):
            Предварительно вычисленные ID токенов. Используется, если tokenizer=None.
        num_images_per_prompt (int):
            Сколько копий эмбеддингов сделать для каждого текста (например, когда на один текст генерируется несколько изображений).

    Returns:
        torch.Tensor:
            Эмбеддинги текста формы `[batch_size * num_images_per_prompt, embedding_dim]`.
    """
    # Конвертация строки по типу 
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Получение количества токенов
    batch_size = len(prompt)

    # Настройка токенайзера
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    # Получаем эмбеддинги
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    # Использовать pooled-выход (pooler_output) модели CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(device=device, dtype=text_encoder.dtype)

    # Дублируем эмбеддинги для каждого изображения на запрос
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds 

def encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt: Union[str, List[str]]="",
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    """
    Генерирует пошаговые эмбеддинги текста и возвращает их вместе с маской внимания.

    Функция принимает либо строку, либо список строк `prompt`, токенизирует их
    (если передан `tokenizer`), пропускает через T5-энкодер и возвращает:
      - `prompt_embeds`: тензор размерности
         [batch_size * num_images_per_prompt, seq_len, hidden_dim]
      - `txt_attention_mask`: соответствующую маску внимания
         [batch_size, seq_len]

    Args:
        text_encoder (T5EncoderModel):
            Модель для получения эмбеддингов текста.
        tokenizer (T5TokenizerFast или None):
            Токенизатор для преобразования текста в input_ids.
            Если None, `text_input_ids` должен быть передан вручную.
        max_sequence_length (int):
            Максимальная длина последовательности токенов для токенизации.
        prompt (str или List[str], optional):
            Один текст или список текстов для кодирования.
        num_images_per_prompt (int, optional):
            Количество копий эмбеддингов на каждую подсказку
            (например, когда генерируется несколько изображений
            на одну и ту же подпись).
        device (torch.device, optional):
            Устройство для вычислений. Если None —
            используется текущее устройство модели.
        text_input_ids (torch.Tensor, optional):
            Если токенизатор не передан, сюда можно подать
            заранее подготовленные ID токенов.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            prompt_embeds — эмбеддинги текста,
            txt_attention_mask — маска внимания для текста.
    """
    # Конвертация строки по типу 
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Получение количества токенов
    batch_size = len(prompt)

    # Настройка токенайзера
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        txt_attention_mask = text_inputs.attention_mask
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    # Получение эмбеддингов
    text_input_ids = text_input_ids.to(device)
    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # Получение пула
    _, seq_len, _ = prompt_embeds.shape

    # Дублируем эмбеддинги для каждого изображения на запрос
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, txt_attention_mask

@torch.no_grad()
def create_embedding(
    tokenizers,
    text_encoders,
    folder_path: str,
    file: str,
    resolutions: tuple[int, int],
    cache_ext: str = ".npflux",
    recreate_cache: bool = False,
    pipe_prior_redux = None,
    exist_npz_path: str = "",
    redux_image_path: str = ""
) -> dict:
    """
    Генерирует или загружает из кэша «пустые» текстовые эмбеддинги для одного изображения.

    Аргументы:
        tokenizers (list):  список токенизаторов [clip_tokenizer, t5_tokenizer]
        text_encoders (list): список текстовых энкодеров [clip_encoder, t5_encoder]
        folder_path (str):   путь к папке с изображением и (опционально) .txt-файлом
        file (str):          имя файла изображения внутри folder_path
        cache_ext (str):     расширение кэша (по умолчанию ".npflux")
        resolutions (tuple[int, int]):  список разрешений (используется для каких-то downstream-целей), по умолчанию [1024]
        recreate_cache (bool): если True — принудительно пересоздать кэш даже при наличии файла
        pipe_prior_redux:    опциональный pipeline для дополнительной обработки эмбеддингов
        exist_npz_path (str): путь к уже существующему .npflux-файлу (если хотим перезаписать другой кэш)
        redux_image_path (str): путь к альтернативному изображению (для pipe_prior_redux)

    Возвращает:
        dict: словарь с полями
            - image_path, image_path_sha, folder_path, file, resolutions  
            - text_path (если .txt найден), text_path_sha  
            - npz_path, npz_path_sha  
    """
    # Собираем базовые метаданные
    filename, _ = os.path.splitext(file)
    image_path = os.path.join(folder_path, file)
    image_path_sha = compute_file_sha256(image_path)

    json_obj = {
        "image_path": image_path,
        "image_path_sha": image_path_sha,
        "folder_path": folder_path,
        "file": file,
        "resolutions": resolutions
    }

    # При наличии текстового файла рядом с изображением считываем его
    caption_ext = ".txt"
    text_path = os.path.join(folder_path, f"{filename}{caption_ext}")
    content = ""
    if os.path.exists(text_path):
        json_obj["text_path"] = text_path
        try:
            content = open(text_path, encoding="utf-8").read()
            json_obj["text_path_sha"] = compute_file_sha256(text_path)
        except UnicodeDecodeError:
            # если встречаются не-UTF8 символы — очищаем
            content = open(text_path, errors="ignore").read()
            json_obj["text_path_sha"] = ""

    # Определяем путь к .npflux-кэшу
    file_path = os.path.join(folder_path, filename)
    npz_path = exist_npz_path if exist_npz_path and os.path.exists(exist_npz_path) else f"{file_path}{cache_ext}"
    json_obj["npz_path"] = npz_path

    # Если кэш уже есть и не надо его пересоздавать — возвращаем метаданные
    if not recreate_cache and os.path.exists(npz_path):
        json_obj["npz_path_sha"] = json_obj.get("npz_path_sha", compute_file_sha256(npz_path))
        return json_obj

    # Вычисляем эмбеддинги текста (clip + t5) через вспомогательную функцию
    prompt_embeds, pooled_prompt_embeds, txt_attention_masks = compute_text_embeddings(
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        prompt=content,
        device=text_encoders[0].device
    )

    # Убираем лишние оси, переводим на CPU
    npz_dict = {
        "prompt_embed":           prompt_embeds.squeeze(0).cpu(),
        "pooled_prompt_embed":    pooled_prompt_embeds.squeeze(0).cpu(),
        "txt_attention_mask":     txt_attention_masks.squeeze(0).cpu(),
    }

    # 6) При наличии pipe_prior_redux — дополнительно обрабатываем эмбеддинги
    if pipe_prior_redux is not None:
        img_path = redux_image_path or image_path
        image = load_image(img_path)
        prior_out = pipe_prior_redux(image,
                                     prompt_embeds=prompt_embeds,
                                     pooled_prompt_embeds=pooled_prompt_embeds)
        
        # Заполняем оси
        npz_dict["prompt_embed"]        = prior_out.prompt_embeds.squeeze(0).cpu()
        npz_dict["pooled_prompt_embed"] = prior_out.pooled_prompt_embeds.squeeze(0).cpu()

    # 7) Сохраняем кэш и возвращаем метаданные
    torch.save(npz_dict, npz_path)
    return json_obj

def compute_text_embeddings(
    text_encoders: list[nn.Module],
    tokenizers: list,
    prompt: Union[str, list[str]],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Вычисляет текстовые эмбеддинги для заданного промпта на указанном устройстве.

    Эта функция:
      1. Отключает градиенты (`torch.no_grad`), так как эмбеддинги используются только для инференса.
      2. Вызывает `encode_prompt`, который возвращает:
         - пошаговые эмбеддинги от T5 (`prompt_embeds`),
         - «pooled» эмбеддинги от CLIP (`pooled_prompt_embeds`),
         - маску внимания от T5 (`txt_attention_masks`).
      3. Переносит эмбеддинги в нужное устройство (`device`).
      4. Возвращает три тензора: `prompt_embeds`, `pooled_prompt_embeds` и `txt_attention_masks`.

    Args:
        text_encoders (list[nn.Module]):
            Список энкодеров в том порядке, в котором их ожидает `encode_prompt`.
        tokenizers (list):
            Соответствующие токенизаторы для энкодеров.
        prompt (str | list[str]):
            Один или несколько текстовых промптов для кодирования.
        device (torch.device):
            Устройство (CPU или CUDA), на котором должны находиться выходные тензоры.

    Returns:
        Tuple[
            torch.Tensor,  # prompt_embeds: [batch_size, seq_len, hidden_dim]
            torch.Tensor,  # pooled_prompt_embeds: [batch_size, hidden_dim]
            torch.Tensor   # txt_attention_masks: [batch_size, seq_len]
        ]
    """
    # Отключаем автоградиент для ускорения и экономии памяти
    with torch.no_grad():
        # Генерируем все необходимые текстовые эмбеддинги
        prompt_embeds, pooled_prompt_embeds, txt_attention_masks = encode_prompt(
            text_encoders,
            tokenizers,
            prompt,
            device=device
        )
        # Убедимся, что эмбеддинги находятся на нужном устройстве
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    # Возвращаем три тензора: шаговые эмбеддинги, pooled-эмбеддинги и маску внимания
    return prompt_embeds, pooled_prompt_embeds, txt_attention_masks

def vae_encode(vae, image):
    """
    Кодирует входное изображение в латентное пространство вариационного автокодировщика (VAE).

    Args:
        vae: объект вариационного автокодировщика с методом .encode() и атрибутом .device.
        image (torch.Tensor): Тензор изображения формы (C, H, W) или (batch, C, H, W).

    Returns:
        dict: Словарь с ключом 'latent' — тензор латентного представления (на CPU).
    """
    # Собираем батч из одного изображения
    pixel_values = []
    pixel_values.append(image)

    # Конкатенируем список тензоров вдоль новой оси (batch)
    pixel_values = torch.stack(pixel_values).to(vae.device)

    # Отключаем вычисление градиентов — экономия памяти и скорость при инференсе
    with torch.no_grad():
        # Пропускаем батч через энкодер VAE и берём выборку из нормального распределения латентов
        latent = vae.encode(pixel_values).latent_dist.sample().squeeze(0)
        # Освобождаем память GPU от промежуточных тензоров
        del pixel_values
        gc.collect()
        torch.cuda.empty_cache()

    # Переносим результат на CPU и упаковываем в словарь
    latent_dict = {
        'latent': latent.cpu()
    }
    return latent_dict


def read_image(
        image_path: Path
):
    """
    Считывает изображение из файла по пути `image_path`, включая файлы с Unicode-путями,
    декодирует его в формат NumPy- массива и конвертирует из BGR в RGB.

    Args:
        image_path (str): Путь к файлу изображения. Может содержать Unicode-символы.

    Returns:
        np.ndarray or None: Трёхмерный массив изображения в формате RGB,
        либо None, если чтение или декодирование не удалось.
    """
    try:
        # Читаем сырые байты из файла (работает с Unicode-путями)
        raw = np.fromfile(image_path, dtype=np.uint8)
        # Декодируем буфер байтов в изображение OpenCV (формат BGR)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) 

        if image is not None:
            # Перекодируем из BGR (стандарт OpenCV) в RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to open {image_path}.")
    except Exception as e:
        # Любые ошибки чтения или декодирования выводим в консоль
        print(f"An error occurred while processing {image_path}: {e}")
        image = None

    return image

@torch.no_grad()
def cache_multiple(
    vae,
    json_obj: Dict[str, Any],
    latent_ext: str = ".npfluxlatent",
    recreate_cache: bool = False
) -> Dict[str, Any]:
    """
    Обрабатывает сразу несколько изображений из json_obj:
      - читает исходники (factual_image, ground_true, redux, mask),
      - кодирует их с помощью VAE,
      - сохраняет латентные представления в файлы,
      - обновляет json_obj метаданными (пути и SHA).

    Args:
        vae: экземпляр вариационного автокодировщика с методами .encode() и .device.
        json_obj: словарь с путями:
            "npz_path", "factual_image_path", "ground_true_path",
            опционально "redux_image_path" и "factual_image_mask_path".
        resolution: целевое разрешение.
        latent_ext: расширение для файлов с латентами.
        recreate_cache: флаг перезаписи существующих кэшей.

    Returns:
        Обновлённый json_obj с дополнительными ключами:
          - "<image_class>": {"latent_path", "latent_path_sha", "npz_path_sha" (для factual и ground_true)}
          - "redux_image": аналогично,
          - "factual_image_masked_image" и "factual_image_mask": пути и sha.
    """
    # Путь к .npz, нужный для хешей
    npz_path = json_obj["npz_path"]

    # Список основных классов изображений и их путей в json
    image_files = [
        ("factual_image", json_obj["factual_image_path"]),
        ("ground_true",   json_obj["ground_true_path"])
    ]

    # Трансформация: в тензор и нормализация в [-1,1]
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Обработка factual_image и ground_true
    factual_image = None
    f_height = f_width = 0
    for image_class, image_path in image_files:
        filename, _ = os.path.splitext(image_path)

        # Инициализируем подсловарь для класса, если его нет
        json_obj.setdefault(image_class, {})
        latent_cache_path = f"{filename}{latent_ext}"
        json_obj[image_class]["latent_path"] = latent_cache_path

        # Читаем изображение
        image_np = read_image(image_path)
        if image_np  is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        # Обновляем размер бакета
        image_height, image_width, _ = image_np.shape
        json_obj['bucket'] = f"{image_width}x{image_height}"

        # Пропускаем, если кэш уже есть и не требуется пересоздание
        if os.path.exists(latent_cache_path) and not recreate_cache:
            # Если ещё нет SHA в json, добавляем
            if 'latent_path_sha' not in json_obj[image_class]:
                json_obj[image_class]['latent_path_sha'] = compute_file_sha256(latent_cache_path)
                json_obj[image_class]['npz_path_sha']    = compute_file_sha256(npz_path)
            continue

        # Преобразуем изображение в тензор и нормализуем
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        image = image / 255.0
        image = (image - 0.5) / 0.5

        # Сохраняем factual_image для маскировки позже
        if image_class == "factual_image":
            factual_image = image.unsqueeze(0)
            f_height, f_width = image_height, image_width

        # Кодируем через VAE и сохраняем латент
        latent_dict = vae_encode(vae, image)
        torch.save(latent_dict, latent_cache_path)
        json_obj[image_class]['latent_path_sha'] = compute_file_sha256(latent_cache_path)

    # Освобождаем память
    gc.collect()
    torch.cuda.empty_cache()

    # Если уже есть маскированное изображение и маска — завершаем
    if (
        "factual_image_masked_image" in json_obj and
        "factual_image_mask" in json_obj and
        not recreate_cache
    ):
        return json_obj

    # Рассчитываем масштабный фактор VAE (например, 8 для стандартных архитектур)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Обработка redux_image (если задана)
    if "redux_image_path" in json_obj:
        redux_image_path = json_obj["redux_image_path"]
        filename, _ = os.path.splitext(redux_image_path)
        redux_cache_path = f"{filename}{latent_ext}"

        redux_image = read_image(redux_image_path)
        if redux_image  is None:
            raise FileNotFoundError(f"Не удалось загрузить redux изображение: {redux_image_path}")

        # Подгоняем под фактический размер factual_image
        redux_image = cv2.resize(
            redux_image,
            (int(f_width), int(f_height)),
            interpolation=cv2.INTER_LANCZOS4
        )

        redux_image = train_transforms(redux_image)
        latent_dict = vae_encode(vae, redux_image)
        torch.save(latent_dict, redux_cache_path)

        json_obj.setdefault("redux_image", {})
        json_obj["redux_image"]["latent_path"]        = redux_cache_path
        json_obj["redux_image"]['latent_path_sha']    = compute_file_sha256(redux_cache_path)
    else:
        # Если redux не задан — оставляем пустой путь
        json_obj.setdefault("redux_image", {})["latent_path"] = ""

    # Подготовка путей для маски и маскированного изображения
    mask_path = json_obj["factual_image_mask_path"]
    filename, _ = os.path.splitext(mask_path)
    masked_latent_path = f"{filename}_masked_image{latent_ext}"
    mask_latent_path   = f"{filename}{latent_ext}"

    json_obj.setdefault("factual_image_masked_image", {})["latent_path"] = masked_latent_path
    json_obj.setdefault("factual_image_mask", {})["latent_path"]        = mask_latent_path

    # Если оба кэша уже есть и recreate_cache=False — возвращаем
    if (
        os.path.exists(masked_latent_path) and
        os.path.exists(mask_latent_path) and
        not recreate_cache
    ):
        return json_obj

    # Настройка процессора масок
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,
        vae_latent_channels=vae.config.latent_channels,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
        do_resize=False
    )

    # Читаем и готовим маску
    mask_image = read_image(mask_path)
    if mask_image  is None:
        raise FileNotFoundError(f"Не удалось загрузить redux изображение: {mask_path}")

    # Конвертируем
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    mask_image = mask_processor.preprocess(mask_image, height=f_height, width=f_width)

    # Проверка фактического изображения
    if factual_image is None:
        raise FileNotFoundError(f"Не удалось загрузить фактическое изображение.")    

    # Применяем маску к factual_image
    masked_image = factual_image * (1 - mask_image)
    masked_image = masked_image.to(device=vae.device)

    # Рассчитываем форму латентного представления
    height, width = factual_image.shape[-2:]
    height = 2 * (height // (vae_scale_factor * 2))
    width  = 2 * (width  // (vae_scale_factor * 2))

    # Кодируем маскированное изображение
    masked_latent = vae.encode(masked_image).latent_dist.sample().squeeze(0)

    # Приводим маску к форме латентов: (1,8*8, H, W)
    mask_image = mask_image[:, 0, :, :]
    mask_image = mask_image.view(1, height, vae_scale_factor, width, vae_scale_factor)
    mask_image = mask_image.permute(0, 2, 4, 1, 3)
    mask_image = mask_image.reshape(1, vae_scale_factor**2, height, width)

    # Сохраняем латент маскированного изображения
    torch.save({'latent': masked_latent.cpu()}, masked_latent_path)
    json_obj["factual_image_masked_image"]['latent_path_sha'] = compute_file_sha256(masked_latent_path)

    # Сохраняем латент самой маски
    torch.save({'latent': mask_image.squeeze().cpu()}, mask_latent_path)
    json_obj["factual_image_mask"]['latent_path_sha'] = compute_file_sha256(mask_latent_path)

    return json_obj

def unwrap_model(model, accelerator: Accelerator):
    """
    Разворачивает (unwraps) модель, обёрнутую через Accelerate и/или torch.compile.

    Args:
        model: обёрнутый или скомпилированный torch-модуль.
        accelerator: экземпляр accelerate.Accelerator, через который модель готовилась.

    Returns:
        Исходный torch.nn.Module без обёрток Accelerate и torch.compile.
    """
    # Снимаем обёртку Accelerate (например, для FSDP, DDP и т.п.)
    model = accelerator.unwrap_model(model)
    # Если модель была скомпилирована (torch.compile), _orig_mod — это оригинал
    model = model._orig_mod if is_compiled_module(model) else model
    return model

# ============ #
#              #
#  ACCELERATE  #
#              # 
# ============ #

def ensure_accelerate():
    """
    Если скрипт запущен напрямую, а не через `accelerate launch`,
    автоматически перезапускает его под Accelerate на Windows и Linux.
    """
    # Accelerate выставляет LOCAL_RANK>=0; по умолчанию —1
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        # Строим команду: <python> -m accelerate launch <текущий скрипт> <остальные args>
        cmd = [
            sys.executable,
            "-m", "accelerate", "launch",
            sys.argv[0]
        ] + sys.argv[1:]
        logging.info("Перезапуск через Accelerate:", " ".join(cmd))
        # Запускаем новый процесс и завершаем текущий
        subprocess.run(cmd, check=True)
        sys.exit()  # никогда не вернёмся сюда

# ================== #
#                    #
#  Алгоритмы         #
#                    #
# ================== #

def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """
    Вычисляет SHA-256-хеш файла и возвращает его в виде шестнадцатеричной строки.

    Args:
        file_path (str | Path): Путь к файлу, для которого нужно получить SHA-256 хеш.

    Returns:
        str: Шестнадцатеричный SHA-256 хеш файла.  
             В случае ошибки при чтении файла возвращается пустая строка.
    """
    # Приводим вход к объекту Path, чтобы можно было работать везде одинаково
    path = Path(file_path)
    
    # Инициализируем объект хеширования SHA-256
    sha256 = hashlib.sha256()

    try:
        # Открываем файл в бинарном режиме и читаем его кусками по 8 КБ,
        # чтобы не загружать весь файл в память
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        # Возвращаем шестнадцатеричное представление хеша
        return sha256.hexdigest()
    except Exception as e:
        # Если произошла любая ошибка (нет доступа, файл не найден и т.п.),
        # печатаем её и возвращаем пустую строку
        print(f"Ошибка вычисления SHA-256 для файла '{path}': {e}")
        return ""
    
def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 0.0,
    mode_scale: float = 0.0,
):
    """
    Вычисляет плотность (u) для сэмплирования временных шагов (timesteps) при обучении модели SD3.

    Аргументы:
        weighting_scheme (str): схема взвешивания (например, 'logit_normal', 'mode', 'logit_snr', 'uniform').
        batch_size (int): размер батча.
        logit_mean (float): среднее значение для нормального распределения в логит-пространстве.
        logit_std (float): стандартное отклонение для нормального распределения в логит-пространстве.
        mode_scale (float): параметр масштаба для 'mode'-взвешивания.

    Возвращает:
        torch.Tensor: значения u для каждого примера в батче, лежащие в диапазоне [0, 1].
    """
    if weighting_scheme == "logit_normal":
        # Используем логит-нормальное распределение как в разделе 3.1 статьи SD3.
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        # Режимно-смещённое распределение
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "logit_snr":
         # Используем логарифм отношения сигнал/шум, как указано в статье (https://arxiv.org/pdf/2411.14793)
        logsnr = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(-logsnr/2)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Вычисляет весовую схему потерь для обучения SD3 (Scaling Rectified Flow Transformers).

    Аргументы:
        weighting_scheme (str): выбирает способ взвешивания — "sigma_sqrt", "cosmap" или другие.
        sigmas (Tensor): шумовые коэффициенты, соответствующие временным шагам.

    Возвращает:
        Tensor: вес для каждого элемента, одинаковой формы с sigmas.
    """
    if sigmas is not None:
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            weighting = torch.ones_like(sigmas)
    return weighting

def is_url(path: str) -> bool:
    """
    Проверка, является ли строка URL
    """
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")

# ================== #
#                    #
#  Основная функция  #
#                    #
# ================== #

def main():

    # Проверка запуска
    #ensure_accelerate()

    # Корень там, где запускается файл
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", mode="a")
        ]
    )

    # Настройка PyTorch
    logging.info("Настройка PyTorch")
    torch._dynamo.config.recompile_limit = 100

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Создание директорий
    logging.info("Создание директорий")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Создание путей к метаинформации
    logging.info("Создание путей к метаинформации")
    metadata_path = os.path.join(TRAIN_DATA_DIR, f'metadata_{METADATA_PREFIX}.json')
    val_metadata_path =  os.path.join(TRAIN_DATA_DIR, f'val_metadata_{METADATA_PREFIX}.json')
    logging.info(f"metadata_path: {metadata_path}")
    logging.info(f"val_metadata_path: {val_metadata_path}")

    # Создание конфигурации для оптимизатора (accelerator)
    logging.info("Создание конфигурации для оптимизатора (accelerator)")
    accelerator_project_config = ProjectConfiguration(
        project_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR
    )

    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )

    # Создание оптимизатора (accelerator)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        log_with=REPORT_TO,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Для обучения со смешанной точностью мы приводим все нетренируемые веса (VAE, нетренируемый текстовый энкодер без LoRA и нетренируемый трансформер без LoRA) к половинной точности,
    # так как эти веса используются только для инференса, и хранение их в полной точности не требуется
    weight_dtype = DTYPE_MAP.get(accelerator.mixed_precision, torch.float32)
    logging.info(f"Смешанная точность: {weight_dtype}")

    # Загрузка планировщика и базовой модели
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
        subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Подготовка датасета
    if True:
        logging.info(f"Подготовка датасета")
        supported_image_types = ['.jpg','.jpeg','.png','.webp']
        files = glob.glob(
            f"{TRAIN_DATA_DIR}/**",
            recursive=True
        )
        image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_types]

        # Списки файлов
        gt_files = []
        factual_image_files = []
        factual_image_masks = []

        # Фильтрация изображений по суффиксу _G
        for f in image_files:
            # Получаем базовое название изображения
            base_name = os.path.basename(f)
            filename, _ = os.path.splitext(base_name)
            
            # Получаем индексы
            gt_index = find_index_from_right(filename, "_G")
            factual_index = find_index_from_right(filename, "_F")
            mask_index = find_index_from_right(filename, "_M")

            # Заполняем списки файлов
            if gt_index > 0:
                gt_files.append(f)
            elif mask_index > 0 and mask_index > factual_index:
                factual_image_masks.append(f)
            elif factual_index > 0:
                factual_image_files.append(f)

        # Создать сопоставление между базовым именем файла (без суффикса) и файлом эталонных данных
        gt_mapping = {}
        for gt_file in gt_files:
            # Получаем базовое название изображения
            base_name = os.path.basename(gt_file)
            filename, _ = os.path.splitext(base_name)
            
            # Получаем имя файла без суффикса
            suffix_index = find_index_from_right(filename, "_G")
            filename_without_suffix = filename[:suffix_index]
            
            # Заполняем сопоставление изображений базовое-эталон
            subdir = os.path.dirname(gt_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"  # Remove '_G'
            gt_mapping[mapping_key] = gt_file

        # Создать сопоставление между базовым именем файла (без суффикса) и файлом маски
        mask_mapping = {}
        for mask_file in factual_image_masks:
            # Получаем базовое название изображения
            base_name = os.path.basename(mask_file)
            filename, _ = os.path.splitext(base_name)
            
            # Получаем имя маски без суффикса
            mask_index = find_index_from_right(filename, "_M")
            filename_without_suffix = filename[:mask_index]
            
            # Получаем индекс базового изображения
            factual_index = find_index_from_right(filename_without_suffix, "_F")          
            if factual_index > 0:
                filename_without_suffix = filename[:factual_index]
            
            # Заполняем сопоставление изображений базовое-маска
            subdir = os.path.dirname(mask_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"  # Remove '_G'
            mask_mapping[mapping_key] = mask_file

        # Создать сопоставление базовое-эталон-маска
        factual_pairs = []
        for factual_file in factual_image_files:
            # Получаем базовое название изображения
            base_name = os.path.basename(factual_file)
            filename, _ = os.path.splitext(base_name)
            
            # Получаем название без суффикса
            suffix_index = find_index_from_right(filename, "_F")
            filename_without_suffix = filename[:suffix_index]
            
            # Получаем индекс
            subdir = os.path.dirname(factual_file)
            mapping_key = f"{subdir}_{filename_without_suffix}"
            
            # Перебираем эталонные изображения на основе базового имени файла
            if mapping_key in gt_mapping:
                # Эталонное изображение
                gt_file = gt_mapping[mapping_key]
                
                # Поиск маски
                if mapping_key in mask_mapping:
                    # Маска
                    mask_file = mask_mapping[mapping_key]
                    
                    # Задаем полное сопоставление
                    factual_pairs.append((gt_file, factual_file, mask_file))   

        # Подготовка и загрузка существующих метаданных
        logging.info(f"Подготовка метаданных")
        full_datarows = []
        if len(factual_pairs) > 0:
            # Загружаем существующую метаинформацию
            if os.path.exists(metadata_path) and os.path.exists(val_metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as readfile:
                    metadata_datarows = json.loads(readfile.read())
                    full_datarows += metadata_datarows
                    
                with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                    val_metadata_datarows = json.loads(readfile.read())
                    full_datarows += val_metadata_datarows
            # Если нет - генерируем её
            else:
                # Загрузка токенайзеров
                tokenizer_one = CLIPTokenizer.from_pretrained(
                    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    subfolder="tokenizer",
                )

                tokenizer_two = T5TokenizerFast.from_pretrained(
                    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    subfolder="tokenizer_2",
                    legacy=False
                )

                # Загрузка моделей для токенайзеров
                text_encoder_cls_one = import_model_class_from_model_name_or_path(
                    model_name_or_path=PRETRAINED_MODEL_NAME,
                    subfolder="text_encoder"
                )

                text_encoder_cls_two = import_model_class_from_model_name_or_path(
                    model_name_or_path=PRETRAINED_MODEL_NAME,
                    subfolder="text_encoder_2"
                )

                # Загрузка энкодеров
                text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

                # Загрузка автоэнкодера
                vae = AutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME,
                    cache_dir=MODEL_CACHE_DIR,
                    subfolder="vae",
                )

                # Морозим веса (они меняться не будут)
                vae.requires_grad_(False)
                text_encoder_one.requires_grad_(False)
                text_encoder_two.requires_grad_(False)

                # Перемещаем всё на асселятор
                vae.to(accelerator.device, dtype=torch.float32)
                text_encoder_one.to(accelerator.device, dtype=weight_dtype)
                text_encoder_two.to(accelerator.device, dtype=weight_dtype)

                # Собираем всё в массивы
                tokenizers = [tokenizer_one, tokenizer_two]
                text_encoders = [text_encoder_one, text_encoder_two]

                # Генерируем пустые эмбеддинги
                create_empty_embedding(
                    tokenizers=tokenizers,
                    text_encoders=text_encoders,
                    cache_path=os.path.join(MODEL_CACHE_DIR, "empty_embedding.npflux"),
                    recreate=RECREATE_CACHE
                )

                # Генерируем эмбеддинги
                embedding_objects = []
                for gt_file, factual_image_file, factual_image_mask in tqdm(factual_pairs, "Генерация эмбеддингов"):
                    # Получаем изображение и его директорию
                    file_name = os.path.basename(factual_image_file)
                    folder_path = os.path.dirname(factual_image_file)
                    
                    # Создание текстового ембеддинга на основе фактического изображения
                    f_json = create_embedding(
                        tokenizers=tokenizers,
                        text_encoders=text_encoders,
                        folder_path=folder_path,
                        file=file_name,
                        resolutions=(HEIGHT, WIDTH),
                        cache_ext=".npFluxFill",
                        recreate_cache=RECREATE_CACHE
                    )
                    
                    # Заполняем связки изображений
                    f_json["ground_true_path"] = gt_file
                    f_json["factual_image_path"] = factual_image_file
                    f_json["factual_image_mask_path"] = factual_image_mask
                    embedding_objects.append(f_json)

                # Очистка памяти
                del text_encoders, tokenizers
                gc.collect()
                torch.cuda.empty_cache()

                # Кэширование латентов
                metadata_datarows = []
                for json_obj in tqdm(embedding_objects, "Кэширование латентов"):
                    full_obj = cache_multiple(
                        vae=vae,
                        json_obj=json_obj,
                        latent_ext=".npFluxFillLatent",
                        recreate_cache=RECREATE_CACHE
                    )                   
                    metadata_datarows.append(full_obj)

                full_datarows += metadata_datarows

                # Очистка памяти
                del vae, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
                gc.collect()
                torch.cuda.empty_cache()

            # Выделение валидационного сета
            if VALIDATION_RATIO > 0:
                # Вычисляем процентное соотношение
                train_ratio = 1 - VALIDATION_RATIO
                validation_ratio = VALIDATION_RATIO

                # Защита от ситуаций, когда запись всего одна
                if len(full_datarows) == 1:
                    full_datarows = full_datarows + full_datarows.copy()
                    validation_ratio = 0.5
                    train_ratio = 0.5

                # Разбиение записей
                training_datarows, validation_datarows = train_test_split(
                    full_datarows,
                    train_size=train_ratio,
                    test_size=validation_ratio
                )

                # Сохранение инфо о тренировочных данных
                datarows = training_datarows

                # Сохранение инфо о валидационных данных
                if len(validation_datarows) > 0:
                    with open(val_metadata_path, "w", encoding='utf-8') as outfile:
                        outfile.write(json.dumps(validation_datarows))

                # Очистка памяти
                del validation_datarows
            else:
                # Сохранение инфо о тренировочных данных
                datarows = metadata_datarows

            # Сохранение обновленной метаинформации о тренировочных данных
            with open(metadata_path, "w", encoding='utf-8') as outfile:
                outfile.write(json.dumps(datarows))

        # Очистка памяти
        gc.collect()
        torch.cuda.empty_cache()

    # Кол-во записей с данными
    datarows = datarows * REPEATS

    # Кол-во записей с данными
    offload_device = accelerator.device

    # Загрузка модели
    if True:
        # Настройка квантования
        quantization_config = GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16 
        )

        # Загрузка из репозитория  
        if is_url(MODEL_PATH):
            transformer = FluxTransformer2DModel.from_single_file(
                pretrained_model_link_or_path_or_dict=MODEL_PATH,
                quantization_config=quantization_config,
                cache_dir=MODEL_CACHE_DIR
            ).to(offload_device)
        else:     
            transformer = FluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=MODEL_PATH,
                subfolder="transformer",
                cache_dir=MODEL_CACHE_DIR
            ).to(offload_device, dtype=weight_dtype)

    # Запрещаем обучение параметров трансформера
    transformer.requires_grad_(False)

    # Если включён gradient checkpointing, активируем его в трансформере
    if GRADIENT_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()


    # Конфигурируем LoRA-адаптацию для attention-слоёв
    transformer_lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=LORA_LAYERS,
    )

    # Применяем добавочный адаптер к трансформеру
    transformer.add_adapter(transformer_lora_config)

    # Собираем списки для заморозки определённых слоёв
    layer_names = []
    freezed_layers = []
    
    # Для каждого параметра трансформера
    for name, param in transformer.named_parameters():
        layer_names.append(name)
        if "transformer" in name:
            # Получаем номер слоя из имени вида "transformer.3.layer..."
            name_split = name.split(".")
            layer_order = name_split[1]
            # Если этот слой в списке для заморозки — отключаем градиенты
            if int(layer_order) in freezed_layers:
                param.requires_grad = False

    def save_model_hook(models, weights, output_dir):
        """
        Сохраняет веса LoRA (Low-Rank Adaptation) для модели или моделей, если текущий процесс является основным.
        
        Параметры:
        - models (list): Список моделей, содержащих LoRA-адаптеры.
        - weights (list): Список весов, соответствующих моделям. Используется для удаления веса из очереди сохранения.
        - output_dir (str): Путь к директории, куда будут сохранены веса.

        Поведение:
        - Проверяется, что текущий процесс — главный.
        - Из моделей извлекаются LoRA-слои и преобразуются к формату diffusers.
        - Сохраняются извлечённые веса LoRA в указанную директорию.
        - Копируется файл весов `pytorch_lora_weights.safetensors` с новым именем, соответствующим имени директории.
        """
        # Проверяем, является ли текущий процесс главным (в мультипроцессной среде).
        if accelerator.is_main_process:
            # Инициализация переменной, в которую будут сохранены преобразованные LoRA-слои
            transformer_lora_layers_to_save = None
            # Обход всех моделей в списке
            for model in models:
                # Проверяем, является ли модель экземпляром ожидаемого типа (например, обёрнутый трансформер)
                if isinstance(model, type(unwrap_model(transformer, accelerator))):
                    # Получаем state_dict модели и конвертируем его в формат diffusers
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                
                else:
                    # Если модель не того типа — выбрасываем ошибку
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # Удаляем соответствующий вес из списка weights, чтобы он не был повторно сохранён
                weights.pop()

            # Проверка успешности получения слоев
            if transformer_lora_layers_to_save is None:
                raise ValueError("transformer_lora_layers_to_save is None")
            
            # Сохраняем LoRA-веса, используя метод пайплайна
            FluxFillPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save
            )
            
            # Формируем путь к новому файлу, имя которого соответствует имени папки
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"

            # Исходный путь к весам, сохранённым в формате safetensors
            ori_file = f"{output_dir}/pytorch_lora_weights.safetensors"

            # Если файл существует — копируем его с новым именем
            if os.path.exists(ori_file): 
                shutil.copy(ori_file, file_path)

    def load_model_hook(models, input_dir):
        """
        Загружает веса LoRA-адаптера из директории `input_dir` и применяет их к соответствующей модели в списке `models`.

        Параметры:
        - models (list): Список моделей, одна из которых — ожидаемый трансформер (например, UNet с LoRA).
        - input_dir (str): Путь к директории, содержащей сохранённые LoRA-веса в формате safetensors.

        Поведение:
        - Извлекается модель из списка `models`, если она соответствует ожидаемому типу.
        - Загружается словарь LoRA-весов (`lora_state_dict`) из директории.
        - Отбираются только ключи, относящиеся к `transformer`, и преобразуются к формату PEFT.
        - Веса загружаются в модель, регистрируются несовместимые ключи, если они есть.
        - Предупреждение логируется, если обнаружены неожиданные ключи в LoRA-весах.
        """
        # Переменная для хранения найденной модели трансформера
        transformer_ = None

        # Обработка всех моделей из списка `models`
        while len(models) > 0:
            model = models.pop()
            # Проверяем, соответствует ли модель типу ожидаемой (например, обёрнутый трансформер)
            if isinstance(model, type(unwrap_model(transformer, accelerator))):
                transformer_ = model
            else:
                # Если тип модели неожиданный — выбрасываем исключение
                raise ValueError(f"unexpected save model: {model.__class__}")

        # Загружаем веса LoRA (в формате state_dict) из указанной директории
        lora_state = FluxFillPipeline.lora_state_dict(input_dir)

        # Проверка модели LoRA
        if lora_state is None:
            raise ValueError("lora_state_dict is None")
        
        # Если вернулся кортеж (state_dict, something), берём только state_dict
        if isinstance(lora_state, tuple) and len(lora_state) >= 1:
            lora_state_dict = lora_state[0]
        elif isinstance(lora_state, dict):
            lora_state_dict = lora_state
        else:
            raise TypeError(f"Unexpected return type from lora_state_dict: {type(lora_state)}")

        # Оставляем только те ключи, которые относятся к "transformer", убирая префикс "transformer."
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }

        # Преобразуем UNet-слои в формат, ожидаемый PEFT
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

        # Загружаем параметры в модель. Возвращаются несовместимые ключи
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")

        # Если есть неожиданные ключи — логируем предупреждение
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logging.warning(f"Загрузка весов адаптера из state_dict привела к появлению неожиданных ключей, не найденных в модели:\n{unexpected_keys}.")

    # Регистрируем хуки сохранения и загрузки состояния модели через Accelerator (для LoRA)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Включаем поддержку TensorFloat-32 (TF32) на Ampere GPU — это может ускорить обучение
    # Подробнее: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Если используется mixed precision с FP16, нужно привести только обучаемые параметры (например, LoRA) к float32
    if MIXED_PRECISION == "fp16":
        models: list[Module] = [cast(Module, transformer)]
        cast_training_params(models, dtype=torch.float32)

    # Извлекаем параметры LoRA у которых requires_grad=True
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Оборачиваем параметры в словарь с LR, как ожидает PyTorch-совместимый оптимизатор
    transformer_lora_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": LEARNING_RATE
    }
    params_to_optimize = [transformer_lora_parameters_with_lr]
    
    # Выбор оптимизатора AdamW.
    if OPTIMIZER_NAME.lower() == "adamw":
        # Использование квантованного оптимизатора
        if USE_8BIT_ADAM:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Создание оптимизатора
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=ADAM_WEIGHT_DECAY,
            eps=ADAM_EPSILON,
        )

    # Выбор оптимизатора Prodigy.
    if OPTIMIZER_NAME.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        # Защита на случай ввода LEARNING_RATE от AdamW
        if LEARNING_RATE <= 0.1:
            logging.warning("Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0")

        # Создание оптимизатора
        optimizer = optimizer_class(
            params_to_optimize,
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            beta3=PRODIGY_BETA3,
            d_coef=PRODIGY_D_COEF,
            weight_decay=round(ADAM_WEIGHT_DECAY),
            eps=ADAM_EPSILON,
            decouple=PRODIGY_DECOUPLE,
            use_bias_correction=PRODIGY_USE_BIAS_CORRECTION,
            safeguard_warmup=PRODIGY_SAFEGUARD_WARMUP,
        )
    
    # Создание набора данных на основе входной директории.
    train_dataset = CachedMaskedPairsDataset(
        datarows=datarows,
        conditional_dropout_percent=CAPTION_DROPOUT
    )

    # Создание BucketBatchSampler для группировки по "похожим" объектам
    bucket_batch_sampler = BucketBatchSampler(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        drop_last=True
    )

    # Инициализация DataLoader с использованием bucket-батчера
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=bucket_batch_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=NUM_WORKERS
    )
    
    # Планировка обучения и вычисления, связанная с количеством шагов тренировки.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRAD_ACCUMULATION_STEPS)
    max_train_steps = NUM_TRAIN_EPOCHS * num_update_steps_per_epoch

    # Пересчёт числа шагов обновления параметров за эпоху, т.к. размер train_dataloader мог измениться после настройки данных.
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Настройки VAE
    vae_config_shift_factor = 0.1159
    vae_config_scaling_factor = 0.3611
    vae_config_block_out_channels = [
        128,
        256,
        512,
        512
    ]

    # Создаем lr_scheduler — планировщик изменения learning rate в процессе обучения
    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=COSINE_RESTARTS
    )

    # Подготавливаем оптимизатор, даталоадер и scheduler для ускорения с помощью accelerate
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # Чистим память
    gc.collect()
    torch.cuda.empty_cache()
    
    # Подготавливаем модель для работы с accelerate с настройкой размещения устройства
    transformer = accelerator.prepare(transformer)
    
    # Инициализируем трекеры (например, для логгинга метрик), только на основном процессе
    if accelerator.is_main_process:
        tracker_name = "FluxFill"
        try:
            accelerator.init_trackers(tracker_name)
        except:
            print("Trackers not initialized")
    
    # Вычисляем итоговый размер батча с учетом процессов и накопления градиентов
    total_batch_size = TRAIN_BATCH_SIZE * accelerator.num_processes * GRAD_ACCUMULATION_STEPS

    # Использование SageAttention
    if USE_SAGE_ATT:
        from sageattention import sageattn
        F.scaled_dot_product_attention = sageattn
    
    # Вывод инфо
    logging.info(f"Количество примеров: {len(train_dataset)}")
    logging.info(f"Количество эпох: {num_train_epochs}")
    logging.info(f"Количество шагов обновления за эпоху: {num_update_steps_per_epoch}")
    logging.info(f"Максимальное количество шагов обучения: {max_train_steps}")
    logging.info(f"Мгновенный размер батча на устройство: {TRAIN_BATCH_SIZE}")
    logging.info(f"Общий размер батча для обучения (с учётом параллельности, распределения и накопления): {total_batch_size}")
    logging.info(f"Шаги накопления градиента: {GRAD_ACCUMULATION_STEPS}")
    logging.info(f"Общее количество оптимизационных шагов = {max_train_steps}")
    
    # Обнуляем счетчики
    global_step = 0  # Общий шаг
    first_epoch = 0  # Начальная эпоха
    resume_step = 0  # Шаг, с которого восстановились

    # Пытаемся загрузить веса и состояние тренировки из сохранённой контрольной точки (checkpoint)
    if RESUME_FROM_CHECKPOINT and RESUME_FROM_CHECKPOINT != "":
        # Использование конкретного чекпоинта
        if RESUME_FROM_CHECKPOINT != "latest":
            # Пользователь указал конкретный путь к checkpoint — используем basename данного пути
            path = os.path.basename(RESUME_FROM_CHECKPOINT)
        # Иначе - последнего доступного
        else:
             # Пользователь запросил “latest” — ищем последний checkpoint в output_dir
            dirs = os.listdir(OUTPUT_DIR)
            dirs = [d for d in dirs if d.startswith(SAVE_NAME)]
            # Сортируем по номеру шага (int из части после "-")
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        # Если checkpoint не найден — начинаем новую тренировку
        if path is None:
            accelerator.print(
                f"Контрольная точка '{RESUME_FROM_CHECKPOINT}' не найдена. Запуск новой тренировки."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # Загружаем состояние тренировки из найденного checkpoint
            accelerator.print(f"Восстановление из контрольной точки {path}")
            accelerator.load_state(os.path.join(OUTPUT_DIR, path))
            global_step = int(path.split("-")[-1])

            # Определяем, с какого этапа продолжить тренировку
            initial_global_step = global_step
            resume_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Создаём индикатор прогресса    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Шаги",
        disable=not accelerator.is_local_main_process, # показываем только на главном процессе
    )

    # Обработка guidance_embeds — проверяем, нужно ли обрабатывать guidance
    if accelerator.unwrap_model(transformer).config.guidance_embeds:
        handle_guidance = True
    else:
        handle_guidance = False

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """
        Возвращает расширенные сигмы шума для заданных временных шагов difussion scheduler.

        Аргументы:
            timesteps (Tensor): тензор с номерами временных шагов (например, [t1, t2, ...])
            n_dim (int): целевое число размерностей, к которому нужно расширить сигму
            dtype: тип данных в результате (float32 по умолчанию).

        Логика:
        1) Берем сигмы и временные тики из копии scheduler'а, переводим их на нужное устройство и dtype.
        2) Находим индексы в schedule_timesteps, соответствующие каждому timesteps.
        3) Извлекаем нужные сигмы по найденным индексам и выравниваем тензор.
        4) Расширяем тензор до требуемого числа измерений (n_dim) за счёт вставки новых осей.
        5) Возвращаем итоговый тензор sigma.
        """

        # Берем сигмы и временные шаги из scheduler
        sigmas = torch.as_tensor(
            noise_scheduler_copy.sigmas,
            dtype=dtype,
            device=accelerator.device
        )
        schedule_timesteps = torch.as_tensor(
            noise_scheduler_copy.timesteps,
            dtype=torch.long,
            device=accelerator.device
        )
        timesteps = timesteps.to(accelerator.device).long()

        # Определяем индексы сигм для каждого переданного timestep
        step_indices = []
        for t in timesteps:
            # torch.where быстрее и чуть чище, чем nonzero
            matches = torch.where(schedule_timesteps == t)[0]
            if matches.numel() == 0:
                raise RuntimeError(f"Timestep {int(t.item())} не найден в scheduler.timesteps")
            # берём первый попавшийся
            step_indices.append(matches[0].item())

        # Превращаем список индексов в LongTensor
        idx_tensor = torch.tensor(step_indices, dtype=torch.long, device=accelerator.device)

        # Извлекаем сигмы по индексам и делаем плоский вид
        sigma = sigmas[idx_tensor].flatten()

        # Расширяем размерность тензора до n_dim, чтобы он подходил для дальнейших вычислений
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma
    
    # Основной цикл обучения по эпохам
    for epoch in range(first_epoch, num_train_epochs):
        # Перевод модели в режим обучения
        transformer.train()

        # Цикл по батчам
        for step, batch in enumerate(train_dataloader):
            # Обнуляем градиенты
            optimizer.zero_grad()

            # Аккумулирование градиентов при градиентном накоплении
            with accelerator.accumulate(transformer):
                # Очистка памяти
                gc.collect()
                torch.cuda.empty_cache()
                
                # Перемещаем входные данные на нужное устройство
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                ground_trues = batch["ground_true"].to(accelerator.device)
                factual_images = batch["factual_image"].to(accelerator.device)
                factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)
                
                # Случайно выбираем между ground_true и factual_image
                r = random.random()
                latents = factual_images
                if r < REG_RATIO:
                    latents = ground_trues
                    # Обнуляем эмбеддинги текста — убираем влияние текста при регуляризации
                    prompt_embeds = torch.zeros_like(prompt_embeds).to(accelerator.device)
                    pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds).to(accelerator.device)
                    
                # Масштабируем latent-представления с использованием параметров VAE
                latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                latents = latents.to(dtype=weight_dtype)
                                
                # scale ground trues with vae factor
                ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                ground_trues = ground_trues.to(dtype=weight_dtype)
                
                factual_image_masked_images = (factual_image_masked_images - vae_config_shift_factor) * vae_config_scaling_factor
                factual_image_masked_images = factual_image_masked_images.to(dtype=weight_dtype)
                
                # Заглушка для text_ids (не используется)
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                
                # Вычисляем масштаб VAE — используется позже
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                 # Генерируем ID для латентного изображения
                latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2] // 2,
                    latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # Добавляем шум: комбинируем обычный шум и смещение (noise_offset)
                noise = torch.randn_like(latents) + NOISE_OFFSET * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                bsz = latents.shape[0] # размер батча
                
                # Сэмплируем timestep'ы с учётом схемы взвешивания (может быть логит-нормаль)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=WEIGHTING_SCHEME,
                    batch_size=bsz,
                    logit_mean=LOGIT_MEAN,
                    logit_std=LOGIT_STD,
                    mode_scale=MODE_SCALE,
                )

                # Вычисление timestep
                if noise_scheduler_copy.timesteps is None:
                    raise ValueError("noise_scheduler_copy.timesteps is None")
                    
                indices = (u * noise_scheduler_copy.config["num_train_timesteps"]).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                
                # Создание зашумленного входа для модели по формуле z_t = (1 - σ) * x + σ * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                
                # Упаковка латентов для передачи в модель
                packed_noisy_latents = FluxFillPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )

                # Реализация масочного дропаута по случайной вероятности
                if MASK_DROPOUT > random.random():
                    # Если случайное значение < mask_dropout — заменяем маску на единичную
                    factual_image_masks = torch.ones_like(factual_image_masks)
                
                # Упаковываем factual_image_masks для подачи в модель
                packed_factual_image_masks = FluxFillPipeline._pack_latents(
                    factual_image_masks,
                    batch_size=latents.shape[0],
                    num_channels_latents=vae_scale_factor * vae_scale_factor,
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # Упаковываем masked images
                packed_factual_image_masked_images = FluxFillPipeline._pack_latents(
                    factual_image_masked_images,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # Объединяем маскированные изображения и маски по канальному измерению
                masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)

                # Объединяем зашумлённые латенты и масочные латенты
                cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                
                # Обрабатываем guidance: если включён guidance_embeds — расширяем guidance_scale на размер батча
                if handle_guidance:
                    guidance = torch.tensor([GUIDANCE_SCALE], device=accelerator.device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                
                with accelerator.autocast():
                    # Прогон через трансформер: предсказываем шум residual у noisy входа
                    model_pred = transformer(
                        hidden_states=cat_model_input,
                        encoder_hidden_states=prompt_embeds,
                        joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timesteps / 1000,  # временной шаг (в нормированном формате)
                        img_ids=latent_image_ids,   # позиционные ID для батчей изображений
                        txt_ids=text_ids,           # позиционные ID для текстовых токенов 
                        guidance=guidance,          # guidance_scale tensor или None
                        return_dict=False
                    )[0]
                
                # После предсказания "разворачиваем" тензоры латентного изображения обратно
                model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=latents.shape[2] * vae_scale_factor,
                    width=latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # Вычисляем целевую величину как разницу между шумом и ground_trues
                # (noise играют роль ε, а задача — научить модель предсказывать этот шум)
                target = noise - ground_trues
                
                # Получаем весовые коэффициенты для SD3, чтобы масштабировать вклад каждого тензора
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=WEIGHTING_SCHEME,
                    sigmas=sigmas
                )
                
                # Вычисляем регуляционный MSE-loss со взвешиванием по каждому примеру по batch'у
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )   
                loss = loss.mean()

                # Обратное распространение ошибки
                accelerator.backward(loss)
                step_loss = loss.detach().item()

                # Освобождаем память от больших объектов
                del loss, latents, target, model_pred,  timesteps,  bsz, noise, noisy_model_input

                # Если синхронное обновление градиентов — выполняем обрезку (clipping)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, MAX_GRAD_NORM)

                # Перекладываем модель обратно на устройство (GPU), если она временно выгружалась
                transformer.to(accelerator.device)

                # Шаг оптимизатора + обновление lr
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Обновляем глобальный шаг, если были синхронизации
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                # Получаем текущий LR — для логов
                lr = lr_scheduler.get_last_lr()[0]
                lr_name = "lr"
                if OPTIMIZER_NAME == "prodigy":
                    if resume_step>0 and resume_step == global_step:
                        lr = 0
                    else:
                        # У продиги learning rate хранится в двух параметрах: d * lr
                        lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                    lr_name = "lr/d*lr"

                # Логируем значения: шаговую потерю, lr, эпоху
                logs = {"step_loss": step_loss, lr_name: lr, "epoch": epoch}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                
                # Прерываем обучение, если достигли максимального числа шагов
                if global_step >= max_train_steps:
                    break
                del step_loss

                # Очистка памяти
                gc.collect()
                torch.cuda.empty_cache()

                # Сохраняем состояние модели через определённые интервалы шагов
                if global_step % SAVE_MODEL_STEPS == 0 and SAVE_MODEL_STEPS > 0:
                    # Проверяем, является ли текущий процесс основным (для предотвращения избыточных сохранений в многопроцессорной среде)
                    if accelerator.is_main_process:
                        save_path = os.path.join(OUTPUT_DIR, f"{SAVE_NAME}-{epoch}-{global_step}")
                        accelerator.save_state(save_path)
                        logging.info(f"Состояние сохранено в {save_path}")
                
                # Проводим валидацию через определённые интервалы шагов, если указаны метаданные валидации
                if global_step % SAVE_MODEL_STEPS == 0 and SAVE_MODEL_STEPS > 0 and os.path.exists(val_metadata_path):
                    # Сохраняем состояние генераторов случайных чисел перед валидацией
                    before_state = torch.random.get_rng_state()
                    np_seed = abs(int(SEED)) if SEED is not None else np.random.seed()
                    py_state = python_get_rng_state()
                    with torch.no_grad():
                        # Разворачиваем модель для доступа к её параметрам
                        transformer = unwrap_model(transformer, accelerator)
                        # Фиксируем состояние генераторов случайных чисел для воспроизводимости
                        np.random.seed(VAL_SEED)
                        torch.manual_seed(VAL_SEED)
                        dataloader_generator = torch.Generator()
                        dataloader_generator.manual_seed(VAL_SEED)
                        # Устанавливаем детерминированный режим для cuDNN
                        torch.backends.cudnn.deterministic = True
                        
                        # Загружаем метаданные валидации
                        validation_datarows = []
                        with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                            validation_datarows = json.loads(readfile.read())
                        
                        # Устанавливаем датасет для валидации
                        if len(validation_datarows)>0:
                            # Подготавливаем датасет
                            validation_dataset = CachedMaskedPairsDataset(validation_datarows,conditional_dropout_percent=0)                         
                            batch_size  = 1
                            val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size, drop_last=True)

                            # Инициализируем DataLoader с использованием BucketBatchSampler
                            val_dataloader = torch.utils.data.DataLoader(
                                validation_dataset,
                                batch_sampler=val_batch_sampler, # Используем кастомный sampler для группировки по длине
                                collate_fn=collate_fn,
                                num_workers=NUM_WORKERS
                            )

                            logging.info("Начинаем вычисление потерь на валидации (val_loss)")
                            
                            # Счетчики валидации
                            total_loss = 0.0
                            num_batches = len(val_dataloader)

                            # Валидация
                            if num_batches == 0:
                                # Если нет данных для валидации — пропускаем
                                logging.info("Нет данных для валидации, пропускаем.")
                            else:
                                # Перебираем батчи с индикатором tqdm
                                enumerate_val_dataloader = enumerate(val_dataloader)
                                for i, batch in tqdm(enumerate_val_dataloader,position=1):                                    
                                    # Очистка памяти
                                    gc.collect()
                                    torch.cuda.empty_cache()

                                    # Перемещение всех тензоров батча на устройство и масштабирование
                                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                                    txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                                    ground_trues = batch["ground_true"].to(accelerator.device)
                                    factual_images = batch["factual_image"].to(accelerator.device)
                                    factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                                    factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)

                                    # Масштабирование latent-изображений через VAE-параметры
                                    factual_images = (factual_images - vae_config_shift_factor) * vae_config_scaling_factor
                                    factual_images = factual_images.to(dtype=weight_dtype)

                                    # Масштабируем ground truth
                                    ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                                    ground_trues = ground_trues.to(dtype=weight_dtype)
                                    
                                    # Регуляризационный выбор: случайно заменяем latents
                                    latents = factual_images
                                    if random.random() < REG_RATIO:
                                        prompt_embeds = torch.zeros_like(prompt_embeds).to(accelerator.device)
                                        pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds).to(accelerator.device)
                                        latents = ground_trues
                                    
                                    # Подготовка заглушек text_ids
                                    text_ids = torch.zeros(
                                        prompt_embeds.shape[1], 3
                                    ).to(device=accelerator.device, dtype=weight_dtype)
                                    
                                    # Вычисление VAE коэффициента масштабирования
                                    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                                    # Подготовка идентификаторов латентов изображений
                                    latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                                        latents.shape[0],
                                        latents.shape[2] // 2,
                                        latents.shape[3] // 2,
                                        accelerator.device,
                                        weight_dtype,
                                    )
                                    
                                    # Подготовка шума
                                    noise = torch.randn_like(latents) + NOISE_OFFSET * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                                    bsz = latents.shape[0]
                                    
                                    # Сэмплирование временных шагов
                                    u = compute_density_for_timestep_sampling(
                                        weighting_scheme=WEIGHTING_SCHEME,
                                        batch_size=bsz,
                                        logit_mean=LOGIT_MEAN,
                                        logit_std=LOGIT_STD,
                                        mode_scale=MODE_SCALE,
                                    )
                                    indices = (u * noise_scheduler_copy.config["num_train_timesteps"]).long()
                                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                                    
                                    # Добавление шума в latent-изображение
                                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                                    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                                    
                                   # Упаковка noisy_latents и масок
                                    packed_noisy_latents = FluxFillPipeline._pack_latents(
                                        noisy_model_input,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=latents.shape[1],
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )

                                    # Упаковка factual_image
                                    packed_factual_image_masks = FluxFillPipeline._pack_latents(
                                        factual_image_masks,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=vae_scale_factor * vae_scale_factor,
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )
                                    # Упаковка factual_image
                                    packed_factual_image_masked_images = FluxFillPipeline._pack_latents(
                                        factual_image_masked_images,
                                        batch_size=latents.shape[0],
                                        num_channels_latents=latents.shape[1],
                                        height=latents.shape[2],
                                        width=latents.shape[3],
                                    )
                                    
                                    # Сборка входа для модели: noisy_latents + masked_images
                                    masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)
                                    cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                                    
                                    # Guidance, если включён
                                    if handle_guidance:
                                        guidance = torch.tensor([GUIDANCE_SCALE], device=accelerator.device)
                                        guidance = guidance.expand(latents.shape[0])
                                    else:
                                        guidance = None
                                    
                                    with accelerator.autocast():
                                        # Предсказание шума
                                        model_pred = transformer(
                                            hidden_states=cat_model_input,
                                            timestep=timesteps / 1000,
                                            guidance=guidance,
                                            pooled_projections=pooled_prompt_embeds,
                                            encoder_hidden_states=prompt_embeds,
                                            txt_ids=text_ids,
                                            img_ids=latent_image_ids,
                                            return_dict=False,
                                            joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                                        )[0]
                                    
                                    # Распаковка латентов
                                    model_pred = FluxFillPipeline._unpack_latents(
                                        model_pred,
                                        height=latents.shape[2] * vae_scale_factor,
                                        width=latents.shape[3] * vae_scale_factor,
                                        vae_scale_factor=vae_scale_factor,
                                    )

                                    # Вычисляем вес ошибки и полную ошибку для данного батча
                                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=WEIGHTING_SCHEME, sigmas=sigmas)

                                    # Получение текущих метрик
                                    target = noise - ground_trues
                                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=WEIGHTING_SCHEME, sigmas=sigmas)
                                    
                                    # Вычисление потерь
                                    loss = torch.mean(
                                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                        1,
                                    )
                                    loss = loss.mean()
                                    total_loss+=loss.detach()

                                    # Очистка памяти
                                    del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    
                                # Вычисляем среднюю потерю на валидации    
                                avg_loss = total_loss / num_batches
                                
                                # Получаем текущий LR для логов
                                lr = lr_scheduler.get_last_lr()[0]
                                lr_name = "val_lr"
                                if OPTIMIZER_NAME == "prodigy":
                                    lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                                    lr_name = "val_lr lr/d*lr"
                                
                                # Собираем лог
                                logs = {"val_loss": avg_loss, lr_name: lr, "epoch": epoch}
                                logging.info(logs)

                                # Обновляем прогресс-бар текущими значениями
                                progress_bar.set_postfix(**logs)

                                # Логируем метрики через accelerate
                                accelerator.log(logs, step=global_step)

                                # Освобождаем память
                                del num_batches, avg_loss, total_loss

                            # Освобождаем память
                            del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
                            gc.collect()
                            torch.cuda.empty_cache()
                            logging.info("Конец валидации (End val_loss)")
                        
                    # Восстанавливаем генераторы случайных чисел до состояния перед валидацией
                    np.random.seed(np_seed)
                    torch.random.set_rng_state(before_state)

                    # Имеем состояние Python‑генератора: version, state, gauss
                    torch.backends.cudnn.deterministic = False
                    version, state, gauss = py_state
                    python_set_rng_state((version, tuple(state), gauss))
            
                    # Освобождаем память
                    gc.collect()
                    torch.cuda.empty_cache()

        # Валидация
        # TODO: Выделить в отдельный блок
        
        # Пропустить, если шаг обучения меньше заданного
        if global_step < SKIP_STEP:
            continue
         
        # Сохраняем генераторы случайных чисел до начала валидации
        before_state = torch.random.get_rng_state()
        np_seed = abs(int(SEED)) if SEED is not None else np.random.seed()
        py_state = python_get_rng_state()
        
        # Условие сохранения модели
        if (epoch >= SKIP_EPOCH and epoch % SAVE_MODEL_EPOCHS == 0) or epoch == NUM_TRAIN_EPOCHS - 1 or (global_step % SAVE_MODEL_STEPS == 0 and SAVE_MODEL_STEPS > 0):
            # accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(OUTPUT_DIR, f"{SAVE_NAME}-{epoch}-{global_step}")
                accelerator.save_state(save_path)
                logging.info(f"Состояние сохранено в {save_path}")
        
        # Проверка необходимости валидации (и существования пути к данным)
        if ((epoch >= SKIP_EPOCH and epoch % VALIDATION_EPOCHS == 0) or epoch == NUM_TRAIN_EPOCHS - 1 or (global_step % SAVE_MODEL_STEPS == 0 and SAVE_MODEL_STEPS > 0)) and os.path.exists(val_metadata_path):
            with torch.no_grad():
                # Оптимизация памяти через accelerate
                transformer = unwrap_model(transformer, accelerator)
                # Фиксируем генераторы случайных чисел для воспроизводимости
                np.random.seed(VAL_SEED)
                torch.manual_seed(VAL_SEED)
                dataloader_generator = torch.Generator()
                dataloader_generator.manual_seed(VAL_SEED)
                torch.backends.cudnn.deterministic = True
                
                # Получение метаинформации о валидационном наборе
                validation_datarows = []
                with open(val_metadata_path, "r", encoding='utf-8') as readfile:
                    validation_datarows = json.loads(readfile.read())
                
                # Подготовка датасета и валидация
                if len(validation_datarows)>0:
                    # Инициализируем датасет
                    validation_dataset = CachedMaskedPairsDataset(validation_datarows,conditional_dropout_percent=0)

                    # Размер батча
                    batch_size  = 1
                    val_batch_sampler = BucketBatchSampler(validation_dataset, batch_size=batch_size, drop_last=True)

                    # DataLoader с поддержкой BucketBatchSampler
                    val_dataloader = torch.utils.data.DataLoader(
                        validation_dataset,
                        batch_sampler=val_batch_sampler, #use bucket_batch_sampler instead of shuffle
                        collate_fn=collate_fn,
                        num_workers=NUM_WORKERS
                    )

                    logging.info("Начало валидации (val_loss)")
                    
                    # Счетчики потерь
                    total_loss = 0.0
                    num_batches = len(val_dataloader)

                    # Валидация
                    if num_batches == 0:
                        # Если нет данных для валидации — пропускаем
                        logging.info("Нет данных для валидации, пропускаем.")
                    else:
                        # Проход по валидационным батчам
                        enumerate_val_dataloader = enumerate(val_dataloader)
                        for i, batch in tqdm(enumerate_val_dataloader,position=1):                            
                            # Очистка памяти
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Перенос данных на устройство
                            prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                            txt_attention_masks = batch["txt_attention_masks"].to(accelerator.device)
                            ground_trues = batch["ground_true"].to(accelerator.device)
                            factual_images = batch["factual_image"].to(accelerator.device)
                            factual_image_masks = batch["factual_image_mask"].to(accelerator.device)
                            factual_image_masked_images = batch["factual_image_masked_image"].to(accelerator.device)
                            
                            # Масштабируем ground truth
                            ground_trues = (ground_trues - vae_config_shift_factor) * vae_config_scaling_factor
                            ground_trues = ground_trues.to(dtype=weight_dtype)
                            
                            # Заглушка текстовых ID
                            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)

                            # Латенты 
                            latents = factual_images
                            latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                            latents = latents.to(dtype=weight_dtype)

                            # Вычисление коэффициента масштабирования
                            vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                            # ID для латентных изображений
                            latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                                latents.shape[0],
                                latents.shape[2] // 2,
                                latents.shape[3] // 2,
                                accelerator.device,
                                weight_dtype,
                            )
                            
                            # Шум для входов
                            noise = torch.randn_like(latents) + NOISE_OFFSET * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(accelerator.device)
                            bsz = latents.shape[0]

                            # Сэмплинг временных шагов
                            if noise_scheduler_copy.timesteps is not None:
                                u = compute_density_for_timestep_sampling(
                                    weighting_scheme=WEIGHTING_SCHEME,
                                    batch_size=bsz,
                                    logit_mean=LOGIT_MEAN,
                                    logit_std=LOGIT_STD,
                                    mode_scale=MODE_SCALE,
                                )
                                indices = (u * noise_scheduler_copy.config["num_train_timesteps"]).long()
                                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)
                            
                            # Генерация входов с шумом
                            sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
                            
                            # Упаковка входов
                            packed_noisy_latents = FluxFillPipeline._pack_latents(
                                noisy_model_input,
                                batch_size=latents.shape[0],
                                num_channels_latents=latents.shape[1],
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            
                            # Упаковка factual_image
                            packed_factual_image_masks = FluxFillPipeline._pack_latents(
                                factual_image_masks,
                                batch_size=latents.shape[0],
                                num_channels_latents=vae_scale_factor * vae_scale_factor,
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            # Упаковка factual_image
                            packed_factual_image_masked_images = FluxFillPipeline._pack_latents(
                                factual_image_masked_images,
                                batch_size=latents.shape[0],
                                num_channels_latents=latents.shape[1],
                                height=latents.shape[2],
                                width=latents.shape[3],
                            )
                            
                            # Сборка входа для модели: noisy_latents + masked_images
                            masked_image_latents = torch.cat((packed_factual_image_masked_images, packed_factual_image_masks), dim=-1)
                            cat_model_input = torch.cat((packed_noisy_latents, masked_image_latents), dim=2)
                           
                            # Guidance, если включён
                            if handle_guidance:
                                guidance = torch.tensor([GUIDANCE_SCALE], device=accelerator.device)
                                guidance = guidance.expand(latents.shape[0])
                            else:
                                guidance = None
                            
                            with accelerator.autocast():
                                # Предсказание остаточного шума
                                model_pred = transformer(
                                    hidden_states=cat_model_input,
                                    timestep=timesteps / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds,
                                    encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                    joint_attention_kwargs = {'attention_mask': txt_attention_masks},
                                )[0]
                            
                            # Распаковка латентов
                            model_pred = FluxFillPipeline._unpack_latents(
                                model_pred,
                                height=latents.shape[2] * vae_scale_factor,
                                width=latents.shape[3] * vae_scale_factor,
                                vae_scale_factor=vae_scale_factor,
                            )

                            # Вычисляем loss и аккумулируем
                            target = noise - ground_trues                           
                            weighting = compute_loss_weighting_for_sd3(weighting_scheme=WEIGHTING_SCHEME, sigmas=sigmas)
                            
                            # Вычисление потерь
                            loss = torch.mean(
                                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                                1,
                            )
                            loss = loss.mean()
                            total_loss+=loss.detach()

                            # Освобождение памяти
                            del latents, target, loss, model_pred,  timesteps,  bsz, noise, packed_noisy_latents
                            gc.collect()
                            torch.cuda.empty_cache()

                        # Средняя валидационная потеря    
                        avg_loss = total_loss / num_batches
                        
                        # Текущий LR для логов
                        lr = lr_scheduler.get_last_lr()[0]
                        lr_name = "val_lr"
                        if OPTIMIZER_NAME == "prodigy":
                            lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                            lr_name = "val_lr lr/d*lr"
                        logs = {"val_loss": avg_loss, lr_name: lr, "epoch": epoch}
                        
                        # Выводим результат
                        logging.info(logs)
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)

                        # Освобождаем память
                        del num_batches, avg_loss, total_loss

                    # Очистка после валидации    
                    del validation_datarows, validation_dataset, val_batch_sampler, val_dataloader
                    gc.collect()
                    torch.cuda.empty_cache()
                    logging.info("Конец валидации (End val_loss)")
        
        # Восстанавливаем состояния генераторов случайных чисел после валидации
        np.random.seed(np_seed)
        torch.random.set_rng_state(before_state)
        torch.backends.cudnn.deterministic = False
        version, state, gauss = py_state
        python_set_rng_state((version, tuple(state), gauss))
        
        # Очистка памяти
        gc.collect()
        torch.cuda.empty_cache()
        
    # Завершение обучения
    accelerator.end_training()
    print(f"Сохранено в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()