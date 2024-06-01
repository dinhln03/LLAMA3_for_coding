# -*- coding: utf-8 -*-
import keras.engine.training
from typing import Callable
from typing import Tuple
from typing import List
from typing import Union
from util_types import types_of_loco
from network_model.distillation.distillation_model_builder import DistllationModelIncubator
from keras.optimizers import Optimizer, SGD
from network_model.build_model import builder, builder_with_merge
from model_merger.keras.merge_model import ModelMerger
from keras.layers import Concatenate
from keras.callbacks import Callback
from model_merger.keras.siamese import SiameseBuilder
from model_merger.keras.proc.calculator import calc_l1_norm
from model_merger.pytorch.proc.distance.calculator import L1Norm
import torch.nn
from network_model.builder import keras_builder, pytorch_builder

DLModel = Union[keras.engine.training.Model, torch.nn.Module]
ModelBuilderResult = Union[DLModel, Tuple[DLModel, List[Callback]], Tuple[DLModel, Callable[[str], Callback]]]

ModelBuilder = Union[Callable[[int], ModelBuilderResult],
                     Callable[[Union[str, Tuple[str, str]]], ModelBuilderResult],
                     DistllationModelIncubator]


def init_input_image(size: types_of_loco.input_img_size):
    def builder_of_generator(class_num: int, channels: int = 1, optimizer: Optimizer = SGD()):
        """
        Ganのgenerator部を作成する
        :param class_num
        :param channels:色の出力変数（白黒画像なら1）
        :param optimizer: 2次元の畳み込みウィンドウの幅と高さ 整数なら縦横比同じに
        :return: discriminator部のモデル
        """
        return builder(class_num, size, channels, optimizer)
    return builder_of_generator


def build_wrapper(img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  model_name: str = "model1",
                  optimizer: Optimizer = SGD()) -> Union[ModelBuilder, pytorch_builder.PytorchModelBuilder]:
    """
    モデル生成をする関数を返す
    交差検証をかける際のラッパーとして使う
    :param img_size:
    :param channels:
    :param model_name:
    :param optimizer:
    :return:
    """
    if callable(optimizer):
        return pytorch_builder.PytorchModelBuilder(img_size=img_size,
                                                   channels=channels,
                                                   model_name=model_name,
                                                   opt_builder=optimizer)
    return keras_builder.build_wrapper(img_size, channels, model_name, optimizer)


def build_with_merge_wrapper(base_model_num: int,
                             img_size: types_of_loco.input_img_size = 28,
                             channels: int = 3,
                             model_name: str = "model1",
                             optimizer: Optimizer = SGD(),
                             model_merger: ModelMerger = ModelMerger(Concatenate(),
                                                                     metrics=['accuracy'])) -> ModelBuilder:
    return lambda class_num: builder_with_merge(base_model_num,
                                                model_merger,
                                                class_num,
                                                img_size,
                                                channels,
                                                optimizer,
                                                model_name)


def build_siamese(q: float,
                  img_size: types_of_loco.input_img_size = 28,
                  channels: int = 3,
                  model_name: str = "model1",
                  optimizer=SGD(),
                  loss_func=None,
                  calc_distance=None,
                  save_best_only=True,
                  save_weights_only=False,
                  save_base_filepath: str = None,
                  is_inceptionv3: bool = False,
                  decide_dataset_generator=None,
                  nearest_data_ave_num=1,
                  will_calc_real_data_train=False):
    use_distance = calc_distance
    if use_distance is None:
        use_distance = L1Norm() if callable(optimizer) else calc_l1_norm

    def build(class_num: int):
        base_model = builder(class_num, img_size, channels, optimizer, model_name)
        shame_builder = SiameseBuilder(base_model)
        return shame_builder.build_shame_trainer_for_classifivation(q,
                                                                    optimizer,
                                                                    loss_func,
                                                                    use_distance,
                                                                    save_base_filepath,
                                                                    save_best_only,
                                                                    save_weights_only
                                                                    )
    print(type(decide_dataset_generator))
    return pytorch_builder.PytorchSiameseModelBuilder(q,
                                                      img_size,
                                                      channels,
                                                      model_name,
                                                      optimizer,
                                                      loss_func,
                                                      use_distance,
                                                      is_inceptionv3,
                                                      decide_dataset_generator,
                                                      nearest_data_ave_num,
                                                      will_calc_real_data_train) if callable(optimizer) else build
