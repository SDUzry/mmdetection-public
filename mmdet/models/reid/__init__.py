# Copyright (c) OpenMMLab. All rights reserved.
from .base_reid import BaseReID
from .fc_module import FcModule
from .gap import GlobalAveragePooling
from .linear_reid_head import LinearReIDHead
from .fc_head import FcHead

__all__ = ['BaseReID', 'GlobalAveragePooling', 'LinearReIDHead', 'FcModule', 'FcHead']