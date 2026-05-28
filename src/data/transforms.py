#src/data/transforms.py

"""
Transforms MONAI per MRI cerebrali T1 skull-stripped.
Separato da dataset.py per coerenza con MAISI (scripts/transforms.py).
"""

from typing import Optional
import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
)


def get_vae_transforms(
    patch_size:tuple=(64, 64, 64),
    val_patch_size:Optional[tuple]=None,
    is_train:bool=True,
    random_aug:bool=True,
    k:int=4,
    output_dtype:torch.dtype=torch.float32,
) -> Compose:
    """
    Transforms per il training/validazione del VAE.
    
    Pipeline:
    1. Caricamento NIfTI
    2. EnsureChannelFirst
    3. Orientamento RAS
    4. Resize/Pad a 256³
    5. Normalizzazione percentile [0, 99.5] → [0, 1]
    6. Augmentazioni MRI (solo training)
    7. SpatialPad + RandSpatialCrop a patch_size (solo training)
    8. EnsureType float32
    """
    common_transforms=[
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(256, 256, 256),
            mode="constant",
            constant_values=0,
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.0,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if is_train:
        aug_transforms=[]
        if random_aug:
            aug_transforms=[
                RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.0, 0.3)),
                RandGibbsNoised(keys=["image"], prob=0.3, alpha=(0.5, 1.0)),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2.0)),
                RandHistogramShiftd(keys=["image"], prob=0.05, num_control_points=10),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)),
                RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
                RandScaleIntensityd(keys=["image"], prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.05),
                RandZoomd(
                    keys=["image"],
                    prob=0.3,
                    min_zoom=0.7,
                    max_zoom=1.3,
                    keep_size=True,
                    mode="bilinear",
                ),
                RandRotated(
                    keys=["image"],
                    prob=0.3,
                    range_x=0.1,
                    range_y=0.1,
                    range_z=0.1,
                    keep_size=True,
                    mode="bilinear",
                ),
            ]

        crop_transforms=[
            SpatialPadd(keys=["image"], spatial_size=patch_size),
            RandSpatialCropd(
                keys=["image"],
                roi_size=patch_size,
                random_size=False,
                random_center=True,
            ),
        ]

        final_transforms=[EnsureTyped(keys=["image"], dtype=output_dtype)]
        return Compose(common_transforms + aug_transforms + crop_transforms + final_transforms)

    else:
        if val_patch_size is None:
            val_crop=[DivisiblePadd(keys=["image"], k=k)]
        else:
            val_crop=[ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=val_patch_size,
            )]
        final_transforms=[EnsureTyped(keys=["image"], dtype=output_dtype)]
        return Compose(common_transforms + val_crop + final_transforms)


def get_encoding_transforms()->Compose:
    """
    Transforms per l'encoding del dataset con il VAE trainato.
    Identiche alle VAE transforms ma senza augmentazioni e crop —
    encodiamo il volume intero 256³.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(256, 256, 256),
            mode="constant",
            constant_values=0,
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.0,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])