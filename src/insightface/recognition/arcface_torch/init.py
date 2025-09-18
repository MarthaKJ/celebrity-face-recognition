from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    if name == "r34":
        return iresnet34(False, **kwargs)
    if name == "r50":
        return iresnet50(False, **kwargs)
    if name == "r100":
        return iresnet100(False, **kwargs)
    if name == "r200":
        return iresnet200(False, **kwargs)
    if name == "r2060":
        from .iresnet2060 import iresnet2060

        return iresnet2060(False, **kwargs)

    if name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    if name == "mbf_large":
        from .mobilefacenet import get_mbf_large

        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    if name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=256,
            depth=12,
            num_heads=8,
            drop_path_rate=0.1,
            norm_layer="ln",
            mask_ratio=0.1,
        )

    if name == "vit_t_dp005_mask0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=256,
            depth=12,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.0,
        )

    if name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=512,
            depth=12,
            num_heads=8,
            drop_path_rate=0.1,
            norm_layer="ln",
            mask_ratio=0.1,
        )

    if name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=512,
            depth=12,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.0,
        )

    if name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=512,
            depth=24,
            num_heads=8,
            drop_path_rate=0.1,
            norm_layer="ln",
            mask_ratio=0.1,
            using_checkpoint=True,
        )

    if name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=512,
            depth=24,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.05,
            using_checkpoint=True,
        )

    if name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=768,
            depth=24,
            num_heads=8,
            drop_path_rate=0.05,
            norm_layer="ln",
            mask_ratio=0.05,
            using_checkpoint=True,
        )

    if name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer

        return VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=num_features,
            embed_dim=1024,
            depth=48,
            num_heads=8,
            drop_path_rate=0.1,
            norm_layer="ln",
            mask_ratio=0,
            using_checkpoint=True,
        )

    raise ValueError
