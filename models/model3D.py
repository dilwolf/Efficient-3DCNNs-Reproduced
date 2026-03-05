import torch
from torch import nn
from models import shufflenetv2
from models.shufflenetv2 import get_fine_tuning_parameters


def load_model(opt):
    model = shufflenetv2.get_model(
        num_classes=opt.num_classes,
        sample_size=opt.input_size,
        width_mult=opt.width_mult
    )

    # Load pretrained weights
    if opt.pretrain_path:
        checkpoint = torch.load(
            opt.pretrain_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(checkpoint["state_dict"])

        # Replace classifier for fine-tuning
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(in_features, opt.n_finetune_classes)
        )

        parameters = get_fine_tuning_parameters(model, opt.ft_portion)
        return model, parameters

    return model, model.parameters()