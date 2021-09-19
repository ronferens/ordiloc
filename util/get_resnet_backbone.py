import torchvision.models as models
import torch
import argparse

if __name__ == '__main__':
    """
    Downloading and saving the requested ResNet backbone to the 'models' folder
    The resnet_type argument Supports: resnet34, resnet50, resnet152
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name", help="The requested ResNet model (e.g. resnet34, resnet50, resnet152")
    args = arg_parser.parse_args()

    if args.model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
    else:
        raise ValueError('Unsupported model name - {}'.format(args.model_name))

    # Removing the last two layer
    backbone_model = torch.nn.Sequential(*(list(model.children())[:-2]))

    # Saving the model
    torch.save(backbone_model, '../models/backbones/{}.pth'.format(args.model_name))
