from data import Kermany_DataSet
import timm
import wandb
import os
from timm.models.swin_transformer import SwinTransformer
from utils import *
from res_models import *
from model_running import *
from convnext import convnext_base, convnext_large, convnext_xlarge
from dino_class import dino

hyperparameter_defaults = dict(
    epochs=1,
    seed=25,
    batch_size=2,
    learning_rate=1e-4,
    optimizer="adam",
    mom=0.9,
    weight_decay=0,
    architecture='dino',
    pretrain=False,
)

wandb.init(config=hyperparameter_defaults, project="Dino_Test")
config = wandb.config


def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    torch.manual_seed(hash("by removing stochasticity") % wandb.config.seed)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % wandb.config.seed)
    def_args = dot_dict({
        "train": ["../../../data/kermany/train"],
        "val": ["../../../data/kermany/val"],
        "test": ["../../../data/kermany/test"],
    })
    label_names = [
        "NORMAL",
        "CNV",
        "DME",
        "DRUSEN",
    ]
    return def_args, device, label_names


def Get_Model(config, device):
    model = None
    if config.architecture == "res18":
        model = Resnet18(4, pretrained=config.pretrain)
    elif config.architecture == "res50":
        model = Resnet50(4, pretrained=config.pretrain)
    elif config.architecture == "res101":
        model = Resnet101(4, pretrained=config.pretrain)
    elif config.architecture == "res152":
        model = Resnet152(4, pretrained=config.pretrain)
    elif config.architecture == "wide_resnet50_2":
        model = Wide_Resnet50_2(4, pretrained=config.pretrain)
    elif config.architecture == "wide_resnet101_2":
        model = Wide_Resnet101_2(4, pretrained=config.pretrain)

    if config.architecture[:3] == 'vit' or config.architecture[:4] == 'deit':
        model = timm.create_model(config.architecture, pretrained=config.pretrain, num_classes=4,
                                  img_size=(496, 512))

    if config.architecture[:4] == 'swin':
        w = int(config.architecture[4:])
        SwinTransformer(img_size=(496, 512), patch_size=4, in_chans=1, num_classes=4,
                        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                        window_size=w, mlp_ratio=4., qkv_bias=True,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, )

    if config.architecture[:4] == 'effi':
        model = timm.create_model(config.architecture, pretrained=config.pretrain, num_classes=4,
                                  )

    if config.architecture == 'convmixer_1536_20':
        model = timm.create_model(config.architecture, pretrained=config.pretrain, num_classes=4)
    elif config.architecture == 'convmixer_768_32':
        model = timm.create_model(config.architecture, pretrained=config.pretrain, num_classes=4)
    elif config.architecture == 'convmixer_1024_20_ks9_p14':
        model = timm.create_model(config.architecture, pretrained=config.pretrain, num_classes=4)

    if config.architecture == 'convnext_base':
        model = convnext_base(pretrained=config.pretrain, num_classes=4
                              )
    elif config.architecture == 'convnext_large':
        model = convnext_base(pretrained=config.pretrain, num_classes=4)
    elif config.architecture == 'convnext_xlarge':
        model = convnext_base(pretrained=config.pretrain, num_classes=4)

    if config.architecture == 'dino':
        model = dino(4, pretrained=config.pretrain)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    wandb.watch(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"Total Params": pytorch_total_params})
    wandb.log({"Trainable Params": pytorch_total_params_train})

    return model


def Handle_Data(def_args):
    if config.architecture == "dino":
        size = (496, 496)
    else:
        size = (496, 512)
    train_dataset = Kermany_DataSet(def_args.train[0], size=size)
    val_dataset = Kermany_DataSet(def_args.val[0], size=size)
    test_dataset = Kermany_DataSet(def_args.test[0], size=size)

    train_weights = make_weights_for_balanced_classes(train_dataset, [i for i in range(4)])
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               # shuffle=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)
    return test_loader, train_loader, val_loader


def Get_Optimizer(model):
    optimizer = None
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, momentum=config.mom,
                                        weight_decay=config.weight_decay)
    return optimizer


def main():
    def_args, device, label_names = init()

    print("gettin data")
    test_loader, train_loader, val_loader = Handle_Data(def_args)

    print("gettin model")
    print(f'architecture:{config.architecture}"')
    model = Get_Model(config, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Get_Optimizer(model)
    #########################################################################################################
    #                                                 TRAINING                                              #
    #########################################################################################################

    print("starting training:\n\n")
    Train(criterion, device, label_names, model, optimizer, train_loader, val_loader, config.epochs, test_loader)

    print("finita la comedia ")


if __name__ == '__main__':
    main()
    print("\n\n\n\nWe done did it mates")
