from vit_pytorch import ViT, Dino
import torch


class dino(torch.nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()

        self.model = ViT(
            image_size=496,
            patch_size=16,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048
        )

        ######################################################################### v = Recorder(model)

        self.learner = Dino(
            self.model,
            image_size=496,
            hidden_layer='to_latent',  # hidden layer name or index, from which to extract the embedding
            projection_hidden_size=496,  # projector network hidden dimension
            projection_layers=4,  # number of layers in projection network
            num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
            student_temp=0.9,  # student temperature
            teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay=0.9,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay=0.9,
            # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        )

    def forward(self, x):
        # print(f'input {x.shape}')
        x = self.learner(x)
        # print(f'output {x.shape}')
        return x
