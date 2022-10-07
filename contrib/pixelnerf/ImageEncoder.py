import jittor as jt
from jittor import nn
from jittor.models import resnet34


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet34(True)

    def execute(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)

        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = nn.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = jt.concat(latents, dim=1)
        return latents
