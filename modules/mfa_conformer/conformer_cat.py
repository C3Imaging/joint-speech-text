import torch
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling, BatchNorm1d

from modules.mfa_conformer.wenet.transformer.encoder_cat import ConformerEncoder


class Conformer(torch.nn.Module):
    def __init__(
        self,
        n_mels=80,
        num_blocks=6,
        output_size=256,
        embedding_dim=192,
        input_layer="conv2d2",
        pos_enc_layer_type="rel_pos",
    ):
        super(Conformer, self).__init__()
        self.conformer = ConformerEncoder(
            input_size=n_mels,
            num_blocks=num_blocks,
            output_size=output_size,
            input_layer=input_layer,
            pos_enc_layer_type=pos_enc_layer_type,
        )
        self.pooling = AttentiveStatisticsPooling(output_size * num_blocks)
        self.bn = BatchNorm1d(input_size=output_size * num_blocks * 2)
        self.fc = torch.nn.Linear(output_size * num_blocks * 2, embedding_dim)

    def forward(self, feat, lens):
        x, masks = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x


def conformer_cat(
    n_mels=80,
    num_blocks=6,
    output_size=256,
    embedding_dim=192,
    input_layer="conv2d",
    pos_enc_layer_type="rel_pos",
):
    model = Conformer(
        n_mels=n_mels,
        num_blocks=num_blocks,
        output_size=output_size,
        embedding_dim=embedding_dim,
        input_layer=input_layer,
        pos_enc_layer_type=pos_enc_layer_type,
    )
    return model
