import torch
from torch import nn

from .blocks import BasicBlock2D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(
        self,
        block,
        num_blocks,
        width,
        in_channels=3,
        recall=True,
        group_norm=False,
        output_space=False,
        **kwargs,
    ):
        super().__init__()

        self.recall = recall
        self.output_space = output_space    
        self.original_width = int(width)  # frozen version for input channels
        self.width = self.original_width  # mutable version for layer construction
        self.group_norm = group_norm
        self.in_channels = in_channels

        proj_conv = nn.Conv2d(in_channels, self.original_width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        # added this
        conv_recall_output = nn.Conv2d(self.original_width + in_channels + 2, self.original_width, kernel_size=3,
                                       stride=1, padding=1, bias=False)
        conv_recall_only = nn.Conv2d(self.original_width + in_channels, self.original_width, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        # added this
        conv_output_only = nn.Conv2d(self.original_width + 2, self.original_width, kernel_size=3,
                                     stride=1, padding=1, bias=False)

        if recall and output_space:
            recur_layers = [conv_recall_output]
        elif recall:
            recur_layers = [conv_recall_only]
        elif output_space:
            recur_layers = [conv_output_only]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, self.original_width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(self.width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, prev_output=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        batch_size, _, H, W = x.size()
        all_outputs = torch.zeros((batch_size, iters_to_do, 2, H, W)).to(x.device)

        for i in range(iters_to_do):
            concat_inputs = [interim_thought]

            if self.recall:
                concat_inputs.append(x)
                
            # changed from here
            if self.output_space:
                if prev_output is None:
                    prev_output = torch.zeros((batch_size, 2, H, W)).to(x.device)
                concat_inputs.append(prev_output)

            recur_input = torch.cat(concat_inputs, dim=1)
            interim_thought = self.recur_block(recur_input)
            out = self.head(interim_thought)
            all_outputs[:, i] = out
            prev_output = out.detach() 
            
        return out, interim_thought, all_outputs


def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, output_space=False)


def dt_net_recall_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, output_space=False)


def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True, output_space=False)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True, output_space=False)


def dt_net_outputspace_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, output_space=True)


def dt_net_outputspace_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True, output_space=True)


def dt_net_recall_outputspace_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, output_space=True)


def dt_net_recall_outputspace_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True, output_space=True)
