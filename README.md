# RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
Unofficial implementation in PyTorch of the optical flow model presented in "[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)"

## Quick Start
```
import torch
from raft import RAFT

batch_size = 2

frame_1 = torch.rand(batch_size, 3, 256, 256)
frame_2 = torch.rand(batch_size, 3, 256, 256)

optical_flow = RAFT(
                  small = False,
                  pretrained = "weights/raft-sintel.pth"
               )

optical_flow_estimate = optical_flow(frame_1, frame_2, iters = 20)
```
