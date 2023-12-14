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

## Acknowledgments
- Original repository: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
- Code parts: [hmorimitsu/ptlflow](https://github.com/hmorimitsu/ptlflow/tree/main)

**If you plan to use RAFT in your work, please cite the original paper:**
```
@misc{teed2020raft,
      title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow}, 
      author={Zachary Teed and Jia Deng},
      year={2020},
      eprint={2003.12039},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
