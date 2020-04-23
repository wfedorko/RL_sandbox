import torch

transpose_tuple=(2,0,1)

def preprocess_frame(frame):
    frame = frame.transpose(transpose_tuple)
    frame = torch.from_numpy(frame)
    frame = frame.to(dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame