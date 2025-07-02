from contour_diffusion.Encoder import TransformerModel
import torch

model = TransformerModel(2, 185, 128, 128, 8, 8).to('cuda')
model.load_state_dict(torch.load('transformer_encoder.pt'))
encoder = model.encoder
encoder.half()
torch.save(encoder.state_dict(), 'encoder.pt')