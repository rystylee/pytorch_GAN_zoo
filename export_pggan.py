import torch
from hubconf import PGAN


def main():
    use_gpu = True if torch.cuda.is_available() else False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # trained on high-quality celebrity faces "celebA" dataset
    # this model outputs 512 x 512 pixel images
    model = PGAN(model_name='celebAHQ-512',
                 pretrained=True, useGPU=use_gpu)
    model = model.netG.to(device)
    model.eval()
    print(model)

    batch_size = 1
    z_dim = 512

    # Prepare input z vector
    noise_z = torch.ones(batch_size, z_dim).to(device)

    traced_script_module = torch.jit.trace(model, (noise_z))
    output = traced_script_module(noise_z)

    name = 'PGGAN_{}.pt'.format(512)
    traced_script_module.save(name)
    print('Succeed to save traced script module!')


if __name__ == "__main__":
    main()
