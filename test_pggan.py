import torch
import torchvision
import torchvision.transforms as transforms


def toPIL(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)


def main():
    use_gpu = True if torch.cuda.is_available() else False

    # trained on high-quality celebrity faces "celebA" dataset
    # this model outputs 512 x 512 pixel images
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                           'PGAN', model_name='celebAHQ-512',
                           pretrained=True, useGPU=use_gpu)

    num_images = 1
    noise, _ = model.buildNoiseData(num_images)
    print(noise.size())
    with torch.no_grad():
        generated_images = model.test(noise)

    grid = torchvision.utils.make_grid(generated_images)
    grid.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    grid_img = toPIL(grid)
    print(grid.size())
    grid_img.show()


if __name__ == "__main__":
    main()
