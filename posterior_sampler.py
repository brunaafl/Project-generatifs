import torch
import matplotlib.pyplot as plt
import yaml
import argparse

from data import get_dataset, get_dataloader
from denoisers.DnCNN.get_dncnn import create_model_DnCNN
from denoisers.hierarquicalVAE.get_VAE import create_model_nvae
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.img_utils import clear_color
from tasks import create_operator
from gibbs_sampler import GibbsSampler

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default="configs/data_config.yaml")
    parser.add_argument('--model_config', type=str, default="configs/model_config.yaml")
    parser.add_argument('--model_type', type=str, default="DDMP")
    parser.add_argument('--diffusion_config', type=str, default="configs/diffusion_config.yaml")
    # Change here depending on the type of inverse problem
    parser.add_argument('--operator_config', type=str, default="configs/gaussianblur_config.yaml")
    parser.add_argument('--gibbs_config', type=str, default="configs/gibbs_config.yaml")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    # Get configs
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    operator_config = load_yaml(args.operator_config)  # the operator config
    gibbs_config = load_yaml(args.gibbs_config)
    # Load model
    print("Load model")
    if args.model_type == "DDMP":
        model = create_model(**model_config)
    elif args.model_type == "DnCNN":
        model = create_model_DnCNN(**model_config)
    elif args.model_type == "NVAE":
        model = create_model_nvae(**model_config)
    model = model.to(device)
    model.eval()
    # Load diffusion sampler
    print("Load sampler")
    sampler = create_sampler(**diffusion_config)
    # Load operator
    print("Load operator")
    operator = create_operator(**operator_config, device=device)
    print("Operator name: {}".format(operator.display_name))
    # Load dataset
    print("Load dataset")
    dataset = get_dataset(**data_config)
    num_test_images = len(dataset)
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    for i, img in enumerate(dataloader):
        X = img.to(device)
        #sigma = X.mean(dim=1)
        #sigma = sigma.to(device)
        Y = operator.forward(X)
        #sigma = estimate_sigma(Y.numpy(), channel_axis=0, average_sigmas=True)
        sigma = torch.tensor(0.05).to(device)
        Y = Y + sigma*torch.randn(X.shape).to(device)

        # Plot noisy image
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(clear_color(Y))
        ax.set_title('Noisy image')
        ax.axis('off')
        plt.savefig(f"results/{i}_noisy.png", dpi=200, bbox_inches='tight')

        plt.show()

        gibbs = GibbsSampler(
                             Y=Y, 
                             sigma=sigma, 
                             operator=operator, 
                             sampler=sampler, 
                             model=model,
                             model_type=args.model_type,
                             device=device, 
                             **gibbs_config)

        X_MC, Z_MC = gibbs.run()

        fig, axes = plt.subplots(1, 4, figsize=(20, 20))

        axes[0].imshow(clear_color(X))
        axes[0].set_title('True image')
        axes[0].axis('off');
        
        axes[1].imshow(clear_color(Y))
        axes[1].set_title('Noisy image')
        axes[1].axis('off');
        
        axes[2].imshow(clear_color(torch.mean(Z_MC[:,:,:,gibbs_config["N_bi"]:gibbs_config["N_MC"]], axis=-1)));
        axes[2].set_title('Z Reconstructed image')
        axes[2].axis('off');
        
        axes[3].imshow(clear_color(torch.mean(X_MC[:,:,:,gibbs_config["N_bi"]:gibbs_config["N_MC"]], axis=-1)));
        axes[3].set_title('X Reconstructed image')
        axes[3].axis('off');

        plt.savefig(f"results/{i}_celeba_test.png", dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    main()
