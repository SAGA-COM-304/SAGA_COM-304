import torch
from PIL import Image
from cosmos_tokenizer.image_lib import ImageTokenizer
from torchvision.transforms import ToTensor, Resize, ToPILImage

def tokenize_image(path, device, model_name) -> torch.Tensor:
    """
    Tokenizes an input image into a latent tensor representation using a specified
    pre-trained image tokenizer model. The function loads the image, processes it into
    a tensor, and utilizes a pre-trained encoder model to extract latent features.

    Arguments:
        path (str): Path to the input image file to be tokenized.
        device (torch.device): Device on which the computation will occur
            (e.g., "cpu" or "cuda").
        model_name (str): Name of the image tokenizer model to use. The model
            checkpoint is expected to reside in a specific directory for loading.

    Returns:
        torch.Tensor: A latent tensor representation of the input image.
    """
    image = Image.open(path).convert('RGB')
    input_tensor = ToTensor()(image).unsqueeze(0).to(device)
    # TODO: choose where we download the model
    # snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=f"checkpoints/{model_name}")
    encoder = ImageTokenizer(checkpoint_enc=f'checkpoints/{model_name}/encoder.jit')
    (latent, _) = encoder.encode(input_tensor)
    return latent


def detokenize_image(latent, model_name) -> Image.Image:
    """
    Detokenizes a latent image representation into a PIL Image using a specified model and
    device. The function leverages the decoder part of an image tokenizer model to convert
    latents back into image form.

    Parameters:
    latent : Tensor
        The latent tensor representation of an image to be decoded.
    model_name : str
        The name of the model to use for decoding, used to locate the model checkpoint.

    Returns:
    Image.Image
        A PIL Image obtained after decoding the latent representation.
    """
    decoder = ImageTokenizer(checkpoint_dec=f'checkpoints/{model_name}/decoder.jit')
    output_tensor = decoder.decode(latent)
    output_tensor = output_tensor.squeeze(0).cpu()
    to_pil = ToPILImage()
    image = to_pil(output_tensor)
    return image