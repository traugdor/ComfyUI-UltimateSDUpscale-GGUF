from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

class Sampler:
    @staticmethod
    def sample(noise, guider, sampler, sigmas, latent):
        latent_copy = latent.copy()
        output, denoised = SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent_copy)
        return output

    @staticmethod
    def encode(image, vae):
        t = vae.encode(image[:,:,:,:3])
        return ({"samples":t})
