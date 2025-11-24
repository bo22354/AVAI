# AVAI
Generator Class:
- Implements SRResNet (Super-Resolutiom Residual Network)
- Designed to take small image and exapnd it (filling in details)
- Residual Block is to implement residual connects allowing for it to just learn the difference/error compared to whole image at that layer



Structure for Method 1 report:
2. Selected Methods & Justification
    Goal: Contrast your two approaches. You are picking one method for "Perceptual Quality" (GAN) and one for "Flexibility" (INR).

2.1. Method 1: Generative Adversarial Networks (SRGAN)
    Focus: Solving the "over-smoothing" problem of standard CNNs.

    Rationale for Selection:
       - Standard CNNs trained with MSE (L2 Loss) suffer from "regression to the mean." They average out all possible high-frequency details to minimize error, resulting in blurry images.
       - You selected a GAN to break this limitation. The Adversarial Loss forces the model to generate details that are indistinguishable from real images, effectively hallucinating realistic textures (grass, hair, fabric) that MSE-based models miss.

    Architectural Justification (The Code We Wrote):
        - Generator (SRResNet):
            - Used Residual Blocks to allow the network to learn the difference (residual) between LR and HR, rather than learning the image from scratch. This stabilizes deep network training.
            - Used PixelShuffle instead of Deconvolution to avoid checkerboard artifacts and improve computational efficiency.
        - Discriminator (VGG-Style):
            - Used Strided Convolutions rather than MaxPooling to learn optimal downsampling filters for texture discrimination.
            - Used LeakyReLU to prevent gradient saturation ("dying ReLU").
        - Loss Function Strategy:
            - Explain the shift from Pixel Loss (MSE) to Perceptual Loss (VGG Feature Loss + Adversarial Loss). You care about semantic similarity, not just pixel-grid alignment.
