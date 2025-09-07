import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(
    image: np.ndarray,
    prob_map: np.ndarray,
    sxy_gaussian: int = 3,
    compat_gaussian: int = 3,
    sxy_bilateral: int = 40,
    srgb_bilateral: int = 15,
    compat_bilateral: int = 5,
    iterations: int = 5
) -> np.ndarray:

    h, w = prob_map.shape
    n_labels = 2

    # Prepare softmax probabilities
    probs = np.zeros((n_labels, h, w), dtype=np.float32)
    probs[0, :, :] = 1.0 - prob_map  # background
    probs[1, :, :] = prob_map        # change

    # Create DenseCRF model
    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(probs.reshape((n_labels, -1)))
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian term (smoothness)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)

    # Add pairwise Bilateral term (edge-preserving)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=image,
        compat=compat_bilateral
    )

    # Run inference
    Q = d.inference(iterations)
    preds = np.array(Q).reshape((n_labels, h, w))
    refined_mask = np.argmax(preds, axis=0).astype(np.uint8)

    return refined_mask
