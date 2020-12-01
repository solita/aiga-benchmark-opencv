"""Ongoing efforts to refactor the matcher to skimage / scikit-learn"""

import numpy as np
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
from cover_recognition import FTEST_DIR, SPORT_COV_LARGE


def plot_orb_matches(glob="*_*.jpg", train_fpath=SPORT_COV_LARGE):
    """Taken straight from skimage tutorial, left relatively untouched"""

    fpath_test = next(FTEST_DIR.glob(glob))
    if not fpath_test.exists():
        raise RuntimeError("no matching file found!")

    with Image.open(train_fpath) as fimg:
        img_train = rgb2gray(np.array(fimg))

    with Image.open(fpath_test) as fimg:
        img_test = rgb2gray(np.array(fimg))

    # need to tinker with this to better capture the logo
    descriptor_extractor = ORB(fast_n=30, n_scales=30,
                               fast_threshold=0.05,
                               n_keypoints=1400)

    descriptor_extractor.detect_and_extract(img_train)
    keypoints_train = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_test)
    keypoints_test = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2,
                                max_ratio=0.8,  # will need tuning
                                max_distance=0.25,  # will need tuning
                                cross_check=True)
    _, ax = plt.subplots()
    plt.gray()

    plot_matches(ax, img_train, img_test,
                 keypoints_train, keypoints_test,
                 matches, keypoints_color=(1, 0, 0, 0.1),
                 matches_color=(0, 1, 0, 0.2))

    ax.axis('off')
    ax.set_title("Original Image vs. Shelf Image")
    plt.show()


if __name__ == "__main__":
    plot_orb_matches()
