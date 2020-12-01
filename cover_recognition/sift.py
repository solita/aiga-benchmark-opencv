"""SIFT playground for magazine cover matching

Even with fancy CNN-based approaches, SIFT is still *really good*:
https://arxiv.org/abs/2003.01587

n.b.: SIFT patent expired in 2020, so it appears to be commercially viable.
See relevant HN thread here: https://news.ycombinator.com/item?id=22519866

Parts of code here were directly adapted from the OpenCV tutorial.
"""

from pathlib import Path
import cv2
from cover_recognition.paths import (SPORT_COV_LARGE, SPORT_COV_SMALL,
                                     EEVA_COV_LARGE, FTEST_DIR)

FLANN_INDEX_KDTREE = 1


def _crop_top(img, fraction=1, scale=1):
    img_cut = img[:int(fraction * img.shape[0])]
    enlarged = tuple(s*scale for s in img_cut.shape[:2])[::-1]
    img_cut = cv2.resize(img_cut, enlarged,
                         interpolation=cv2.INTER_CUBIC)
    return img_cut


def plot_matching_keypoints(ftrain, ftest, saveto="", crop_top_fraction=0.25,
                            tot_scaler=3, dist_thresh=0.5):
    """Matches SIFT keypoints and plots the output to a file"""

    img_train_bgr = cv2.imread(str(ftrain))
    # img_train_mpl = cv2.cvtColor(img_train_bgr, cv2.COLOR_BGR2RGB)
    img_train = cv2.cvtColor(img_train_bgr, cv2.COLOR_BGR2GRAY)

    img_top = _crop_top(img_train, crop_top_fraction, tot_scaler)
    img_top_bgr = _crop_top(img_train_bgr, crop_top_fraction, tot_scaler)
    # img_top_mpl = _crop_top(img_train_mpl, crop_top_fraction, tot_scaler)

    img_test_bgr = cv2.imread(str(ftest))
    # img_test_mpl = cv2.cvtColor(img_test_bgr, cv2.COLOR_BGR2RGB)
    img_test = cv2.cvtColor(img_test_bgr, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp_train, des1 = sift.detectAndCompute(img_top, None)
    kp_test, des2 = sift.detectAndCompute(img_test, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < dist_thresh * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=matchesMask,
        flags=0)

    img_matches_bgr = cv2.drawMatchesKnn(img_top_bgr, kp_train,
                                         img_test_bgr, kp_test,
                                         matches, None, **draw_params)

    if saveto:
        saveto.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(saveto), img_matches_bgr)
    else:
        raise NotImplementedError  # TODO


if __name__ == "__main__":
    # as a short demo, this below reads images (low- and high-res covers)
    # from designated paths defined in `paths.py` and cross-checks those
    # versus *all* the images in a `paths.FTEST_DIR` folder, writing the
    # resulting images into ./figs/
    fig_dir = Path("figs").resolve()

    for fpath_test in FTEST_DIR.glob("*"):
        rel_source = fpath_test.relative_to(FTEST_DIR)
        print(f"Looking for magazines at {rel_source}")

        # two block for cropped/non-cropped Sport cover matching
        plot_matching_keypoints(
                ftrain=SPORT_COV_LARGE,  # do full-res matching
                ftest=fpath_test,
                saveto=fig_dir/f"lsport-{rel_source}",
                crop_top_fraction=1.00,  # no cropping, use full image
                tot_scaler=3)  # fills the entire vspace
        plot_matching_keypoints(
                ftrain=SPORT_COV_SMALL,  # do lowres matching, only use the top
                ftest=fpath_test,
                saveto=fig_dir/f"csport-{rel_source}",
                crop_top_fraction=.25,  # no cropping, use full image
                tot_scaler=5)  # fills the entire vspace

        # TODO: same but for eeva - need to optimize for eeva logo?
        plot_matching_keypoints(
                ftrain=EEVA_COV_LARGE,  # do full-res matching
                ftest=fpath_test,
                saveto=fig_dir/f"leeva-{rel_source}",
                crop_top_fraction=1.00,  # no cropping, use full image
                tot_scaler=3)  # fills the entire vspace
        plot_matching_keypoints(
                ftrain=EEVA_COV_LARGE,  # only use the top
                ftest=fpath_test,
                saveto=fig_dir/f"teeva-{rel_source}",
                crop_top_fraction=.25,  # no cropping, use full image
                tot_scaler=3)  # fills the entire vspace
