import mxnet as mx
import numpy as np

min_object_covered = [0.1, 0.3, 0.5, 0.7, 0.9]  # use 5 augmenters
aspect_ratio_range = (0.75, 1.33)  # use same range for all augmenters
area_range = [(0.1, 1.0), (0.2, 1.0), (0.2, 1.0), (0.3, 0.9), (0.5, 1.0)]
min_eject_coverage = 0.3
max_attempts = 50
aug = mx.image.det.CreateMultiRandCropAugmenter(min_object_covered=min_object_covered,
                                                aspect_ratio_range=aspect_ratio_range, area_range=area_range,
                                                min_eject_coverage=min_eject_coverage, max_attempts=max_attempts,
                                                skip_prob=0)
aug.dumps()  # show some details

a = mx.nd.ones((300, 300, 3))
label = np.array([
    [0, 0.1, 0.1, 0.4, 0.6]
])
aug(a, label)
