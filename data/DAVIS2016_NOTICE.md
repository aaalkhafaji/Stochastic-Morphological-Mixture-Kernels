# DAVIS 2016 external-data notice

Step 6 uses the official DAVIS 2016 single-object segmentation ground-truth masks and the official pre-computed benchmark segmentations. The public archives are not redistributed in this repository. `code/run_davis_step6.py` downloads them from the official DAVIS/ETH locations, records archive hashes, and uses only masks/split files; RGB frames are not used by the estimator.

Primary source methods: NLC, FST, SAL, TRC, MSG, CVOS. The paper cites Perazzi et al., CVPR 2016, DOI 10.1109/CVPR.2016.85.
