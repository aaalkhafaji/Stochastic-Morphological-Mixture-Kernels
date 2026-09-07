# Data provenance and third-party notices

## Public data in the complete research bundle

MNIST source images are attributed to Yann LeCun, Corinna Cortes, and Christopher
J. C. Burges. The experiment retains selected source pixels and original
partition row IDs, with archive URLs and hashes in `data/source_manifest.json`.
The paper cites LeCun et al. (1998). No new license for the original MNIST data is
asserted here.

Fashion-MNIST is attributed to Han Xiao, Kashif Rasul, and Roland Vollgraf,
Zalando Research (2017). The original repository's MIT notice is supplied
unchanged as `data/Fashion_MNIST_LICENSE.txt`.
Source: https://github.com/zalandoresearch/fashion-mnist

The synthetic masks, corruption realizations, result arrays, and fitted models
were generated in this revision. The supplied data are small selected public
subsets and their binary derivatives, not replacement releases of the complete
original datasets. Dependencies are named and versioned in `requirements.txt`;
third-party numerical software is not bundled.


Original project code and generated outputs: see LICENSE_NOTICE.md. Dependencies are installed separately under their own licenses.

## Oxford-IIIT Pet — Step 5 external validation dependency

The Step-5 working branch can use the Oxford-IIIT Pet pixel-level trimap annotations from the University of Oxford Visual Geometry Group. The upstream host states that the dataset is available under Creative Commons Attribution-ShareAlike 4.0 International and that image copyrights remain with the original owners. The project does not redistribute the Oxford annotation archive. See `data/OXFORD_IIIT_PET_NOTICE.md` for the official source and expected MD5.
