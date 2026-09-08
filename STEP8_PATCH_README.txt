Step 8 checksum-stage patch
===========================
This patch fixes the post-campaign checksum stage in .github/workflows/step8-modern-baselines.yml.
The previous loop attempted sha256sum on the nested output directory step5_oxford_pet and exited with code 1.
The revised workflow recursively hashes files only, including nested result files.
Copy the contents of this patch into the repository root, replace matching files, commit, push, wait for Package integrity to pass, then run a NEW Step 8 workflow.
