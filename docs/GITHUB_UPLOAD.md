# GitHub upload instructions

1. Extract `JVCIR_GitHub_Repository_v2.1.0_under25MB.zip`.
2. Create a new GitHub repository.
3. Commit/upload the extracted **contents**, not the ZIP archive itself.
4. Keep the large files listed in `ZENODO_ASSETS.csv` out of Git history. They belong in the Zenodo archival snapshot.
5. Add the final GitHub repository URL to `README.md`, `CITATION.cff`, and `.zenodo.json`.
6. Tag the archival commit as `v2.1.0`.
7. Archive that tagged release through Zenodo and add the resulting DOI back to the metadata files in the next maintenance commit/release.

The compact repository archive is designed to remain below 25 MB while preserving all source code and clean inputs required for full regeneration.
