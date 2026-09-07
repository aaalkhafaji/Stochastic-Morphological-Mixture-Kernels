# Precomputed models

The fitted `.joblib` estimators are intentionally excluded from the compact GitHub package to keep the repository download below 25 MB.

They are available in the companion Zenodo archival snapshot (`JVCIR_Zenodo_Archive_v2.1.0.zip`) once the record is published. Alternatively, regenerate them from the supplied clean inputs with:

```bash
python code/reproduce.py --level full --workdir ../smm-full
```

Do not add regenerated binary model files to Git history; keep them local or attach them to the archival release.
