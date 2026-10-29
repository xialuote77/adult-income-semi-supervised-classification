# Review and reproducibility

## Authoritative source

The source of the original work is code/COMP30027_2026_asst1_[student-id].ipynb and Report.pdf in the supplied A1 folder. The top-level sklearn notebook is a different implementation and was not used as the original submission. The original report is 10 pages; a text scan found no student identifiers or contact addresses. It is copied unchanged as reports/historical-coursework-report.pdf. Raw demographic records are excluded.

## Corrections

1. The original threshold search split labelled training data but generated pseudo-labels using the earlier full-training model, and reused its categorical vocabulary. Validation labels could therefore indirectly influence pseudo-label selection. The reviewed run fits the teacher and transformations on the training partition only during selection.
2. The original predictor adds two log-posterior arrays without subtracting a duplicate log prior, despite a comment saying it does. The reviewed score removes one copy. A separate calculation from learned Gaussian/categorical densities checks the resulting log-odds.
3. The original test imputation uses test medians despite a comment saying training median. The reviewed model stores training medians and applies them unchanged. In the supplied files, missingness is mostly categorical; the new check also exercises a synthetic missing continuous value.
4. Unknown categories have a reserved extra category in every fitted CategoricalNB. Encoder fitting is local to each training subset. Refitted final models use the full labelled training set after threshold selection.
5. An empty confident pseudo-label selection retains the supervised model. It does not fall back to adding all unlabelled rows.

The selected threshold changes from the historical 12 to 10. Historical positive recall/F1 (approximately .44/.55 supervised and .49/.58 semi-supervised) are retained only in the original report. Current results are approximately .5307/.6076 and .5578/.6150. These are not interchangeable experiments.

## Method and checks

Complete-case filtering of the 13 predictor columns retains 15,076 labelled and 15,059 unlabelled rows. fnlwgt is not a predictor. The validation split is 20%, stratified, seed 42. The fixed threshold grid and first-maximum tie rule are specified in source. Only positive F1 selects a threshold. The optional unlabelled-with-labels CSV is never read.

Checks verify disjoint training/validation row indices; training-only medians and fit counts; one-prior log-odds against an independently calculated density formula; finite predictions for an unseen category; use of stored medians for missing continuous inputs; and the final labelled-plus-pseudo-labelled row count. Results and input SHA-256 hashes are stored in results/run-record.json.

The source files contain a small number of exact duplicate full rows (6 labelled, 8 unlabelled, 7 test). They are not automatically removed: no person identifier is available to establish whether identical feature records are the same individual. The split assertion establishes distinct row indices, not independent individuals. No grouped/person-level or temporal generalisation claim is made.

The script and three notebook code cells were executed top-to-bottom using Python 3.13.13, NumPy 2.4.6, pandas 3.0.3 and scikit-learn 1.8.0. Jupyter/nbformat were not installed in the available runtimes. A script-based notebook execution fallback ran each code cell in source order, retained stream/figure outputs and produced a static HTML preview; the notebook JSON structure was checked. No claim is made that it was executed inside the Jupyter application. Numeric outputs matched between CLI and notebook execution. Saved figures were visually inspected.

## Interpretation limits

This is a retrospective coursework split whose test results were previously inspected. The revised threshold selection does not use test labels, but the test is not new. A single split, no confidence intervals and no repeated-seed comparison leave the small F1 difference unquantified. Complete-case training may change the represented population. The independence/Gaussian assumptions and sensitive demographic attributes require careful interpretation. Historical language calls the procedure label propagation, while the code actually implements one-round confidence-based self-training.
