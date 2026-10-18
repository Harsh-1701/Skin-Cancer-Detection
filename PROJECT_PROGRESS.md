Project Progress
Current status
The main implementation is complete enough for a project demonstration. The latest code has been pushed to GitHub on main, and the local working tree was clean after the push.
Repository: https://github.com/Harsh-1701/Skin-Cancer-Detection
Latest recorded commit: 38b2a28 — Finalized skin lesion classification project
Completed
- Built a skin-lesion classification pipeline using EfficientNet.
- Prepared a combined dataset with nine labels, including an unknown class.
- Implemented training and evaluation workflows for EfficientNet-B0 and EfficientNet-B4.
- Added a Streamlit application for image upload and prediction.
- Added Grad-CAM visualisation to accompany predictions.
- Added a PDF report download.
- Added an evaluation page with model metrics and plots.
- Added project-team and guide information to the application.
- Created a GitHub repository and pushed the finished code to main.
- Kept datasets and model checkpoint files out of normal source-code commits through .gitignore.
Current model setup
EfficientNet-B0 is used for live inference in the web application. EfficientNet-B4 is retained for comparison on the evaluation page.
The verified test results currently recorded for the project are:
Metric	B0	B4
Accuracy	80.96%	79.48%
Weighted F1-score	0.7976	0.7906
Macro F1-score	0.7220	0.6713
Melanoma recall	46.34%	55.56%
SCC recall	30.85%	28.72%


The two models do not lead on every metric. B0 has higher overall accuracy and macro F1 in this evaluation, while B4 has higher melanoma recall. This distinction should be preserved in presentations and the written report.
Before submission
- [ ] Clone the repository into a clean folder and confirm the setup instructions work.
- [ ] Confirm the required checkpoint exists locally and the application starts.
- [ ] Test image upload, prediction, Grad-CAM display and PDF download.
- [ ] Check the evaluation page and confirm every plot loads.
- [ ] Verify that the report uses the same test metrics shown above.
- [ ] Confirm all references and dataset acknowledgements are included in the final paper.
- [ ] Keep the original dataset and model checkpoints backed up locally; they are not included in the Git repository.
- [ ] Review requirements.txt against the packages actually used by the application and evaluation scripts.
Known limitations and cautions
- Class distribution is uneven, with nv making up a large share of the dataset.
- Recall remains low for some minority classes, including squamous cell carcinoma.
- Grad-CAM provides a visual explanation of model attention, not a clinical explanation.
- The application is an academic prototype and must not be presented as a replacement for professional diagnosis.
Next work
The priority is final verification and documentation rather than adding new features. Run the application from a clean environment, check the repository instructions, and capture final screenshots for the report and presentation. Avoid changing model code close to submission unless a reproducible issue is found.
Notes
This file records the current project state. Update the metrics and checklist if the model, dataset split, checkpoint or evaluation procedure changes.