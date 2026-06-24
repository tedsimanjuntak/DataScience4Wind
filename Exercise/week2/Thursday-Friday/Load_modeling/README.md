# Surrogate Modelling for Wind Turbine Loads: A Guide to the Notebooks

This guide accompanies the three notebooks for the surrogate modelling exercise on the IEA 3.4 MW wind turbine. It explains what you will do in each notebook, which variables you can and should change, and what to look for in the code.

---

## General Notes Before You Start

### Run end-to-end first

Before you start experimenting, run each notebook **from top to bottom without changing anything**. This verifies that your environment is set up correctly and that all cells execute successfully. If a cell fails, it may leave variables in an inconsistent state, causing confusing errors in later cells. Use **Kernel → Restart & Run All** to get a clean slate.

### File structure and relative paths

The notebooks expect a specific folder layout relative to the notebooks location (`.ipynb` files). If you move the notebooks or data files, the `FileNotFoundError` that you will see on the data-loading cell is the most common symptom.

### Python environment: JupyterHub and local setup

The exercise runs on a shared **JupyterHub** server. All required packages are pre-installed in the default kernel, so you do not need to install or configure anything before opening the notebooks.

**Advanced users — local setup:** If you prefer to work independently on your own machine, an environment YAML file is provided alongside the notebooks. You can use it to recreate the exact environment locally with `conda env create -f environment.yml`, followed by `conda activate <env-name>`. This is entirely optional and intended for students who are comfortable managing conda environments and want to run the notebooks outside JupyterHub.

### RAM usage: shared server etiquette

The JupyterHub server has a fixed memory budget shared across all students working simultaneously, so keeping your RAM footprint low is important to prevent kernel crashes for yourself and others. Several operations in these notebooks are memory-intensive and deserve special attention. To work economically: keep parameter grids small while prototyping and expand them only for a final run and **restart the kernel** (`Kernel → Restart`) when you are done with a notebook section or switching between notebooks to release memory. Avoid keeping multiple notebooks with active kernels running at the same time unless you need to.

### Keeping parameters consistent

Notebook 2 reproduces the same train/validation split that you explored in Notebook 1. For this to work, you must use **exactly the same values** for `Parameters and Settings` (e.g., `SELECTED_REGIME`, `SPLIT_METHOD`, `TRAIN_FRACTION`, `RANDOM_STATE`, and all binning parameters) in both notebooks. If you change a split parameter in Notebook 1 and forget to update Notebook 2, the models in Notebook 2 will be trained on a different split than the one you analysed.

### Saving and naming models

Notebook 2 saves trained models as `.pkl` bundles in the `student/` folder. **Before each experiment**, change `RUN_TAG` to a descriptive string (e.g. `"log_y_relu"` or `"kfold_xgb"`) so that you do not overwrite a model you want to keep.

---

## Notebook 1: Data Exploration and Train/Validation/Test Splits

This notebook loads the simulation data for one operating regime, applies a train/validation split strategy, and performs a statistical analysis and visualisation of the resulting splits.

### Parameters and settings

All configuration is concentrated in a single parameter cell at the top of the notebook. The key variables to change are `SELECTED_REGIME` (one of `"Normal"`, `"Idling"`, `"Shut"`, `"Start"`), which selects which turbine operating mode you are working with; and `TARGET_COL`, which selects which output channel to focus on for splitting and statistical analysis. A table of available target columns is provided in the notebook. Start with `SELECTED_REGIME = "Normal"` and one power or load column, then repeat the analysis for a different regime or target and compare the results.

### Splitting strategies

The variable `SPLIT_METHOD` controls how the full-factorial training pool is divided into training and validation sets. The five available strategies are `"random"`, `"holdout_wsp"`, `"holdout_ti"`, `"stratified_target"`, and `"stratified_joint"`. Try each method in turn and observe what changes in the subsequent plots: which (Wind Speed, TI) regions end up in validation, whether the input and target distributions match between train and val, and what happens to the linear regression R² when validation covers a wind speed band that was not seen during training. For `holdout_wsp`, vary `HOLDOUT_WSP_RANGE` (e.g. change from `(9.0, 11.0)` to a wider band like `(6.0, 12.0)`) to see how much input-space coverage you sacrifice. For stratified splits, vary `TRAIN_FRACTION` between `0.6` and `0.85` and note how the validation set size and target distribution change. `RANDOM_STATE` controls reproducibility, so changing it with an otherwise identical configuration lets you assess how sensitive the split is to the random draw.

### Statistical analysis: correlations and descriptive statistics

After the split is applied, the notebook computes Pearson correlation matrices between the selected input columns (`SELECTED_INPUT_COLS`) and the target. You can extend `SELECTED_INPUT_COLS` beyond the default `Wind Speed` and `Turbulence Intensity` by uncommenting `RotSpd_mean`, `PitAnBl1_mean`, or `TrqGen_mean`. When you do, compare the correlation matrices across train, val, and test: do the correlations agree? Are any two inputs so highly correlated that adding the second one provides little information? The descriptive statistics table shows whether the splits cover similar ranges and means for each variable, where a large discrepancy in the mean or max of Wind Speed between train and val is a warning sign that the model will be evaluated on conditions it did not learn from.

### Linear regression baseline

A `StandardScaler`-normalised linear regression is fitted on the training set and evaluated on the validation set. Vary `SELECTED_INPUT_COLS` (adding or removing features) and observe how training R² and validation R² change. A large gap between the two suggests overfitting, which is unlikely for a linear model but can appear when inputs are strongly correlated. The printed coefficient table shows the standardised effect of each input; try interpreting the sign and magnitude physically (e.g. does rotor speed have a larger or smaller standardised effect on tower loads than wind speed?). This model is intentionally too simple for production use, rather its purpose is to give you a quantitative baseline and an interpretable view of feature relevance before moving to the more flexible surrogates in Notebook 2.

### K-fold cross-validation demo

The K-fold demonstration at the bottom of the notebook applies stratified K-fold to the full factorial dataset and prints target statistics for each fold. Change `n_splits` (e.g. try 3 versus 10) and `nbins` to see how the fold sizes and target coverage change. This section is informational, meaning that understanding K-fold here will help you interpret the `USE_KFOLD` option in Notebook 2, where it is used during hyperparameter search.

### Interactive plots

Four Plotly figures are produced. The **input space scatter** (Plot 1) shows the (Wind Speed, TI) locations of each point coloured by dataset membership. Toggle the legend entries to isolate train, val, or test, and compare this across different `SPLIT_METHOD` values. The **histograms** (Plot 2) let you visually confirm whether wind speed, TI, and the target are similarly distributed across splits. The **target vs inputs scatter** (Plot 3) reveals the nonlinear structure of the data. The **binned average** (Plot 4) is a de-noised version of this view; discrepancies in the binned curves between train and val indicate that the split is not representative in some region.

---

## Notebook 2: Training and Comparing Surrogate Models

This notebook trains three families of surrogate models, neural networks, XGBoost, and Gaussian Process Regression, on the same data used in Notebook 1. All model training and hyperparameter search uses only the training and validation sets.

### Reproducing the split and global configuration

The parameter cell at the top of Notebook 2 mirrors that of Notebook 1. Set `SELECTED_REGIME`, `TARGET_COL`, `SPLIT_METHOD`, `TRAIN_FRACTION`, `RANDOM_STATE`, and the stratification bin parameters to the same values you chose in Notebook 1 before running anything else. `INPUT_COLS` is set to `["Wind Speed", "Turbulence Intensity"]` by default; this is the standard configuration. Extending it to include additional inputs (e.g. adding `"RotSpd_mean"` as explored in Notebook 1) requires updating `INPUT_COLS` here accordingly.

### Neural network with grid search

The NN grid search section is controlled by `NN_PARAM_GRID`, `X_SCALER_NAME`, `Y_SCALER_NAME`, `PRIMARY_METRIC_NAME`, and `USE_KFOLD`. Uncomment combinations in `hidden_layer_sizes` (e.g. add `(32,)`, `(64,)`, `(32, 32)`, `(64, 64)`) and in `activation` (add `"tanh"` or `"logistic"`) and in `alpha` (add `1e-4`, `1e-3`) to compare more architectures. Keep the grid small enough to run in a reasonable time: four `hidden_layer_sizes` × two `activation` × two `alpha` gives 16 combinations, which is a sensible starting point. For `Y_SCALER_NAME`, try switching between `"standard"` and `"log"`.Log-transforming fatigue load targets often improves MAPE because loads span a wide dynamic range. Compare the validation RMSE and MAPE with and without the log transform. Toggle `USE_KFOLD = True` to use stratified K-fold evaluation instead of a single train/val split, which gives a less noisy estimate of generalisation performance at the cost of longer runtime.

### Neural network with Optuna

The Optuna section searches the NN hyperparameter space automatically. Increase `N_TRIALS` from the default of 5 to 20–50 to give Optuna more budget; with only 5 trials the results are dominated by chance. You can widen or narrow the **search space** within the `objective` function: for `alpha`, try changing the bounds from `(1e-3, 1e-1)` to `(1e-5, 1e-1)` to allow stronger regularisation; for `max_iter`, change the categorical list to `[100, 300, 500]` to allow more or fewer training epochs. After running, compare the best Optuna configuration to the best grid-search configuration: are the chosen architectures similar? Which gives lower validation RMSE? Compare training time per trial between grid search (deterministic, exhaustive) and Optuna (adaptive, sample-efficient).

### XGBoost with grid search

`XGB_PARAM_GRID` defines the combinations to try. The most impactful parameters to vary are `n_estimators` (adding e.g. `100` for a faster but weaker model), `max_depth` (adding `7` or `8` to allow deeper trees), and `learning_rate` (adding `0.01` for a slower but potentially better-generalising model). The `subsample` and `colsample_bytree` parameters control stochastic regularisation. where values below `1.0` introduce randomness that can prevent overfitting but also increase variance. By default `USE_KFOLD = True` for the XGBoost section, which is recommended because trees are more prone to overfitting on small validation sets than NNs. Compare XGBoost validation RMSE against the NN results for the same target column and note which metric (RMSE, MAPE, R²) differs most between the two model families.

### XGBoost with Optuna

`N_TRIALS = 20` is the default. Increase this to 50 for a more thorough search. The Optuna search space covers `n_estimators` (50–300), `max_depth` (2–6), `learning_rate` (0.03–0.3), `subsample` (0.7–1.0), `colsample_bytree` (0.7–1.0), and `reg_lambda` (1e-3–1.0). After optimisation, inspect the best parameters: is Optuna preferring shallow or deep trees? A high `reg_lambda` combined with a low `learning_rate` suggests the problem rewards slow, regularised learning. Compare the best Optuna XGBoost to the best Optuna NN across all metrics.

### Gaussian Process Regression — kernel comparison

The variable `GPR_KERNEL_NAMES` lists the kernels to compare (e.g., `"RBF"`, `"Matern32"`, `"Matern52"`, `"RBF+Matern32"`, and `"RBF+Matern52"`). After the comparison, inspect the **fitted lengthscales** printed: a large lengthscale in the Wind Speed dimension relative to TI means the target varies more slowly with wind speed (in the scaled space), and vice versa.

---

## Notebook 3: Evaluating and Comparing Surrogate Models

This notebook loads the trained model bundles from Notebook 2 and the fixed test set, then performs a comprehensive comparison of model quality and behaviour. This is the only notebook that uses the test set for evaluation.

### Configuring models and the test domain

At the top of the notebook, set `SELECTED_REGIME` and `TARGET_COL` to match the models you trained in Notebook 2. Edit `MODEL_DEFINITIONS` to include the models you want to compare: each entry is a `(name, path)` pair, where `name` is a short label that will appear in all plots and tables, and `path` is the file path to the `.pkl` bundle. Use descriptive names (e.g. `"nn_log_y"` rather than `"nn"`) so that plots are self-explanatory. The special entry `("spline_interpolation", "")` adds a 2D cubic spline of the training grid as a deterministic baseline. The sweep domain variables `WSP_MIN`, `WSP_MAX`, `WSP_STEP`, `TI_MIN`, `TI_MAX`, `TI_STEP` control the 1D sweep resolution; `FIXED_TI_VALUES` and `FIXED_WSP_VALUES` select the fixed operating points at which sweeps are shown. Change `FIXED_TI_VALUES` to values that span the training range (e.g. `[4, 10, 20]` if TI goes from 4 % to 25 %) to see model behaviour at low, medium, and high turbulence.

### Spline reference model

The spline is built automatically from the full-factorial training data. It handles missing grid cells (simulations that were unstable and removed) by filling them with the value of the nearest available cell. Because it interpolates the simulation grid exactly, it is an almost perfect approximation within the training domain and should serve as a ceiling for what any ML model can achieve on training-domain test points.

### 1D sweeps over wind speed and turbulence intensity

The sweep plots show each model's predicted load or power as a continuous curve over Wind Speed (at fixed TI) and over TI (at fixed Wind Speed). Change `FIXED_TI_VALUES` and `FIXED_WSP_VALUES` to different operating conditions and compare how the curves evolve. XGBoost curves are piecewise-constant under the hood; this is a structural property of tree-based models, not a sign of poor fitting. Toggle `SHOW_GPR_UNCERTAINTY = True` if you have a GPR model loaded as this adds ±2σ bands around the GPR mean, which become wider outside the training domain, visualising where the model is most uncertain.

### Scatter plot: model predictions vs test data

This plot overlays the model predictions (evaluated at the test set inputs) on top of the actual test data, with Wind Speed on the x-axis. Models that consistently sit above or below the test scatter have a systematic bias. Use the legend to toggle individual models on and off to isolate specific comparisons.

### 2D response surfaces

Each model is evaluated on a fine (WS, TI) grid and plotted as a 3D surface. Rotate the surface using the Plotly controls and compare the shapes: does the XGBoost surface show a "staircase" structure? Is the GPR surface smoother than the NN? Does any model produce physically implausible ridges or valleys outside the training range? Change `WSP_MIN_SURF`, `WSP_MAX_SURF`, `TI_MIN_SURF`, `TI_MAX_SURF` to extend beyond the training domain (deliberately into extrapolation territory) and observe how each model behaves.

### Global metrics table

The colour-coded table compares all loaded models on ME, MAE, MSE, RMSE, MAPE, R², nRMSE\_mean, and nRMSE\_std on the test set. Within each column, darker blue indicates a better value. Pay attention to the distinction between absolute metrics (RMSE, MAE) and relative metrics (MAPE, nRMSE\_mean): a model with a lower RMSE than another may still have a higher MAPE if the low errors happen to occur at low target values. The sign of ME (mean error) reveals systematic bias as a model with ME close to zero but high RMSE has large but symmetric errors, whereas a model with large |ME| is consistently over- or under-predicting.

### Parity plots

Parity plots show predicted versus true values for each model, along with a fitted regression line. A perfect model has all points on the identity line (y = x). A tilted regression line (slope ≠ 1 or intercept ≠ 0) indicates a systematic bias that depends on the magnitude of the target, for example, underprediction of high loads. Compare parity plots across models for the same target: which model's points are tightest around the identity line? Are there systematic outliers in a particular region of load values?

### Error distributions: boxplots and histograms

Boxplots of signed errors and absolute percentage errors show the spread and skewness of each model's errors. A long upper tail in the APE boxplot means the model occasionally makes very large relative errors even if the median is small. The error histograms allow you to compare the full distribution shape across models as a model may have the same RMSE as another but a bimodal error distribution, suggesting it performs well on one subset of the data and poorly on another. Switch `which` between `"error"`, `"pe"`, and `"ape"` to see different perspectives on the same errors.

### Error vs target, wind speed, and turbulence intensity

These scatter plots show whether errors are correlated with the input or target value. Set `x_source` to `"target"` to check for heteroscedasticity (errors growing with target magnitude), to `"wsp"` to check for wind-speed-dependent bias, or to `"ti"` to check for turbulence-intensity-dependent bias. A cloud of points centred on zero with no trend is ideal; a fan shape (errors growing larger at high values) suggests that a log transform of the target would help. A systematic upward or downward trend with wind speed suggests a physically important regime that the model is not capturing correctly.

### Error heatmaps over the (WS, TI) plane

The 2D binned error heatmaps aggregate point-wise percentage errors into a grid and show where each model is systematically wrong. All models use the same colour scale, making direct comparison straightforward. Red regions (positive mean error) indicate systematic overprediction; blue regions indicate underprediction. Compare the heatmaps for the ML models against the spline: regions where the spline has low error but an ML model has high error are candidates for architectural improvement. Change `n_wsp_bins` and `n_ti_bins` (e.g. from `6` to `10`) for a finer spatial resolution, or reduce them for a coarser but more statistically robust view.