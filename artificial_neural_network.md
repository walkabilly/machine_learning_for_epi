---
title: "Artificial Neural Network"
date: "2024-09-23"
output:
      html_document:
        keep_md: true
---


``` r
library(AppliedPredictiveModeling)
library(brulee)
library(tidymodels)
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
```

```
## ✔ broom        1.0.6     ✔ recipes      1.1.0
## ✔ dials        1.3.0     ✔ rsample      1.2.1
## ✔ dplyr        1.1.4     ✔ tibble       3.2.1
## ✔ ggplot2      3.5.1     ✔ tidyr        1.3.1
## ✔ infer        1.0.7     ✔ tune         1.2.1
## ✔ modeldata    1.4.0     ✔ workflows    1.1.4
## ✔ parsnip      1.2.1     ✔ workflowsets 1.1.0
## ✔ purrr        1.0.2     ✔ yardstick    1.3.1
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## ✖ purrr::discard() masks scales::discard()
## ✖ dplyr::filter()  masks stats::filter()
## ✖ dplyr::lag()     masks stats::lag()
## ✖ recipes::step()  masks stats::step()
## • Dig deeper into tidy modeling with R at https://www.tmwr.org
```

``` r
library(torch)
library(themis)
library(tidyverse)
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ forcats   1.0.0     ✔ readr     2.1.5
## ✔ lubridate 1.9.3     ✔ stringr   1.5.1
```

```
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ readr::col_factor() masks scales::col_factor()
## ✖ purrr::discard()    masks scales::discard()
## ✖ dplyr::filter()     masks stats::filter()
## ✖ stringr::fixed()    masks recipes::fixed()
## ✖ dplyr::lag()        masks stats::lag()
## ✖ readr::spec()       masks yardstick::spec()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

``` r
library(sjPlot)
library(finalfit)
library(knitr)
library(gtsummary)
library(mlbench)
```


``` r
data <- read_csv("data_imputed.csv")
```

```
## Rows: 39392 Columns: 29
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr  (6): diabetes, pa_cat, latinx, indigenous, eb_black, fatty_liver
## dbl (23): diabetes_t2, PM_BMI_SR, SDC_AGE_CALC, SDC_MARITAL_STATUS, SDC_EDU_...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
data$diabetes <- NULL

cols <- c("pa_cat", "latinx", "indigenous", "eb_black", "fatty_liver", "SDC_MARITAL_STATUS", "SDC_EDU_LEVEL", "SDC_INCOME", "HS_GEN_HEALTH",  "SDC_BIRTH_COUNTRY", "SMK_CIG_STATUS", "DIS_DIAB_FAM_EVER", "DIS_DIAB_FAM_EVER", "HS_ROUTINE_VISIT_EVER", "DIS_STROKE_EVER", "DIS_COPD_EVER", "DIS_LC_EVER", "DIS_IBS_EVER", "DIS_DIAB_FAM_EVER", "WRK_FULL_TIME", "WRK_STUDENT", "PM_BMI_SR", "PM_WEIGHT_SR_AVG")
data %<>% mutate_at(cols, factor)

data$diabetes_t2 <- as.factor(data$diabetes_t2)
data$PM_BMI_SR <- as.numeric(data$PM_BMI_SR)
data$PM_WEIGHT_SR_AVG <- as.numeric(data$PM_WEIGHT_SR_AVG)
```



``` r
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(10)

data_split <- initial_split(data, prop = 0.70, strata = diabetes_t2)

# Create data frames for the two sets:
train_data <- training(data_split)
table(train_data$diabetes_t2)
```

```
## 
##     0     1 
## 26057  1517
```

``` r
test_data  <- testing(data_split)
table(test_data$diabetes_t2)
```

```
## 
##     0     1 
## 11150   668
```

#### Recipe 

The code below is the recipe for the oversample data that we already ran previously in the logistic regression part. 


``` r
diabetes_rec_oversamp_rf <- recipe(diabetes_t2 ~ ., data = train_data) %>%
  step_upsample(diabetes_t2, over_ratio = 0.5) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())
```

### Model 


``` r
nnet_spec <- 
  mlp(epochs = 1000, hidden_units = 10, penalty = 0.01, learn_rate = 0.1) %>% 
  set_engine("brulee", validation = 0) %>% 
  set_mode("classification")

nnet_wflow <- 
  diabetes_rec_oversamp_rf %>% 
  workflow(nnet_spec)
```

### Fitting the model

``` r
set.seed(10)

nnet_fit <- fit(nnet_wflow, train_data)

nnet_fit %>% extract_fit_engine()
```

```
## Multilayer perceptron
## 
## relu activation
## 10 hidden units,  742 model parameters
## 39,085 samples, 71 features, 2 classes 
## class weights 0=1, 1=1 
## weight decay: 0.01 
## dropout proportion: 0 
## batch size: 39085 
## learn rate: 0.1 
## training set loss after 570 epochs: 0.524
```

## Testing


``` r
val_results <- 
  test_data %>%
  bind_cols(
    predict(nnet_fit, new_data = test_data),
    predict(nnet_fit, new_data = test_data, type = "prob")
  )
```
 
#### Accuracy


``` r
val_results %>% accuracy(truth = diabetes_t2, .pred_class)
```

```
## # A tibble: 1 × 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.854
```

``` r
#val_results %>% spec(truth = diabetes_t2, .pred_class)
```



# References

1. Classification models using a neural network [https://www.tidymodels.org/learn/models/parsnip-nnet/](https://www.tidymodels.org/learn/models/parsnip-nnet/)

2. Single layer neural network. [https://parsnip.tidymodels.org/reference/mlp.html](https://parsnip.tidymodels.org/reference/mlp.html)

3. Experimenting with machine learning in R with tidymodels and the Kaggle titanic dataset. [https://www.r-bloggers.com/2021/08/experimenting-with-machine-learning-in-r-with-tidymodels-and-the-kaggle-titanic-dataset/](https://www.r-bloggers.com/2021/08/experimenting-with-machine-learning-in-r-with-tidymodels-and-the-kaggle-titanic-dataset/)

