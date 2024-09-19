---
title: "Data Wrangling"
author: "Daniel Fuller"
date: "2024-09-19"
output:
      html_document:
        keep_md: true
---



Let's simplify the dataset so we are not working with so many variables. 


``` r
data <- read_csv("data.csv")
```

```
## Warning: One or more parsing issues, call `problems()` on your data frame for details,
## e.g.:
##   dat <- vroom(...)
##   problems(dat)
```

```
## Rows: 41187 Columns: 440
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr   (5): ID, MSD11_PR, MSD11_REG, MSD11_ZONE, MSD11_CMA
## dbl (425): ADM_STUDY_ID, SDC_GENDER, SDC_AGE_CALC, SDC_MARITAL_STATUS, SDC_E...
## lgl  (10): DIS_MH_BIPOLAR_EVER, DIS_GEN_DS_EVER, DIS_GEN_SCA_EVER, DIS_GEN_T...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
data <- select(data, "DIS_DIAB_TYPE", "PM_BMI_SR", "SDC_AGE_CALC", "PA_LEVEL_SHORT", "SDC_EB_ABORIGINAL", "SDC_EB_LATIN", "SDC_EB_BLACK", "DIS_LIVER_FATTY_EVER", "SDC_MARITAL_STATUS", "SDC_EDU_LEVEL", "SDC_INCOME", "HS_GEN_HEALTH", "NUT_VEG_QTY", "NUT_FRUITS_QTY", "ALC_CUR_FREQ", "SDC_BIRTH_COUNTRY", "PA_SIT_AVG_TIME_DAY", "SMK_CIG_STATUS", "SLE_TIME", "DIS_DIAB_FAM_EVER", "HS_ROUTINE_VISIT_EVER", "WH_HRT_EVER", "DIS_STROKE_EVER", "DIS_COPD_EVER", "DIS_LC_EVER", "DIS_IBS_EVER", "DIS_DIAB_FAM_EVER", "WRK_FULL_TIME", "WRK_STUDENT", "PM_WEIGHT_SR_AVG")
```

#### Outcome variable

Let's look at the outcome variable, recode, and drop observations that are not relevant. We know that the GLM function needs a 0/1 variable and we want to recode that way now so we don't need to change it after. We also know we want to keep our gestational diabetes variable because we need it later. 


``` r
table(data$DIS_DIAB_TYPE)
```

```
## 
##    -7     1     2     3 
## 36807   315  2160   425
```

``` r
data <- data %>%
	mutate(diabetes_t2 = case_when(
    DIS_DIAB_TYPE == 2 ~ 1,
    DIS_DIAB_TYPE == -7 ~ 0, 
		TRUE ~ NA_real_
	))

data$diabetes_t2 <- as.factor(data$diabetes_t2)

table(data$diabetes_t2, data$DIS_DIAB_TYPE)
```

```
##    
##        -7     1     2     3
##   0 36807     0     0     0
##   1     0     0  2160     0
```

``` r
data <- data %>%
	mutate(diabetes_gestat = case_when(
    DIS_DIAB_TYPE == 3 ~ 1,
    DIS_DIAB_TYPE == -7 ~ 0, 
		TRUE ~ NA_real_
	))

data$diabetes_gestat <- as.factor(data$diabetes_gestat)

data <- filter(data, diabetes_t2 == 0 | diabetes_t2 == 1 | diabetes_gestat == 1)

table(data$diabetes_t2, data$DIS_DIAB_TYPE)
```

```
##    
##        -7     2     3
##   0 36807     0     0
##   1     0  2160     0
```

``` r
data <- data %>%
	mutate(diabetes = case_when(
    diabetes_t2 == 0 ~ "neg",
    diabetes_t2 == 1 ~ "pos"
	))

table(data$diabetes_t2, data$diabetes)
```

```
##    
##       neg   pos
##   0 36807     0
##   1     0  2160
```

For logistic regression in the case of a cross-section study we want the outcome to be ~10% of the total sample. Here we have `2160/36807*100 = 5.86%`. 

#### Preparing predictor variables

**BMI overweight**


``` r
glimpse(data$PM_BMI_SR)
```

```
##  num [1:39392] NA 28.3 25.5 44.8 NA ...
```

``` r
summary(data$PM_BMI_SR) ### Lots of NAs! 
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##    8.86   23.34   26.58   27.53   30.52   69.40   11124
```

``` r
data <- data %>%
	mutate(bmi_overweight = case_when(
	  PM_BMI_SR >= 25.00 ~ "Overweight",
		PM_BMI_SR < 25.00 ~ "Not Overweight"
	))

table(data$bmi_overweight)
```

```
## 
## Not Overweight     Overweight 
##          10607          17661
```

**Age**


``` r
glimpse(data$SDC_AGE_CALC)
```

```
##  num [1:39392] 47 57 62 64 40 36 63 58 60 41 ...
```

``` r
summary(data$SDC_AGE_CALC) ### Lots of NAs! 
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##    30.0    43.0    52.0    51.5    60.0    74.0
```

``` r
data <- data %>%
	mutate(age_45 = case_when(
	  SDC_AGE_CALC >= 45.00 ~ "Over 45",
		SDC_AGE_CALC < 45.00 ~ "Under 45"
	))

table(data$age_45)
```

```
## 
##  Over 45 Under 45 
##    28415    10977
```

**Physical Activity**


``` r
glimpse(data$PA_LEVEL_SHORT)
```

```
##  num [1:39392] 3 1 NA NA NA 3 1 NA 3 3 ...
```

``` r
table(data$PA_LEVEL_SHORT)
```

```
## 
##     1     2     3 
##  9538 10606 13140
```

``` r
data <- data %>%
	mutate(pa_cat = case_when(
		PA_LEVEL_SHORT == 1 ~ "1_Low Activity",
		PA_LEVEL_SHORT == 2 ~ "2_Moderate Activity",
		PA_LEVEL_SHORT == 3 ~ "3_High Activity"
	))

table(data$pa_cat, data$PA_LEVEL_SHORT)
```

```
##                      
##                           1     2     3
##   1_Low Activity       9538     0     0
##   2_Moderate Activity     0 10606     0
##   3_High Activity         0     0 13140
```

**Racialized**


``` r
table(data$SDC_EB_ABORIGINAL)
```

```
## 
##     0     1 
## 35331  1351
```

``` r
table(data$SDC_EB_LATIN)
```

```
## 
##     0     1 
## 36221   451
```

``` r
table(data$SDC_EB_BLACK)
```

```
## 
##     0     1 
## 36149   518
```

``` r
### Latinx

data <- data %>%
	mutate(latinx = case_when(
		SDC_EB_LATIN == 1 ~ "Yes",
		SDC_EB_LATIN == 0 ~ "No"
	))

table(data$SDC_EB_LATIN, data$latinx)
```

```
##    
##        No   Yes
##   0 36221     0
##   1     0   451
```

``` r
### Indigenous

data <- data %>%
	mutate(indigenous = case_when(
		SDC_EB_ABORIGINAL == 1 ~ "Yes",
		SDC_EB_ABORIGINAL == 0 ~ "No"
	))

table(data$SDC_EB_ABORIGINAL, data$indigenous)
```

```
##    
##        No   Yes
##   0 35331     0
##   1     0  1351
```

``` r
### Black

data <- data %>%
	mutate(eb_black = case_when(
		SDC_EB_BLACK == 1 ~ "Yes",
		SDC_EB_BLACK == 0 ~ "No"
	))

table(data$SDC_EB_BLACK, data$eb_black)
```

```
##    
##        No   Yes
##   0 36149     0
##   1     0   518
```

**Fatty liver disease**


``` r
table(data$DIS_LIVER_FATTY_EVER)
```

```
## 
##   1   2 
##  50 199
```

``` r
data <- data %>%
	mutate(fatty_liver = case_when(
		DIS_LIVER_FATTY_EVER == 1 ~ "Yes",
		DIS_LIVER_FATTY_EVER == 2 ~ "Yes"
	))

data <- data %>%
	mutate(fatty_liver = case_when(
		DIS_LIVER_FATTY_EVER == 1 ~ "Yes",
		DIS_LIVER_FATTY_EVER == 2 ~ "Yes"
	))

data <- data %>% 
                  mutate(fatty_liver = replace_na(fatty_liver, "No"))

table(data$fatty_liver)
```

```
## 
##    No   Yes 
## 39143   249
```

**Replacing all -7 values with NA**


``` r
table(data$diabetes_t2)
```

```
## 
##     0     1 
## 36807  2160
```

``` r
data <- data %>%
   mutate(across(where(is.numeric), ~na_if(., -7)))
```

#### 3. Imputing data 

We are going to imput missing data using the `mice` package. Not going to talk about this really but it's easier to deal with a dataset with no missing data.


``` r
mice_imputed <- mice(data, m = 1)
```

```
## 
##  iter imp variable
##   1   1  DIS_DIAB_TYPE  PM_BMI_SR  PA_LEVEL_SHORT  SDC_EB_ABORIGINAL  SDC_EB_LATIN  SDC_EB_BLACK  DIS_LIVER_FATTY_EVER  SDC_MARITAL_STATUS  SDC_EDU_LEVEL  SDC_INCOME  HS_GEN_HEALTH  NUT_VEG_QTY  NUT_FRUITS_QTY  ALC_CUR_FREQ  SDC_BIRTH_COUNTRY  PA_SIT_AVG_TIME_DAY  SMK_CIG_STATUS  SLE_TIME  DIS_DIAB_FAM_EVER  HS_ROUTINE_VISIT_EVER  WH_HRT_EVER  DIS_STROKE_EVER  DIS_COPD_EVER  DIS_LC_EVER  DIS_IBS_EVER  WRK_FULL_TIME  WRK_STUDENT  PM_WEIGHT_SR_AVG  diabetes_t2  diabetes_gestat
##   2   1  DIS_DIAB_TYPE  PM_BMI_SR  PA_LEVEL_SHORT  SDC_EB_ABORIGINAL  SDC_EB_LATIN  SDC_EB_BLACK  DIS_LIVER_FATTY_EVER  SDC_MARITAL_STATUS  SDC_EDU_LEVEL  SDC_INCOME  HS_GEN_HEALTH  NUT_VEG_QTY  NUT_FRUITS_QTY  ALC_CUR_FREQ  SDC_BIRTH_COUNTRY  PA_SIT_AVG_TIME_DAY  SMK_CIG_STATUS  SLE_TIME  DIS_DIAB_FAM_EVER  HS_ROUTINE_VISIT_EVER  WH_HRT_EVER  DIS_STROKE_EVER  DIS_COPD_EVER  DIS_LC_EVER  DIS_IBS_EVER  WRK_FULL_TIME  WRK_STUDENT  PM_WEIGHT_SR_AVG  diabetes_t2  diabetes_gestat
##   3   1  DIS_DIAB_TYPE  PM_BMI_SR  PA_LEVEL_SHORT  SDC_EB_ABORIGINAL  SDC_EB_LATIN  SDC_EB_BLACK  DIS_LIVER_FATTY_EVER  SDC_MARITAL_STATUS  SDC_EDU_LEVEL  SDC_INCOME  HS_GEN_HEALTH  NUT_VEG_QTY  NUT_FRUITS_QTY  ALC_CUR_FREQ  SDC_BIRTH_COUNTRY  PA_SIT_AVG_TIME_DAY  SMK_CIG_STATUS  SLE_TIME  DIS_DIAB_FAM_EVER  HS_ROUTINE_VISIT_EVER  WH_HRT_EVER  DIS_STROKE_EVER  DIS_COPD_EVER  DIS_LC_EVER  DIS_IBS_EVER  WRK_FULL_TIME  WRK_STUDENT  PM_WEIGHT_SR_AVG  diabetes_t2  diabetes_gestat
##   4   1  DIS_DIAB_TYPE  PM_BMI_SR  PA_LEVEL_SHORT  SDC_EB_ABORIGINAL  SDC_EB_LATIN  SDC_EB_BLACK  DIS_LIVER_FATTY_EVER  SDC_MARITAL_STATUS  SDC_EDU_LEVEL  SDC_INCOME  HS_GEN_HEALTH  NUT_VEG_QTY  NUT_FRUITS_QTY  ALC_CUR_FREQ  SDC_BIRTH_COUNTRY  PA_SIT_AVG_TIME_DAY  SMK_CIG_STATUS  SLE_TIME  DIS_DIAB_FAM_EVER  HS_ROUTINE_VISIT_EVER  WH_HRT_EVER  DIS_STROKE_EVER  DIS_COPD_EVER  DIS_LC_EVER  DIS_IBS_EVER  WRK_FULL_TIME  WRK_STUDENT  PM_WEIGHT_SR_AVG  diabetes_t2  diabetes_gestat
##   5   1  DIS_DIAB_TYPE  PM_BMI_SR  PA_LEVEL_SHORT  SDC_EB_ABORIGINAL  SDC_EB_LATIN  SDC_EB_BLACK  DIS_LIVER_FATTY_EVER  SDC_MARITAL_STATUS  SDC_EDU_LEVEL  SDC_INCOME  HS_GEN_HEALTH  NUT_VEG_QTY  NUT_FRUITS_QTY  ALC_CUR_FREQ  SDC_BIRTH_COUNTRY  PA_SIT_AVG_TIME_DAY  SMK_CIG_STATUS  SLE_TIME  DIS_DIAB_FAM_EVER  HS_ROUTINE_VISIT_EVER  WH_HRT_EVER  DIS_STROKE_EVER  DIS_COPD_EVER  DIS_LC_EVER  DIS_IBS_EVER  WRK_FULL_TIME  WRK_STUDENT  PM_WEIGHT_SR_AVG  diabetes_t2  diabetes_gestat
```

```
## Warning: Number of logged events: 25
```

``` r
data_1 <- complete(mice_imputed, 1) 
write_csv(data_1, "data_imputed_1.csv")
```
