###############################################################################################
# packages
###############################################################################################

library(caret)
library(xgboost)
library(jsonlite)
library(lubridate)
library(knitr)
library(Rmisc)
library(scales)
library(countrycode)
library(highcharter)
library(glmnet)
library(keras)
library(magrittr)
library(tidyverse)

###############################################################################################
# get the data
###############################################################################################

setwd("/KruseA/Desktop/")
tr <- read_csv("train.csv")
te <- read_csv("test.csv")
subm <- read_csv("sample_submission.csv")

###############################################################################################
# flatten cols
###############################################################################################

flatten_json <- . %>% 
  str_c(., collapse = ",") %>% 
  str_c("[", ., "]") %>% 
  fromJSON(flatten = T)

parse <- . %>% 
  bind_cols(flatten_json(.$device)) %>%
  bind_cols(flatten_json(.$geoNetwork)) %>% 
  bind_cols(flatten_json(.$trafficSource)) %>% 
  bind_cols(flatten_json(.$totals)) %>% 
  select(-device, -geoNetwork, -trafficSource, -totals)

tr <- parse(tr)
te <- parse(te)

###############################################################################################
# tidy up data
###############################################################################################

# remove missing col
tr %<>% select(-one_of("campaignCode"))

# remove cols with no info
fea_uniq_values <- sapply(tr, n_distinct)
(fea_del <- names(fea_uniq_values[fea_uniq_values == 1]))
tr %<>% select(-one_of(fea_del))
te %<>% select(-one_of(fea_del))

# put cols with unknown to NA
is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
                                  "(not set)", "<NA>", "unknown.unknown",  "(none)")

tr %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))
te %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))

# create target var
y <- as.numeric(tr$transactionRevenue)
tr$transactionRevenue <- NULL
summary(y)

# replace NA with 0 for target var
y[is.na(y)] <- 0
summary(y)

###############################################################################################
# get data reading for modeling
###############################################################################################

grp_mean <- function(x, grp) ave(x, grp, FUN = function(x) mean(x, na.rm = TRUE))
id <- te[, "fullVisitorId"]
tri <- 1:nrow(tr)

tr_te <- tr %>% 
  bind_rows(te) %>% 
  mutate(date = ymd(date),
         year = year(date) %>% factor(),
         month = year(date) %>% factor(),
         week = week(date) %>% factor(),
         day = day(date) %>% factor(),
         hits = as.integer(hits),
         pageviews = as.integer(pageviews),
         bounces = as.integer(bounces),
         newVisits = as.integer(newVisits),
         isMobile = ifelse(isMobile, 1L, 0L),
         isTrueDirect = ifelse(isTrueDirect, 1L, 0L),
         adwordsClickInfo.isVideoAd = ifelse(!adwordsClickInfo.isVideoAd, 0L, 1L)) %>% 
  select(-date, -fullVisitorId, -visitId, -sessionId) %>% 
  mutate_if(is.character, factor) %>% 
  mutate(pageviews_mean_vn = grp_mean(pageviews, visitNumber),
         hits_mean_vn = grp_mean(hits, visitNumber),
         pageviews_mean_country = grp_mean(pageviews, country),
         hits_mean_country = grp_mean(hits, country),
         pageviews_mean_city = grp_mean(pageviews, city),
         hits_mean_city = grp_mean(hits, city))

fn <- funs(mean, var, .args = list(na.rm = TRUE))

sum_by_dom <- tr_te %>%
  select(networkDomain, hits, pageviews) %>% 
  group_by(networkDomain) %>% 
  summarise_all(fn) 

sum_by_vn <- tr_te %>%
  select(visitNumber, hits, pageviews) %>% 
  group_by(visitNumber) %>% 
  summarise_all(fn) 

sum_by_country <- tr_te %>%
  select(country, hits, pageviews) %>% 
  group_by(country) %>% 
  summarise_all(fn) 

sum_by_city <- tr_te %>%
  select(city, hits, pageviews) %>% 
  group_by(city) %>% 
  summarise_all(fn) 

sum_by_medium <- tr_te %>%
  select(medium, hits, pageviews) %>% 
  group_by(medium) %>% 
  summarise_all(fn) 

sum_by_source <- tr_te %>%
  select(source, hits, pageviews) %>% 
  group_by(source) %>% 
  summarise_all(fn) 


tr_te %<>% 
  left_join(sum_by_city, by = "city", suffix = c("", "_city")) %>% 
  left_join(sum_by_country, by = "country", suffix = c("", "_country")) %>% 
  left_join(sum_by_dom, by = "networkDomain", suffix = c("", "_dom")) %>% 
  left_join(sum_by_medium, by = "medium", suffix = c("", "medium")) %>% 
  left_join(sum_by_source, by = "source", suffix = c("", "_source")) %>% 
  left_join(sum_by_vn, by = "visitNumber", suffix = c("", "_vn")) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) 

#  feature eng
tr_te$operatingSystem = as.character(tr_te$operatingSystem)
tr_te$operatingSystem <- ifelse(tr_te$operatingSystem %in% c("Android","Macintosh","Linux",
                                                             "Windows","iOS","Chrome OS"),
                                tr_te$operatingSystem,"Others")

###############################################################################################
# Run XGB (saved till here)
###############################################################################################

# factors to integers for modeling
tr_te_xgb <- tr_te %>% 
  mutate_if(is.factor, as.integer)

# get test dataset
dtest <- xgb.DMatrix(data = data.matrix(tr_te_xgb[-tri, ]))

# get training dataset
tr_te_xgb <- tr_te_xgb[tri, ]

# split training dataset into real training and eval
idx <- ymd(tr$date) < ymd("20170701")

# traing dataset with idx == T
dtr <- xgb.DMatrix(data = data.matrix(tr_te_xgb[idx, ]), label = log1p(y[idx]))

# eval dataset with idx == F
dval <- xgb.DMatrix(data = data.matrix(tr_te_xgb[!idx, ]), label = log1p(y[!idx]))

# get full training dataset for testing the model
dtrain <- xgb.DMatrix(data = data.matrix(tr_te_xgb), label = log1p(y))





cols <- colnames(tr_te_xgb)
rm(tr_te_xgb); invisible(gc)

# handle unbalance with SMOT


# train model
p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 4,
          eta = 0.05,
          max_depth = 8,
          min_child_weight = 3,
          gamma = 0,
          subsample = 0.8,
          colsample_bytree = 0.5,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtr, p$nrounds, list(val = dval), print_every_n = 100, early_stopping_rounds = 100)

# var importance from model
xgb.importance(cols, model = m_xgb) %>% 
  xgb.plot.importance(top_n = 20)

# test prediction again traingset
pred_xgb_tr <- predict(m_xgb, dtrain)

# make prediction on test dataset
pred_xgb <- predict(m_xgb, dtest)

# submit file
check = as.data.frame(pred_xgb)
colnames(check) = "y"
check$y = ifelse(check$y < 0, 0, expm1(check$y))
id2 = as.vector(id$fullVisitorId)
check$fullVisitorId = id2
check = check %>%
  dplyr::group_by(fullVisitorId) %>% 
  dplyr::summarise(y = log1p(sum(y)))

  
filer = read_csv("sample_submission.csv")
filer2 = merge(filer, check1, by = "fullVisitorId", all.x = T)
filer2$PredictedLogRevenue = NULL
colnames(filer2) = colnames(filer)
filer2$PredictedLogRevenue = round(filer2$PredictedLogRevenue, 5)
write.csv(filer2,"xgb_gs.csv",row.names = F)
