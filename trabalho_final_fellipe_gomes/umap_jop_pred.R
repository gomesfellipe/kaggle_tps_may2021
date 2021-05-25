library(dplyr)
full <- readr::read_csv("train.csv")

set.seed(314)
test <- 
  full %>% 
  group_by(target) %>% 
  sample_frac(0.2) %>% 
  ungroup()

train <- full %>% filter(!id %in% test$id) 

umap_tps <- readRDS("umap_tps_full.rds")

umap_tps_pred <- predict(umap_tps, select(test, -id, -target))

saveRDS(umap_tps_pred, "umap_tps_pred.rds")
