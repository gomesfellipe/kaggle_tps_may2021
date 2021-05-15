# knitr::opts_chunk$set(echo = TRUE)

library(tidyverse)
library(tidymodels)
library(keras)


# DAE ---------------------------------------------------------------------

train <- read_csv("train.csv")
test <- read_csv("test.csv")
full <- bind_rows(train, test)

samples <- sample(1:2, size = nrow(full), replace = T, prob = c(0.8, 0.2))

# set training data
x_train <- 
  full %>% 
  filter(samples==1)  %>% 
  select(-id, -target) %>% 
  mutate_all(scale) %>% 
  as.matrix()

x_val <- 
  full %>% 
  filter(samples==2)  %>% 
  select(-id, -target) %>% 
  mutate_all(scale) %>% 
  as.matrix()

# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train)) %>%
  #layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu", name = "bottleneck") %>%
  #layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = ncol(x_train), activation = "linear")

summary(model)

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = optimizer_adam(
    lr = 0.001 ))


# Columnwise swap noise 
batch_generator <- function(data, batch_size, perc=.15)
{
  function() {
    data_original <- data
    
    ids <- sample(1:nrow(data), batch_size, replace = FALSE)
    
    for(j in 1:ncol(data)){
      to_shuffle <- sample(c(T, F), size = nrow(data[ids, ]), replace = T, prob = c(perc, 1-perc))
      data[ids,][which(to_shuffle), j] <- sample(data[ids,][which(to_shuffle), j])
    }
    list(
      as.matrix(data[ids, ]),
      as.matrix(data_original[ids, ])
    )
  }
}

# Fixed parameters
batch_size <- 128
epochs <- 100

#batch_generator_val <- function(x_val, batch_size, perc=.15)
#{
#  function() {
x_val_generator <- x_val

ids <- sample(1:nrow(x_val_generator), batch_size, replace = FALSE)

for(j in 1:ncol(x_val_generator)){
  to_shuffle <- sample(c(T, F), size = nrow(x_val_generator[ids, ]), replace = T, prob = c(.15, 1-.15))
  x_val_generator[ids,][which(to_shuffle), j] <- sample(x_val_generator[ids,][which(to_shuffle), j])
}
#    list(
#      as.matrix(x_val_generator[ids, ]),
#      as.matrix(x_val[ids, ])
#    )
#  }
#}

## This code works
history <- model %>%
  fit_generator(generator = keras:::as_generator.function(batch_generator(x_train, batch_size)),
                steps_per_epoch = 100,
                
                #validation_data = keras:::as_generator.function(batch_generator(x_val, batch_size)),
                validation_steps = 10,
                validation_data = list(x_val_generator[ids, ], x_val[ids, ]),
                
                epochs = epochs, 
                view_metrics = TRUE, 
                callbacks=list(callback_early_stopping(
                  monitor = "val_loss",
                  min_delta = 0.01,
                  patience = 10,
                  restore_best_weights = TRUE
                )),
                verbose = 2)

# Plot history

as_metrics_df = function(history) {
  
  # create metrics data frame
  df <- as.data.frame(history$metrics)
  
  # pad to epochs if necessary
  pad <- history$params$epochs - nrow(df)
  pad_data <- list()
  for (metric in history$params$metrics)
    pad_data[[metric]] <- rep_len(NA, pad)
  df <- rbind(df, pad_data)
  
  # return df
  df
}

as_metrics_df(history) %>% 
  mutate(epochs = 1:nrow(.)) %>% 
  gather(key, val, -epochs) %>% 
  ggplot(aes(x = epochs, y = val, col=key)) +
  geom_point()+
  geom_smooth(se = F)+
  theme_bw()


mse.ae2 <- evaluate(model, x_train, x_train)
mse.ae2

intermediate_layer_model <- 
  keras_model(inputs = model$input,
              outputs = get_layer(model, "bottleneck")$output)

all_data <- 
  full %>% 
  select(-id, -target) %>% 
  mutate_all(scale) %>% 
  as.matrix()

intermediate_output <- predict(intermediate_layer_model, all_data)

daeta_full <- bind_cols(
  full %>% select(id, target),
  as_tibble(intermediate_output)
)

daeta_train <- daeta_full %>% filter(id %in% train$id)
daeta_test <- daeta_full %>% filter(id %in% test$id)


titanic_folds <- daeta_train %>% rsample::vfold_cv(v = 4,repeats = 2, strata = target)

titanic_recipe <- recipe(target~., data=daeta_train) %>%
  step_rm(id) %>%
  step_zv(all_predictors())

lgbm_spec <- boost_tree(
  mtry = tune()
) %>%  
  set_engine("xgboost") %>% 
  set_mode("classification")

lgbm_wflow <- workflow() %>% 
  add_recipe(titanic_recipe) %>% 
  add_model(lgbm_spec) 

lgbm_params <- 
  dials::parameters(
    # learn_rate(),           # learning_rate
    # trees()                 # num_iterations
    # min_n(),                  # min_data_in_leaf
    # tree_depth(c(1, 16)),     # max_depth
    # sample_prop(c(0.1, 0.9)), # bagging_fraction (to sample_size)
    mtry()                   # feature_fraction
    # loss_reduction(),         # min_gain_to_split
    # reg_lambda = dials::penalty(),
    # reg_alpha = dials::penalty(),
    # num_leaves = dials::tree_depth(c(15, 255))
  )

lgbm_params <- lgbm_params %>% 
  finalize(daeta_train)

set.seed(314)
lgbm_res <-
  lgbm_wflow %>% 
  tune_bayes(
    resamples = titanic_folds,
    param_info = lgbm_params,
    # initial = 9,
    iter = 9999,
    metrics = metric_set(mn_log_loss),
    control = control_bayes(no_improve = 9999, time_limit=10,
                            verbose = TRUE, save_pred = FALSE)
  )

lgbm_res %>%
  collect_metrics()

autoplot(lgbm_res)

lgbm_final_wflow_tun <- 
  finalize_workflow(
    lgbm_wflow,
    select_best(lgbm_res, metric = 'mn_log_loss') )



# rascunho ----------------------------------------------------------------



batch_generator <- function(data, batch_size)
{
  function() {
    ids <- sample(1:nrow(data), batch_size, replace = FALSE)
    
    batch_data <- as.matrix(data[ids, ])
    
    for(j in 1:ncol(batch_data)){
      to_shufle = sample(c(TRUE, FALSE), nrow(batch_data[,j]), prob = c(.15, .85), replace = T)
      batch_data[which(to_shufle), j] = sample(batch_data[which(to_shufle), j][[1]])
    }
    
    list(
      batch_data
    )
  }
}

library(keras)

batch_size <- 64
epochs <- 200
steps_per_epoch <- 5

batch_generator <- function(data, batch_size)
{
  function() {
    ids <- sample(1:nrow(data), batch_size, replace = FALSE)
    list(
      as.matrix(data[ids, -5]),
      to_categorical(as.integer(data[ids, 5]) - 1)
    )
  }
}

iris
batch_generator(data = iris, batch_size = batch_size)




