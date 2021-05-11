library(tidyverse)
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

batch_size <- 64
epochs <- 50
steps_per_epoch <- 5

# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(x_train)) %>%
  # layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu", name = "bottleneck") %>%
  # layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = ncol(x_train), activation = "linear")

# view model layers
summary(model)

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = optimizer_adam(
    lr = 0.001 ))

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

## This code works
history <- model %>%
  fit_generator(generator = keras:::as_generator.function(batch_generator(x_train, batch_size)),
                steps_per_epoch = steps_per_epoch,
                
                validation_data = keras:::as_generator.function(batch_generator(x_val, batch_size)),
                validation_steps = floor(steps_per_epoch/2),
                
                epochs = epochs, 
                view_metrics = TRUE, 
                callbacks=list(callback_early_stopping(
                  monitor = "val_loss",
                  min_delta = 0.01,
                  patience = 10,
                  restore_best_weights = TRUE
                )),
                verbose = 2)

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




