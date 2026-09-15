library(tidyverse)
library(skimr)
library(caret)
library(knitr)
library(recosystem)


# Data Preparation Process ------------------------------------------------


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


set.seed(1, sample.kind = "Rounding")

index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train_set <- edx[-index,]
edx_test_set <- edx[index,]

edx_train_set <- edx_train_set %>%
  semi_join(edx_test_set, by = "movieId")

edx_test_set <- edx_test_set %>%
  semi_join(edx_train_set, by = "movieId")


# Exploratory Data Analysis (EDA) ---------------------------------------

# Basic Structure & Statistic Summary ---------------------------------------------------------
# First 10 rows of the dataset
head(edx, n = 9)

# Dimensions
glimpse(edx)

# Summary Statistic
summary(edx)
skim(edx) # more advanced summary statistic of the dataset

# Rating Distribution -----------------------------------------------------
edx_rating <- edx %>%
  count(rating, name = "num_rating") %>%
  mutate(percentage = round(num_rating / sum(num_rating) * 100, 2)) %>%
  arrange(desc(rating))

edx_rating %>%
  ggplot(aes(x = factor(rating), y = percentage)) +
  geom_col(fill = "#0072B2",
           color = "black") +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title = "Distribution of Ratings",
    x = "Rating",
    y = "Percentage"
  )

edx_rating %>%
  mutate(percentage = sprintf("%.2f%%", percentage))

edx_rating %>%
  summarise(Above_three = sprintf("%.2f%%", round(sum(percentage[1:5]))),
            Below_three = sprintf("%.2f%%", round(sum(percentage[6:10]))))


#  Sparsity Analysis | Ratings per Movie | Ratings per User | Matr --------
# Count how many ratings each movie received and plot the distribution
edx |>
  count(movieId, name = "n_ratings") |>
  ggplot(aes(n_ratings)) +
  geom_histogram(bins = 50,
                 fill = "#0072B2",
                 color = "black") +
  scale_x_log10() +
  labs(
    title = "Number of Ratings per Movie (log scale)",
    x = "Number of Ratings",
    y = "Number of Movies"
  )

# Count how many ratings each user submit and plot the distribution
edx |>
  count(userId, name = "n_ratings") |>
  ggplot(aes(n_ratings)) +
  geom_histogram(bins = 50,
                 fill = "#0072B2",
                 color = "black") +
  scale_x_log10() +
  labs(
    title = "Number of Ratings per User (log scale)",
    x = "Number of Ratings",
    y = "Number of Users",
  )

# Calculate number of unique users, movies and number of ratings
num_users <- n_distinct(edx$userId)
num_movies <- n_distinct(edx$movieId)
num_ratings <- nrow(edx)

# Calculate proportion of missing interactions in the dataset
sparsity <- 1 - (num_ratings / (num_users * num_movies))

# Summary table 
tibble(
  "Unique Users" = num_users,
  "Unique Movies" = num_movies,
  "Unique Ratings" = num_ratings,
  "Possible Combinations" = num_users * num_movies,
  "Sparsity" = paste0(round(sparsity * 100, 2), "%")
) %>% 
  knitr::kable(caption = "Sparsity of the User-Movie Rating Matrix")



# Genre Distribution ------------------------------------------------------

# Separate combined genres into individual genre labels
# and count how many movies have that genre
n_movies <- edx %>%
  distinct(movieId) %>%
  nrow()

edx_genre <- edx %>%
  distinct(movieId, genres) %>%
  separate_rows(genres, sep = "\\|") %>%
  count(genres, name = "num_movies") %>%
  mutate(percentage = num_movies / n_movies * 100) %>%
  arrange(desc(percentage))

edx_genre %>%
  ggplot(aes(x = percentage, y = reorder(genres, percentage))) +
  geom_col(stat = "identity", fill = "#0072B2", color = "black") +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title = "Genre Frequency in the MovieLens Dataset",
    x = "Percentage",
    y = "Genre"
  )

edx_genre %>%
  mutate(percentage = sprintf("%.2f%%", percentage))


# Temporal Trends ---------------------------------------------------------


# Convert Unix time stamp to a readable data format
edx_temp <- edx %>%
  mutate(date = as_datetime(timestamp),
         year = year(date))

# Count number of ratings per year
ratings_per_year <- edx_temp %>%
  group_by(year) %>%
  summarise(num_ratings = n(),
            mean_rating = mean(rating))

# Plot annual rating volume
ratings_per_year %>%
  ggplot(aes(x = year, y = num_ratings)) +
  geom_point(color = "#0072B2", size = 3) +
  geom_line(color = "#0072B2", linewidth = 1) +
  scale_x_continuous(breaks = round(seq(min(ratings_per_year$year), 
                                        max(ratings_per_year$year), by = 2), 1)) +
  scale_y_continuous(breaks = round(seq(min(ratings_per_year$num_ratings), 
                                        max(ratings_per_year$num_ratings), by = 100000), 1)) +
  labs(
    title = "Annual Rating Volume (1995-2009)",
    x = "Year",
    y = "Rating Count"
  )

# Plot mean rating score per year
ratings_per_year %>%
  ggplot(aes(x = year, y = mean_rating)) +
  geom_point(color = "#0072B2", size = 3) +
  geom_line(color = "#0072B2", linewidth = 1) +
  scale_x_continuous(breaks = round(seq(min(ratings_per_year$year), 
                                        max(ratings_per_year$year), by = 1), 1)) +
  scale_y_continuous(breaks = round(seq(min(ratings_per_year$mean_rating), 
                                        max(ratings_per_year$mean_rating), by = 0.09), 1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Mean Rating Score per Year (1995-2009)",
    x = "Year",
    y = "Mean Rating Score"
  )


# Concentration of Rating Activity Across Users and Movies ----------------

# Preparation
user_activity <- edx %>%
  group_by(userId) %>%
  summarize(rating_count = n(), .groups = "drop") %>%
  arrange(desc(rating_count)) %>%
  mutate(cumsum_ratings = cumsum(rating_count))

item_popularity <- edx %>%
  group_by(movieId) %>%
  summarize(rating_count = n(), .groups = "drop") %>%
  arrange(desc(rating_count)) %>%
  mutate(cumsum_ratings = cumsum(rating_count))

total_ratings <- sum(user_activity$rating_count)

# User Activity
# Top 10% users
top_10pct_users <- round(0.1 * nrow(user_activity))
top_10_share <- user_activity %>%
  slice(1:top_10pct_users) %>%
  summarize(share = sum(rating_count) / total_ratings) %>%
  pull(share)

# Bottom 30% users
bottom_30pct_users <- floor(0.3 * nrow(user_activity))
max_bottom_30_users <- user_activity %>%
  arrange(rating_count) %>%
  slice(1:bottom_30pct_users) %>%
  summarise(max_val = max(rating_count)) %>%
  pull(max_val)

# Visualisation of cumulative share of ratings by user activity rank
user_activity %>%
  mutate(
    user_rank  = row_number() / n(),
    cum_share  = cumsum_ratings / total_ratings
  ) %>%
  ggplot(aes(x = user_rank, y = cum_share)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = 0.1, linetype = "dashed", color = "red") +
  geom_hline(yintercept = top_10_share, linetype = "dashed", color = "red") +
  annotate("text", x = 0.15, y = top_10_share - 0.05,
           label = paste0("Top 10% ->", 
                          round(top_10_share * 100, 1), "%"),
           color = "red", size = 3.5, hjust = 0) +
  labs(
    title = "Cumulative Share of Ratings by User Activity Rank",
    x     = "Proportion of Users",
    y     = "Cumulative Share of Total Ratings"
  ) +
  theme_minimal()

# Item Popularity

# Top 5% items
top_5pct_items <- round(0.05 * nrow(item_popularity))
top_5_share <- item_popularity %>%
  slice(1: top_5pct_items) %>%
  summarize(share = sum(rating_count) / total_ratings) %>%
  pull(share)

# Bottom 20% items
bottom_20pct_items <- floor(0.2 * nrow(item_popularity))
max_bottom_20_items <- item_popularity %>%
  arrange(rating_count) %>%
  slice(1:bottom_20pct_items) %>%
  summarise(max_val = max(rating_count)) %>%
  pull(max_val)

# Visualisation of cumulative share of ratings by movie popularity rank
item_popularity %>%
  mutate(
    item_rank = row_number() / n(),
    cum_share = cumsum_ratings / total_ratings
  ) %>%
  ggplot(aes(x = item_rank, y = cum_share)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
  geom_hline(yintercept = top_5_share, linetype = "dashed", color = "red") +
  annotate("text", x = 0.08, y = top_5_share - 0.05,
           label = paste0("Top 5% → ", 
                          round(top_5_share * 100, 1), "%"),
           color = "red", size = 3.5, hjust = 0) +
  labs(
    title = "Cumulative Share of Ratings by Movie Popularity Rank",
    x     = "Proportion of Movies",
    y     = "Cumulative Share of Total Ratings"
  ) +
  theme_minimal()


# Model Development -------------------------------------------------------
# RMSE definition ---------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}
rmse_result <- tibble(Model = character(), RMSE = numeric())
options(pillar.sigfig = 5)

# Global mean model -------------------------------------------------------
mu_hat <- mean(edx_train_set$rating, na.rm = TRUE)
global_rmse <- RMSE(edx_test_set$rating, mu_hat)
rmse_result <- rmse_result %>% add_row(Model = "Global Mean:", RMSE = global_rmse)

cat(sprintf("Model RMSE: %.5f\n", global_rmse))

# Movie Effect Model ------------------------------------------------------
movie_effect <- edx_train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu_hat, na.rm = TRUE))

movie_predict <- edx_test_set %>%
  left_join(movie_effect, by = "movieId") %>%
  mutate(pred = mu_hat + b_i) %>%
  pull(pred)

movie_effect_rmse <- RMSE(edx_test_set$rating, movie_predict)
rmse_result <- rmse_result %>% add_row(Model = "Movie Effect Model:", RMSE = movie_effect_rmse)

cat(sprintf("Movie effect model RMSE: %.5f\n", movie_effect_rmse))



# Movie and User Effect Model ---------------------------------------------
user_effect <- edx_train_set %>%
  left_join(movie_effect, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i), .groups = "drop")

movie_user_predict <- edx_test_set %>% 
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

movie_user_rmse <- RMSE(edx_test_set$rating, movie_user_predict)
rmse_result <- rmse_result %>% add_row(Model = "Movie and User Effects Model", 
                                       RMSE = movie_user_rmse)

cat(sprintf("Movie and User effects model RMSE: %.5f\n", movie_user_rmse))



# Temporal Effect ---------------------------------------------------------
temporal_effect <- edx_train_set %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu_hat - b_i - b_u), .groups = "drop")

temporal_model_pred <- edx_test_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  left_join(temporal_effect, by = "date") %>%
  mutate(pred = mu_hat + b_i + b_u + b_t) %>%
  pull(pred)

temporal_rmse <- RMSE(edx_test_set$rating, temporal_model_pred)
rmse_result <- rmse_result %>% add_row(Model = "Temporal Effect Model", 
                                       RMSE = temporal_rmse)

cat(sprintf("Temporal effect model RMSE: %.5f\n", temporal_rmse))



# Genre Effect ------------------------------------------------------------
genre_effect <- edx_train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  left_join(temporal_effect, by = "date") %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u - b_t))

genre_model_pred <- edx_test_set %>%
  mutate(row_num = row_number(),
         date = round_date(as_datetime(timestamp), unit = "week")) %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  left_join(temporal_effect, by = "date") %>%
  left_join(genre_effect, by = "genres") %>%
  group_by(row_num) %>%
  summarize(pred = mean(mu_hat + b_i + b_u + b_t + b_g)) %>%
  arrange(row_num) %>%
  pull(pred)

genre_rmse <- RMSE(edx_test_set$rating, genre_model_pred)
rmse_result <- rmse_result %>% add_row(Model = "Genre Effect Model", 
                                       RMSE = genre_rmse)

cat(sprintf("Genre effect model RMSE: %.5f\n", genre_rmse))



# Regularization ----------------------------------------------------------
# Pre-computations
train_expanded <- edx_train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week"))

train_expanded_b_g <- train_expanded %>%
  separate_rows(genres, sep = "\\|")

test_expanded <- edx_test_set %>%
  mutate(row_num = row_number(),
         date = round_date(as_datetime(timestamp), unit = "week")) %>%
  separate_rows(genres, sep = "\\|")

# Model Regularization function
compute_rmse <- function(lambda) {
  movie_effect <- edx_train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
  
  user_effect <- edx_train_set %>%
    left_join(movie_effect, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i) / (n() + lambda))
  
  temporal_effect <- train_expanded %>%
    left_join(movie_effect, by = "movieId") %>%
    left_join(user_effect, by = "userId") %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - mu_hat - b_u - b_i) / (n() + lambda))
  
  genre_effect <- train_expanded_b_g %>%
    left_join(movie_effect,  by = "movieId") %>%
    left_join(user_effect, by = "userId") %>%
    left_join(temporal_effect, by = "date") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u - b_t) / (n() + lambda))
  
  pred <- test_expanded %>%
    left_join(movie_effect, by = "movieId") %>%
    left_join(user_effect, by = "userId") %>%
    left_join(temporal_effect, by = "date") %>%
    left_join(genre_effect, by = "genres") %>%
    group_by(row_num) %>%
    summarize(pred = mean(mu_hat + b_i + b_u + b_t + b_g)) %>%
    arrange(row_num) %>%
    pull(pred)
  
  RMSE(edx_test_set$rating, pred)
}

# First stage - selecting the appropriate lambda
lambdas_s1 <- seq(0, 10, 0.25)
rmses_s1 <- sapply(lambdas_s1, compute_rmse)
best_lambda <- lambdas_s1[which.min(rmses_s1)]
cat(sprintf("Stage 1 best lambda: %.2f\n", best_lambda))

rmse_result <- rmse_result %>%
  add_row(Model = "Regularized Genre Effect Model (first stage grid)",
          RMSE = min(rmses_s1))

# Second stage - tuning around best lambda
lambda_s2 <- seq(best_lambda - 1, best_lambda + 1, 0.1)
rmses_s2 <- sapply(lambda_s2, compute_rmse)
best_lambda <- lambda_s2[which.min(rmses_s2)]
cat(sprintf("Stage 2 best lambda: %.2f\n", best_lambda))

rmse_result <- rmse_result %>%
  add_row(Model = "Regularized Genre Effect Model (second stage grid)",
          RMSE = min(rmses_s2))



# Matrix Factorization ----------------------------------------------------
# Refit all terms with best_lambda
movie_effect_lmb <- edx_train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_hat) / (n() + best_lambda))

user_effect_lmb <- edx_train_set %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i) / (n() + best_lambda))

temporal_effect_lmb <- edx_train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  left_join(user_effect_lmb, by = "userId") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu_hat - b_i - b_u) / (n() + best_lambda))

genre_effect_lmb <- edx_train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  left_join(user_effect_lmb, by = "userId") %>%
  left_join(temporal_effect_lmb, by = "date") %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u - b_t) / (n() + best_lambda))

# Compute residuals on train set
residuals <- edx_train_set %>%
  mutate(row_num = row_number(),
         date = round_date(as_datetime(timestamp), unit = "week")) %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  left_join(user_effect_lmb, by = "userId") %>%
  left_join(temporal_effect_lmb, by = "date") %>%
  left_join(genre_effect_lmb, by = "genres") %>%
  group_by(row_num) %>%
  summarize(movieId = first(movieId),
            userId = first(userId),
            resid = first(rating)  - mean(mu_hat + b_i + b_u + b_t + b_g))

# Initialization of Matrix Factorization Model
r <- Reco()

train_reco <- data_memory(residuals$userId, residuals$movieId,
                          residuals$resid, index1 = TRUE)

set.seed(1)
opts_tune_2 <- r$tune(train_reco, opts = list(
  dim = c(30, 35, 40, 45, 50),
  costp_l1 = 0,
  costp_l2 = 0.01,
  costq_l1 = 0,
  costq_l2 = 0.1,
  lrate = 0.05,
  niter = 30,
  nthread = 1,
  seed = 1,
  verbose = FALSE
))

set.seed(1)
r$train(train_reco,
        opts = c(opts_tune_2$min,
                 list(niter = 30, 
                      nthread = 1,
                      seed = 1,
                      verbose = TRUE)))

# Compute predictions on test set
bias_pred <- edx_test_set %>%
  mutate(row_num = row_number(),
         date = round_date(as_datetime(timestamp), unit = "week")) %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  left_join(user_effect_lmb, by = "userId") %>%
  left_join(temporal_effect_lmb, by = "date") %>%
  left_join(genre_effect_lmb, by = "genres") %>%
  group_by(row_num) %>%
  summarize(pred = first(mu_hat) + first(b_i) + first(b_u) + first(b_t) + mean(b_g)) %>%
  arrange(row_num) %>%
  pull(pred)

test_reco <- data_memory(edx_test_set$userId, 
                         edx_test_set$movieId, 
                         index1 = TRUE)
mf_pred <- r$predict(test_reco, out_memory())

final_pred <- bias_pred + mf_pred

mf_rmse <- RMSE(edx_test_set$rating, final_pred)

rmse_result <- rmse_result %>%
  add_row(Model = "Matrix Factorization Model", RMSE = mf_rmse)

cat(sprintf("Matrix Factorization RMSE: %.5f\n", mf_rmse))



# Holdout Set Evaluation --------------------------------------------------

holdout_bias_pred <- final_holdout_test %>%
  mutate(row_num = row_number(),
         date = round_date(as_datetime(timestamp), unit = "week")) %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(movie_effect_lmb, by = "movieId") %>%
  left_join(user_effect_lmb, by = "userId") %>%
  left_join(temporal_effect_lmb, by = "date") %>%
  left_join(genre_effect_lmb, by = "genres") %>%
  group_by(row_num) %>%
  summarize(pred = first(mu_hat) + first(b_i) + first(b_u) + first(b_t) + mean(b_g)) %>%
  arrange(row_num) %>%
  pull(pred)

holdout_reco <- data_memory(final_holdout_test$userId, 
                            final_holdout_test$movieId, 
                            index1 = TRUE)
holdout_mf_pred <- r$predict(holdout_reco, out_memory())

final_holdout_pred <- holdout_bias_pred + holdout_mf_pred

holdout_mf_rmse <- RMSE(final_holdout_test$rating, final_holdout_pred)

rmse_result <- rmse_result %>%
  add_row(Model = "Matrix Factorization Model (final_holdout_set)", RMSE = holdout_mf_rmse)

cat(sprintf("Matrix Factorization Model (final_holdout_set): %.5f\n", holdout_mf_rmse))



