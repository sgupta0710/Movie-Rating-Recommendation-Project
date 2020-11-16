#Code provided in the edx course instructions

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
 #                                          title = as.character(title),
  #                                         genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)

#edx is the test on which we will build our model
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)  


#save the edx and validation data set 
save(edx, file = "edx.RData")
save(validation, file = "validation.RData")

#load other required libraries
library(ggplot2)
library(dplyr)

#lets explore the newly created datasets
str(edx)
str(validation)
summary(edx)

#check any NA value in the dataset
anyNA(edx)
anyNA(validation)

#data exploration analysis
#count of movies in dataset
edx %>% summarise(userscount = n_distinct(userId), moviescount = n_distinct(movieId))

#lets see the maximum number of ratings
edx %>% group_by(rating) %>%
  summarise(count = n()) %>%
  top_n(5) %>% arrange(desc(count))

#lets see number of rating per movie 
edx %>% group_by(movieId) %>% summarise(count = n()) %>% arrange(desc(count)) 

#lets see number of rating per movie - top 5
edx  %>% group_by(movieId, title) %>% summarise(count = n()) %>% arrange(desc(count)) %>% top_n(5) 

#lets plot the number 0f ratings per movie
edx %>% count(movieId) %>% ggplot(aes(n)) + 
  geom_histogram(color = "black" , fill= "light blue") + scale_x_log10() +
  ggtitle("Rating per movie") + theme_gray()

# lets see number of rating per user
edx %>% group_by(userId) %>%
  summarise(count = n()) %>% arrange(desc(count)) %>% top_n(10)

#lets plot the number of ratings per user
edx %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(color = "black", fill = "light blue") +
  scale_x_log10()+ ggtitle("Ratings per user") + theme_gray()

#lets see number of rating per genres. we need to seperate the generes while calculating the rating
edx %>% separate_rows(genres , sep = "\\|") %>%
  group_by(genres) %>% summarise(count = n()) %>%
  arrange(desc(count)) 


#lets plot the above number of rating per genres
edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>%
  summarise(count = n()) %>% arrange(desc(count))%>% ggplot(aes(genres, count))+ 
                                       geom_bar(aes(reorder(genres, -count), fill = genres), stat = "identity")+
                                       labs(title = "Number of rating for each genre") + 
  theme(axis.text.x = element_text(angle= 90, vjust = 50)) + theme_light() 




#Lets Calculate RMSE now
#function for RMSE is

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((predicted_ratings - true_ratings)^2, na.rm = TRUE))
}




#First rmse -where predicted are the average of all the values

mu_hat <- mean(edx$rating)
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

#lets keep on adding different RMSE to a table for better understandings

rmse_results <- data_frame(method = "Average model approach", RMSE= naive_rmse)
rmse_results

#second model - Movie effect model
#taking into account the effect of b_i, 
#where we will be subtracting the mean of rating received 
#by a particular movie from the actual rating of that movie

movie_avgs <- edx %>%
  group_by(movieId) %>% summarise(b_i = mean(rating - mu_hat))

#let plot and analyse
movie_avgs %>%
   ggplot(aes(b_i, color = I("black") ) ) +
  geom_histogram(bins= 10) + 
  labs(title = "Number of movies with the computed b_i") +
  ylab("Number of movies")

#lets see the predictions
predicted_ratings <- mu_hat + validation %>% left_join(movie_avgs , by = 'movieId') %>%
  pull(b_i)


#rmse for second model
rmse_with_bi <- RMSE(predicted_ratings, validation$rating)
rmse_with_bi

#save this result in rmse_results table
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie effect model" , RMSE = rmse_with_bi))
rmse_results


#third model - Movie and user effect model
#taking into account the effect of user also along with the movie

user_avgs <- edx %>% left_join(movie_avgs, by= 'movieId') %>%
  group_by(userId) %>% summarise(b_u = mean(rating - mu_hat - b_i))

#lets plot this and see
user_avgs %>% ggplot(aes(b_u, color= I("black"))) + geom_histogram(bins = 10) + labs("Number of movies with computed b_u") +
  ylab("Number of movies")

#lets calculate predictions
  
predicted_ratings <- validation %>% left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)
#predicted_ratings

#rmse for third model
rmse_with_bi_bu <- RMSE(predicted_ratings, validation$rating)
rmse_with_bi_bu

#save this result in the rmse_results table
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie and User effect model", RMSE = rmse_with_bi_bu))
rmse_results


#regularizing the movie and user effect
#lambda is a tuning parameter. using cross- validation here

lambdas <- seq(0,10,0.25)

#now we will find bias (b_i, b_u) for every lambda value

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot rmses vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  

# The optimal lambda                                                             
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                     RMSE = min(rmses)))

# Check final results
rmse_results %>% knitr::kable()

#loading latex for knitting
tinytex::install_tinytex()
