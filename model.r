library(plyr)
library(nnet)
library(caret)
library(R.utils)

set.seed(12345)

# rows and columns for each figure
all.rows <- 28
all.cols <- 28

every <- 1
rows <- floor(all.rows / every)
cols <- floor(all.cols / every)

if (!exists("train.data")) {
    max.rows <- 10000
    gunzip("train.csv.gz", overwrite=TRUE)
    train.data <- read.csv("train.csv",
                           header = TRUE,
                           nrows=max.rows)
    train.data$label <- as.factor(train.data$label)
    unlink("train.csv")
}

# convert training data into list of labels with a corresponding matrix
get.list <- function(df = train.data) {
    l <- alply(.data = df, .margins = 1, .fun = function (row) {
        r <- list()
        r$label <- row$label
        r$figure <- matrix(lapply(row[1, all.rows*all.cols+1], function(v) v/255), all.rows, all.cols)
        
        return(r)
    })
    return(l)
}

# train.list <- get.list(train.data)
# 
# train.list <- lapply(train.list, function (row) {
#     .cols = seq(1, cols) * every
#     .rows = seq(1, rows) * every
#     
#     row$figure <- matrix(row$figure[.rows, .cols], rows, cols)
#     return(row)
# })
# 
# train.data <- ldply(train.list, function (row) {
#     return(list(label = row$label, unlist(alply(row$figure, 1, function (l) alply(l,1)))))
# })

figure.columns <- seq(2, (1+rows*cols))

train.data[, figure.columns] <- train.data[, figure.columns] / 255

partition <- createDataPartition(train.data[, "label"], p = 0.8, list = FALSE)
train.batch <- train.data[partition,]
test.batch <- train.data[-partition,]

start.time <- Sys.time()
m <- nnet(formula = label ~ .,
          data = train.batch,
          MaxNWts = 100000,
          maxit = 200,
          size = (rows*cols/rows))
print(Sys.time() - start.time)

test.batch$prediction <- predict(object = m, newdata = test.batch, type="class")
r <- test.batch[, c("label", "prediction")]

num.correct <- NROW(which(test.batch$label == test.batch$prediction))
printf("%03.1f%% correct", 100 * num.correct / NROW(test.batch))

# Now run our model against the competition data
if (!exists("competition.data")) {
    gunzip("test.csv.gz", overwrite=TRUE)
    competition.data <- read.csv("test.csv", header = TRUE)
    unlink("test.csv")
}

predictions <- predict(object = m, newdata = competition.data, type = "class")

output <- data.frame(ImageId=1:nrow(competition.data), Label = predictions)
write.table(output, file="output.csv", sep = ",", quote = FALSE, row.names = FALSE, col.names = TRUE)
gzip("output.csv", overwrite = TRUE)
