library(plyr)
library(nnet)
library(caret)
library(R.utils)
library(foreach)

set.seed(12345)

# rows and columns for each figure
all.rows <- 28
all.cols <- 28

every <- 4
rows <- floor(all.rows / every)
cols <- floor(all.cols / every)

featureEngineering <- function (df) {
    if (every > 1) {
        row.cells <- seq(1, all.rows, by = every)
        col.cells <- seq(1, all.cols, by = every)
        
        columns <- c()
        foreach(rc = row.cells) %do% {
            columns <- c(columns, c(col.cells + (((rc[1]-1) * all.cols))))
        }
        columns <- columns + 1  # Skip "label" column
            
        df <- do.call(rbind, alply(df, .margins = 1, function (row) {
            cell.data <- as.list(row[1, columns])
            
            return(data.frame(label = row$label, cell.data))
        }))
    }
    
    return(df)
}

if (!exists("source.data")) {
    max.rows <- 2000
    gunzip("train.csv.gz", overwrite=TRUE, remove = FALSE)
    source.data <- read.csv("train.csv",
                            header = TRUE,
                            nrows=max.rows)
    source.data$label <- as.factor(source.data$label)
    unlink("train.csv")
}
train.data <- source.data
train.data <- featureEngineering(train.data)

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

figure.columns <- seq(2, (1+rows*cols))

train.data[, figure.columns] <- train.data[, figure.columns] / 255

partition <- createDataPartition(train.data[, "label"], p = 0.8, list = FALSE)
train.batch <- train.data[partition,]
test.batch <- train.data[-partition,]

start.time <- Sys.time()
m <- nnet(formula = label ~ .,
          data = train.batch,
          MaxNWts = 100000,
          maxit = 400,
          size = floor(1.5*rows*cols/rows))
print(Sys.time() - start.time)

test.batch$prediction <- predict(object = m, newdata = test.batch, type="class")
r <- test.batch[, c("label", "prediction")]

num.correct <- NROW(which(test.batch$label == test.batch$prediction))
ratio <- num.correct / NROW(test.batch)

stop()


tryCatch(
    load("bestRatio.RData"),
    error = function(cond) bestRatio <<- 0
)

if (ratio > bestRatio) {
    printf("Best yet: %.3f\n",  ratio)
    
    # Now run our model against the competition data
    if (!exists("competition.data")) {
        gunzip("test.csv.gz", overwrite=TRUE, remove = FALSE)
        competition.data <- read.csv("test.csv", header = TRUE)
        unlink("test.csv")
    }
    
    output <- data.frame(ImageId=1:nrow(competition.data))
    output$Label <- predict(object = m, newdata = competition.data, type = "class")
    
    suffix <- paste(sep="-", format(Sys.time(), "%Y%m%d-%H%M"), round(ratio, 3))
    fname <- paste0("output/submission-", suffix, ".csv")
    
    write.table(output, file=fname, sep = ",", quote = FALSE, row.names = FALSE, col.names = TRUE)
    gzip(fname, overwrite = TRUE)
    
    # Commit to github repository
    git <- "c:\\cygwin64\\bin\\git.exe"
    system(paste(git, "add", "-u"))
    system(paste(git, "commit", "--allow-empty", "-m", paste0("'submission [ratio=", round(ratio, 3), "]'")))
    system(paste(git, "tag", paste0("submission-", round(ratio,3))))
    system(paste(git, "push"))
    
    bestRatio <- ratio
    save(bestRatio, file="bestRatio.RData")
    
    rm(competition.data, output)    # Conserve RAM
    
} else {
    printf("Not an improvement %.3f < %.3f\n", ratio, bestRatio)
}
