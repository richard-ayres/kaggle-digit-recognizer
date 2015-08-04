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

# convert training data into list of labels with a corresponding matrix
get.list <- function(df = train.data) {
    rows <- sqrt(NCOL(df) - 1)
    cols <- rows
    
    l <- alply(.data = df, .margins = 1, .fun = function (row) {
        r <- list()
        r$label <- row$label
        r$figure <- matrix(row[1, seq(2, rows*cols+1)], rows, cols)
        
        return(r)
    })
    return(l)
}


reshapeData <- function (df) {
    # Return a list containing a vector of labels
    figures =  alply(df, .margins = 1, function (row) { return(unlist(row[1, seq(2, NCOL(row))])) })
    
    if (!is.null(df$label)) {
        return(list(labels = df$label, figures = figures))
    } else {
        return(list(figures = figures))
    }
}

cutFigure <- function (figure, .every = every) {
    # "figure" is a vector of pixel values
    if (.every <= 1) {
        return(figure)
    }
    
    # cut cells down
    row.cells <- seq(1, all.rows, by = .every)
    col.cells <- seq(1, all.cols, by = .every)
    
    # "keep.columns" is the columns we will keep
    keep.columns <- c()
    foreach(rc = row.cells) %do% {
        keep.columns <- c(keep.columns, c(col.cells + (((rc[1]-1) * all.cols))))
    }
        
    # cells in a block
    block.size <- .every * .every
    my.cells <- c()
    foreach (y=0:(.every-1)) %do% { 
        my.cells <- c(my.cells, (1:.every) + y*all.cols)
    }
    my.cells <- my.cells-1
            
    # Change the value in each interesting cell to the mean of 
    # itself and its neighbours that are to be removed
    foreach(cell = keep.columns) %do% {
        avg <- sum( figure[my.cells + cell] ) / block.size
        figure[cell] <- avg
    }
    
    figure <- figure[keep.columns]
    figure[is.na(figure)] <- 0
    
    return(figure)
}

featureScale <- function (figure) {
    # Take the log(1+x)
    figure <- log1p(figure)
    
    # All values
    location <- 0 #mean(figure)
    range <- sd(figure)
    
    figure <- (figure - location) / range
    
    return(figure)
}

backToDataFrame <- function (l) {
    figures <- as.numeric(unlist(l$figures))
    dim(figures) <- c(NROW(l$figures[[1]]), NROW(l$figures))
    figures <- t(figures)
    
    if (is.null(l$labels)) {
        df <- as.data.frame(figures)
        names(df) <- paste0("pixel", seq(0, NCOL(df)-1))
    } else {
        labels <- as.numeric(l$labels)
        df <- as.data.frame(cbind(labels, figures))
        names(df) <- c("label", paste0("pixel", seq(0, NCOL(df)-2)))
        df$label <- as.factor(df$label)
    }
    
    return(df)
}

featureEngineering <- function (df, do.scale = TRUE) {
    dl <- reshapeData(df)
    dl$figures <- lapply(dl$figures, cutFigure)
    if (do.scale) {
        dl$figures <- lapply(dl$figures, featureScale)
    }
    df <- backToDataFrame(dl)
    return(df)
}

if (!exists("source.data")) {
    max.rows <- 10000
    
    gunzip("train.csv.gz", overwrite=TRUE, remove = FALSE)
    
    colClasses <- c("character", rep("double", all.cols*all.rows))
    source.data <- (read.csv("train.csv", 
                             header = TRUE, 
                             colClasses=colClasses, 
                             col.names = c("label", paste0("pixel", seq(0, (all.rows*all.cols)-1))),
                             nrows=min(max.rows, 500)))
    while(NROW(source.data) < max.rows) {
        new.set <- (read.csv("train.csv", header = FALSE, colClasses=colClasses,
                                       skip=1+NROW(source.data), 
                                       nrows=500,
                                       col.names=names(source.data)))
        source.data <- rbind(source.data, new.set)
    }
    unlink("train.csv")
    rm(new.set)
    
}
train.data <- source.data

# Takes 10 minutes to massage 10,000 figures into 7x7
print("Massaging data...")
start.time <- Sys.time()
train.data <- featureEngineering(train.data)
print(Sys.time() - start.time)

partition <- createDataPartition(train.data[, "label"], p = 0.8, list = FALSE)
train.batch <- train.data[partition,]
test.batch <- train.data[-partition,]

print("Is now a convenient time to save the image?")
stop()  # Convenient time to save image?

print("Training neurons...")
start.time <- Sys.time()
m <- nnet(formula = label ~ .,
          data = train.batch,
          MaxNWts = 100000,
          maxit = 1000,
          size = 100)
#          size = min(50, floor(rows*cols/2) * max(1, NROW(train.batch)/4000))) #floor(1.5*rows*cols/rows))
print(Sys.time() - start.time)

test.batch$prediction <- predict(object = m, newdata = test.batch, type="class")

num.correct <- NROW(which(test.batch$label == test.batch$prediction))
ratio <- num.correct / NROW(test.batch)
printf("Ratio: %.3f\n", ratio)

stop()


tryCatch(
    load("bestRatio.RData"),
    error = function(cond) bestRatio <<- 0
)

if (ratio > bestRatio) {
    printf("Best yet: %.3f\n",  ratio)
    
    # Now run our model against the competition data
    if (!exists("test.data")) {
        gunzip("test.csv.gz", overwrite=TRUE, remove = FALSE)
        test.data <- read.csv("test.csv", header = TRUE)
        unlink("test.csv")
        
        start.time <- Sys.time()
        print(start.time)
        test.data <- featureEngineering(test.data)
        print(Sys.time() - start.time)
    }
    
    test.data$Label <- predict(object = m, newdata = test.data, type = "class")
    test.data$ImageId <- 1:nrow(test.data)
    
    suffix <- paste(sep="-", format(Sys.time(), "%Y%m%d-%H%M"), round(ratio, 3))
    fname <- paste0("output/submission-", suffix, ".csv")
    
    write.table(test.data[,c("ImageId", "Label")], file=fname, sep = ",", quote = FALSE, row.names = FALSE, col.names = TRUE)
    gzip(fname, overwrite = TRUE)
    
    # Commit to github repository
    git <- "c:\\cygwin64\\bin\\git.exe"
    system(paste(git, "add", "-u"))
    system(paste(git, "commit", "--allow-empty", "-m", paste0("'submission [ratio=", round(ratio, 3), "]'")))
    system(paste(git, "tag", paste0("submission-", round(ratio,3))))
#     system(paste(git, "push"))
    
    bestRatio <- ratio
    save(bestRatio, file="bestRatio.RData")
    
    rm(competition.data, output)    # Conserve RAM
    
} else {
    printf("Not an improvement %.3f < %.3f\n", ratio, bestRatio)
}
