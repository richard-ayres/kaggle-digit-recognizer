library(plyr)
library(nnet)
library(caret)
library(R.utils)
library(foreach)
library(functional)
library(parallel)
library(doParallel)

set.seed(12345)

num.cores <- detectCores(logical = FALSE)
if (num.cores > 1) {
    cl <- makeCluster(num.cores)
    registerDoParallel(cl)
    
    my.llply <- Curry(llply, .parallel=TRUE, .paropts = list(.packages = c("foreach"), .export = c("cutFigure")))
    my.alply <- Curry(alply, .parallel=TRUE, .paropts = list(.packages = c("foreach"), .export = c("cutFigure")))
} else {
    my.llply <- llply
    my.alply <- alply
}

reshapeData <- function (df) {
    if (!is.null(df$label)) {
        figures = my.alply(df[, 2:NCOL(df)], .margins = 1, function (row) unlist(row[1, ]))
        return(list(labels = df$label, figures = figures))
    } else {
        figures = my.alply(df[, 1:NCOL(df)], .margins = 1, function (row) unlist(row[1, ]))
        return(list(figures = figures))
    }
}

cutFigure <- function (figure, .every = 1) {
    # "figure" is a vector of pixel values
    if (.every <= 1) {
        return(figure)
    }
    
    all.rows <- sqrt(NROW(figure))
    all.cols <- all.rows
    
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

featureEngineering <- function (df, cut.by, do.scale = TRUE) {
    dl <- reshapeData(df)
    
    dl$figures <- my.llply(dl$figures, if(cut.by > 1) Curry(cutFigure, .every = cut.by) else Identity)
    dl$figures <- my.llply(dl$figures, if(do.scale) featureScale else Identity)

    df <- backToDataFrame(dl)
    return(df)
}

if (!exists("source.data")) {
#     num.rows <- 500
    
    load("train.RData")
    if (exists("num.rows") && num.rows > 0) {
        train.data <- train.data[sample(NROW(train.data), num.rows), ]
    }
}

# Takes 10 minutes to massage 10,000 figures into 7x7
print("Massaging data...")
start.time <- Sys.time()
train.data <- featureEngineering(train.data, cut.by = 2)
print(Sys.time() - start.time)

partition <- createDataPartition(train.data[, "label"], p = 0.8, list = FALSE)
train.batch <- train.data[partition,]
test.batch <- train.data[-partition,]

print("Is now a convenient time to save the image?")
stop()  # Convenient time to save image?

print("Training neurons...")
start.time <- Sys.time()
model.nnet <- nnet(formula = label ~ .,
                   data = train.batch,
                   MaxNWts = (NCOL(train.batch) * 125),
                   maxit = 400,
                   size = floor(NCOL(train.batch)/2))
print(Sys.time() - start.time)

test.batch$prediction <- predict(object = model.nnet, newdata = test.batch, type="class")

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
#     if (!exists("test.data")) {
#         load("test.RData")
# 
#         start.time <- Sys.time()
#         test.data <- featureEngineering(test.data, cut.by = 2)
#         print(Sys.time() - start.time)
#     }
    
    test.data$Label <- predict(object = model.nnet, newdata = test.data, type = "class")
    test.data$ImageId <- 1:nrow(test.data)
    
    suffix <- paste(sep="-", format(Sys.time(), "%Y%m%d-%H%M"), round(ratio, 3))
    fname <- paste0("output/submission-", suffix, ".csv")
    
    write.table(test.data[,c("ImageId", "Label")], file=fname, sep = ",", quote = FALSE, row.names = FALSE, col.names = TRUE)
    gzip(fname, overwrite = TRUE)
    
    # Commit to github repository
    git <- "c:\\cygwin64\\bin\\git.exe"
    system(paste(git, "add", "-u"))
    system(paste(git, "commit", "--allow-empty", "-m", paste0("'submission [ratio=", round(ratio, 3), "]'")))
    system(paste(git, "tag", "-f", paste0("submission-", round(ratio,3))))
    
    bestRatio <- ratio
    save(bestRatio, file="bestRatio.RData")
    
} else {
    printf("Not an improvement %.3f < %.3f\n", ratio, bestRatio)
}
