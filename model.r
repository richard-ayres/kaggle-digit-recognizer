#library(plyr) - plyr seems to have problems with parallelism
library(nnet)
library(caret)
library(R.utils)
library(foreach)
library(functional)
library(parallel)
library(doParallel)

conf <- list(
    seed = 12345,
    num.cores = detectCores(logical = FALSE),
    num.rows = 20000,
    cut.pixels = 2,
    do.normalise = TRUE,
    take.log = TRUE,
    model = "nnet"
)

set.seed(conf$seed)

if (conf$num.cores > 1) {
    cl <- makeCluster(conf$num.cores)
    registerDoParallel(cl)
}

reshapeData <- function (df) {
    if (!is.null(df$label)) {
        labels = df$label
        
        df <- t(df[, -1])
        figures <- lapply(seq_len(ncol(df)), function (i) df[,i])
        
        return(list(labels = labels, figures = figures))
    } else {
        df <- t(df)
        figures <- lapply(seq_len(ncol(df)), function (i) df[,i])
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
    for(rc in row.cells) {
        keep.columns <- c(keep.columns, c(col.cells + (((rc[1]-1) * all.cols))))
    }
        
    # cells in a block
    block.size <- .every * .every
    my.cells <- c()
    for (y in 0:(.every-1)) { 
        my.cells <- c(my.cells, (1:.every) + y*all.cols)
    }
    my.cells <- my.cells-1
            
    # Change the value in each interesting cell to the mean of 
    # itself and its neighbours that are to be removed
    for(cell in keep.columns) {
        avg <- sum( figure[my.cells + cell] ) / block.size
        figure[cell] <- avg
    }
    
    figure <- figure[keep.columns]
    figure[is.na(figure)] <- 0
    
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

featureEngineering <- function (df, cut.by = conf$cut.pixels, take.log = conf$take.log, do.normalise = conf$do.normalise, do.parallel = FALSE) {
    dl <- reshapeData(df)
    
    process <- Identity
    
    if (cut.by > 1) {
        process <- Compose(process, Curry(cutFigure, .every = cut.by))
    }
    if (take.log) {
        process <- Compose(process, log1p)
    }
    if (do.normalise) {
        process <- Compose(process, function (vec) vec/max(vec))
    }
    
    if (do.parallel && exists("cl")) {
        dl$figures <- foreach(i = seq_len(length(dl$figures)), .export=c("cutFigure")) %dopar% { process( dl$figures[[i]] ) }
    } else {
        dl$figures <- lapply(dl$figures, process)
    }

    df <- backToDataFrame(dl)
    return(df)
}

load("train.RData")

# Cut data down to a "reasonable size"
cut.partition <- createDataPartition(train.data$label, p = conf$num.rows/NROW(train.data), list = FALSE)
train.data <- train.data[cut.partition, ]
rm(cut.partition)

# Takes 4.5 seconds to massage 10,000 figures into 7x7 (used to be 10 minutes!)
print("Massaging data...")
start.time <- Sys.time()
train.data <- featureEngineering(train.data)
print(Sys.time() - start.time)

partition <- createDataPartition(train.data$label, p = 0.8, list = FALSE)
train.batch <- train.data[partition,]
test.batch <- train.data[-partition,]
rm(partition)

# print("Is now a convenient time to save the image?")
# stop()  # Convenient time to save image?

if (conf$model == 'nnet') {
    my.train <- (function () {
        num.pixels <- NCOL(train.batch)
        hidden.size = floor(min(200, (10 + num.pixels)/2))
        max.weights <- floor(1.2 * num.pixels * hidden.size)
        
        return( function () nnet(
            formula = label ~ .,
            data = train.batch,
            maxit = 1000, #400,
            size = hidden.size,
            MaxNWts = max.weights,
            decay = 0.01 / hidden.size,
            reltol = 0.000001
        ))
    })()
    my.predict <- Curry(predict, type="class")
    
} else if (conf$model == 'train.nnet') {
    my.train <- function () train(form = label ~ .,
                                  data = train.batch,
                                  method = "nnet",
                                  trControl = trainControl(method="repeatedcv", repeats = 10),
                                  tuneGrid = expand.grid(size = floor(seq(NCOL(train.batch)/5, NCOL(train.batch)/2, length.out=4)),
                                                         decay = seq(0, 0.5, length.out = 5)))
    my.predict <- predict
    
} else if (conf$model == 'train.rf') {
    my.train <- function () train(form = label ~ .,
                                  data = train.batch,
                                  method = "rf",
                                  trControl = trainControl(method="repeatedcv", repeats = 10),
                                  tuneGrid = data.frame(.mtry = NCOL(train.batch)-1))
    my.predict <- predict
}

print("Training....")
start.time <- Sys.time()
model <- my.train()
print(Sys.time() - start.time)

test.batch$prediction <- my.predict(object = model, newdata = test.batch)

num.correct <- NROW(which(test.batch$label == test.batch$prediction))
ratio <- num.correct / NROW(test.batch)
printf("Ratio: %.3f\n", ratio)

# stop()


tryCatch(
    load("bestRatio.RData"),
    error = function(cond) bestRatio <<- 0
)

if (ratio > bestRatio) {
    printf("Best yet: %.3f\n",  ratio)
    
    # Now run our model against the competition data
    if (!exists("test.data")) {
        load("test.RData")

        start.time <- Sys.time()
        test.data <- featureEngineering(test.data)
        print(Sys.time() - start.time)
    }
    
    test.data$Label <- my.predict(object = model, newdata = test.data)
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

if (exists("cl")) {
    stopCluster(cl)
}