# Akmal Aulia

# Doing parallel computing with doSNOW package is quite straightforward, insyaa Allah.  Notice that in the function definition of error_rf(), 
# library(randomForest) gets called everytime the function is triggered - this is necessary when using foreach on the clusters.
# A nice intro to parallel computing can be found in https://computing.llnl.gov/tutorials/parallel_comp/#Abstract.

# call library

library(randomForest)

library(doSNOW)

library(doParallel) # to use detectCores()



# set number of trials

trials <- 4000

d.size <- 1000

cores <- detectCores()



# create dummy dataset

d <- matrix(sample(10*d.size)/(10*d.size), d.size, 10)

colnames(d) <- c("x1", "x2", "x3", "x4", "x5",

                 "x6", "x7", "x8", "x9", "y")

d <<- as.data.frame(d) # make it global



# create function to evaluate

error_rf <- function()

{

  

  library(randomForest)

  

  # create training and test set

  train_fraction <- 0.66

  ind <- sample(1:nrow(d), train_fraction*nrow(d))

  d_train <- d[ind,]

  d_test  <- d[-ind,]

  

  # train Random Forests model

  rf.out <- randomForest(y ~., data = d_train, ntree=127)

  

  # predict test data

  rf.pred <- predict(rf.out, d[-ind,])

  

  # measure prediction performance

  rf.mse <- sum((rf.pred - d_test$y)^2)/nrow(as.matrix(d_test$y))

  

  # return

  return(rf.mse)

}


# execute Random Forest 5 times (SERIALLY)

errSum <- 0


# record time

ti <- Sys.time()


for (i in 1:trials)

{

  errSum <- errSum + error_rf()

}

errSum/trials


# record time elapsed

tf <- Sys.time()

tf - ti

ti <- tf


# execute Random Forest 5 times (PARALLELLY)

cl <- makeCluster(cores-1) # to not overload PC, leave 1 out

registerDoSNOW(cl)


system.time({r <- foreach(icount(trials), combine=rbind) %dopar% error_rf()})

mean(unlist(r))


stopCluster(cl)


# record time elapsed

tf <- Sys.time()

tf - ti

Here is another example of parallel computing.  Instead of using icount(trials) (see argument in foreach()), we can use a normal indices, like i, or j.  We don't need to pass these indices into the function, since R treats these indices as global values. 

cl <- makeCluster(cores-1) # not to overload PC

registerDoSNOW(cl)


system.time({

  r <- foreach(z=1:n_combine, combine=rbind) %dopar%

    run_rf()

})


stopCluster(cl)

# -------------------------------
# The full code looks like this below.  Note that library(randomForest) needs to be inside the function rf_run() 
# since the library needs to be re-uploaded in each core.

# --------------------- prelims ----------------------

# rm(list = ls()) # remove data

set.seed(123) # setting seed

start.time <- Sys.time() # record time lapse

# -----------------------------------------------------




# ------------------- library calls -----------------------

library(randomForest)

library(doSNOW)

# ---------------------------------------------------------


# record time

ti <- Sys.time()


# ----------------------- set constants --------------------

VARIATIONS <- 10

MAX_NTREE <- 500

MAX_MTREE <- 13

MAX_N_SPLIT <- 0.95

# ---------------------------------------------------------


# --------------------------------------------------------

# set number of trials

trials <- 5

d.size <- 1000

cores <- detectCores()

# --------------------------------------------------------


# ------------------- read data ---------------------------

d <<- read.csv(file="DataFile.csv", header=TRUE) # global variable

# ---------------------------------------------------------


# ------------------ define parameters to vary--------------------

val_ntree <- round(seq(3,MAX_NTREE, length.out=VARIATIONS),0)

val_mtry <- round(seq(2,MAX_MTREE, length.out=VARIATIONS),0) # default is #var/3

val_split <- round(seq(0.3, MAX_N_SPLIT, length.out=VARIATIONS),2)

ijk_tab <<- expand.grid(val_split, val_ntree, val_mtry)

n_combine <<- VARIATIONS^3 # make global

# ---------------------------------------------------------


# -------------------- create function to run ----------------------

run_rf <- function()

{

  

  # call library

  library(randomForest)

  

  val_split <- ijk_tab[z,1]

  val_ntree <- ijk_tab[z,2]

  val_mtry <- ijk_tab[z,3]

  

  train_fraction <- val_split

  ind <- sample(1:nrow(d), train_fraction*nrow(d))

  d_train <- d[ind,]

  d_test  <- d[-ind,]

  

  # train Random Forests model

  rf.out <- randomForest(Output ~., data = d_train, ntree=val_ntree, mtry=val_mtry)

  

  # predict test data

  rf.pred <- predict(rf.out, d[-ind,])

  

  # measure prediction performance

  rf.mse <- sum((rf.pred - d_test$Output)^2)/nrow(as.matrix(d_test$Output))

  rf.cor <- cor(rf.pred, d_test$Output)

  

  # measure training performance

  rf.pred_train <- predict(rf.out, d[ind,])

  rf.mse_train <- sum((rf.pred_train - d_train$Output)^2)/nrow(as.matrix(d_train$Output))

  rf.cor_train <- cor(rf.pred_train, d_train$Output)

  

  # importance ranking

  x <- importance(rf.out)

  x_sorted <- cbind(x, seq(1,nrow(x), by=1)) # create index vector

  x_sorted <- x_sorted[order(x_sorted[,1], decreasing=TRUE),] # sort impact

  x_sorted <- cbind(x_sorted, seq(1,nrow(x), by=1)) # assign rank based on impact

  colnames(x_sorted) <- c("impact", "index", "rank")

  x_sorted <- x_sorted[order(x_sorted[,2]),] # sort based on index

  

  # store results

  r_stat <- c(val_split, val_ntree, val_mtry, 

              rf.mse_train, (rf.cor_train)^2, rf.mse, (rf.cor)^2)

  

  # store impact values

  r_impact <- x[1:13]

  

  # store ranking

  r_ranks <- as.numeric(x_sorted[1:13,3])

  

  # return

  return(c(r_stat, r_impact, r_ranks))

}

# ------------------------------------------------------------




# ------------------------- run combinations on clusters---------------

cl <- makeCluster(cores-1) # not to overload PC

registerDoSNOW(cl)


system.time({

  r <- foreach(z=1:n_combine, combine=rbind) %dopar%

    run_rf()

})


stopCluster(cl)

# ---------------------------------------------------------------




# ---------------- compile results matrix -----------------------

results_matrix <- matrix(unlist(r), VARIATIONS^3, 33, byrow=TRUE)

nrow(results_matrix)

ncol(results_matrix)


# give column names for results

colnames(results_matrix) <- c("split", "ntree", "mtry", "mse_train", "R2_train",                    # 4 items

                              "mse_test", "R2_test",                                                # 2

                              "P1_impact", "P2_impact", "P3_impact", "P4_impact", "P5_impact",      # 5

                              "P6_impact", "P7_impact", "P8_impact", "P9_impact", "P10_impact",     # 5

                              "P11_impact", "P12_impact", "P13_impact",                             # 4

                              "P1_rank", "P2_rank", "P3_rank", "P4_rank", "P5_rank",                # 5

                              "P6_rank", "P7_rank", "P8_rank", "P9_rank", "P10_rank",               # 5

                              "P11_rank", "P12_rank", "P13_rank")



# quick check on results

head(results_matrix)

# --------------------------------------------------------------



# show time lapse

Sys.time() - ti



# write results to file

write.csv(file="results_matrix.csv", results_matrix)

# Note that the code above executed the function run_rf() 1000 times, and took only about 10 minutes.  
# One execution of run_rf() roughly takes about approximately 2 to 4 seconds.
#
# Reference:
#
#    https://stackoverflow.com/questions/20239707/parallel-for-loop-r 
#
#    https://nceas.github.io/oss-lessons/parallel-computing-in-r/parallel-computing-in-r.html 
#
#    https://stackoverflow.com/questions/38318139/run-a-for-loop-in-parallel-in-r
