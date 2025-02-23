p <- 30


func.path <- "/Users/caoguanq/Library/CloudStorage/OneDrive-MichiganStateUniversity/Functional graphical model paper/R code simulation/modelA"
save.path <- "~/Downloads/"
# Packages
library(fda)
library(matrixcalc)
library(MASS)
library(mvtnorm)

# Global Parameter Settings
mu <- 5 # number of basis used to generate data
M <- 7 # number of basis used for neighborhood selection
n <- 100
tau <- 100 # number of observations
thres.ctrl <- 0 # recognition threshold epsilon_n = thres.ctrl * lambda_n
tol.abs <- 1e-4  # Tolerance (absolute) in ADMM
tol.rel <- 1e-4  # Tolerance (relative) in ADMM
L <- 100 # number of lambdas

# Read R files
source(paste(func.path,"ADMM.new.R", sep="/"))      # ADMM Function
source(paste(func.path,"prec.rec.R", sep="/"))      # Precision & Recall
source(paste(func.path,"auc.R", sep="/"))           # Computing AUC from ROC
source(paste(func.path,"FPCA.score.R", sep="/"))    # Computing FPCA scores
source(paste(func.path,"bases.func.R", sep="/"))    # Basis functions for FPCA
source(paste(func.path,"A.prelim.func.R", sep="/")) # For Model A generation
source(paste(func.path,"ProxAlg_FGM.R", sep="/"))   # FGLasso

#setwd("C:/Users/s0wang33/Desktop/Functional_graphical")
for(seed in 1:50){
  set.seed(seed)
  Theta <- cov.mat.model.A(p, mu) # p*mu by p*mu large square matrix
  G.true <- matrix(0, p, p) # p by p adjacency matrix
  for(i in 1:p){
    for(j in 1:p){
      if(sum(abs(Theta[((i-1)*mu+1):(i*mu), ((j-1)*mu+1):(j*mu)])) > 0)
        G.true[i,j] <- 1
    }
  }
  
  # 1. Generating delta
  delta <- rmvnorm(n, sigma = solve(Theta))
  
  
  # 2. Observation time
  obs.time <- seq(1/tau, 1, 1/tau) # vector of observation time points of delta
  
  # 3. Fourier basis function for data generation
  b.mat.list <- list()
  for(j in 1:p){
    b.mat.list[[j]] <- fda.fourier.mat(obs.time, mu)
  }
  
  # 4. Observations h_ijk
  h <- array(0, c(n, p, tau))
  for(i in 1:n){
    for(j in 1:p){
      h[i,j,] <- b.mat.list[[j]] %*% 
        matrix(delta[i, ((j-1)*mu+1) : (j*mu)], ncol=1) + rnorm(tau, 0, 0.5)
    }
  }
  
  # Reserved part for PSKL Method
  y.list <- list()
  for(j in 1:p){
    y.list[[j]] <- h[,j,]
  }
  allscore=c()
  for(j in 1:p){
    obs.val.matrix <- matrix(0, nrow=tau, ncol=n)
    for (i in c(1:n)){
      obs.val.vec <- as.vector(h[i, j, ])
      obs.val.matrix[, i] <- obs.val.vec
    }
    bspline.basis <- create.bspline.basis(rangeval=c(0, 1), nbasis=M)
    # Construct a functional data object from the observation
    # bspline basis is the default setting of this function
    # It does not mean that the basis function is bspline!
    fd.object.array <- Data2fd(argvals=obs.time, y=obs.val.matrix, basisobj=bspline.basis)
    # FPCA process
    s=pca.fd(fd.object.array, nharm=M)$scores
    allscore=cbind(allscore,s)
  }
  name=paste0(save.path,"X",seed,".csv")
  write.csv(allscore,file=name)
}

