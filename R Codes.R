options(warn=-1)
library(diffpriv)
f <- function(X) quantile(X)
## target function
n <- 100
## dataset size
mechanism <- DPMechLaplace(target = f, sensitivity = 1/n, dims = 1)
D <- runif(n, min = 0, max = 1)
## the sensitive database in [0,1]^n
pparams <- DPParamsEps(epsilon = 1)
## desired privacy budget
r <- releaseResponse(mechanism, privacyParams = pparams, X = D)
print("Private response r$response:")
r$response
print("nnNon-private response f(D): ")
f(D)

#ADULT DATASET
traind=read.csv("D:\\Differential Privacy\\adult.csv",header=T,na=c(" ?","NA"))
testd=read.csv("D:\\Differential Privacy\\adulttest.csv",header=T,na=c(" ?","NA"))
testd$income=ifelse(testd$income==" <=50K",0,1)
traind=na.omit(traind)
testd=na.omit(testd)
traind$income=ifelse(traind$income==" <=50K",0,1)
traind$income=factor(traind$income)
library("methods")
require(glmnet,quiet=T)
lam=seq(0,25,length=100)

#-------------------------------Ridge-----------------------------------------

m=glmnet(data.matrix(traind[,-15]),as.matrix(traind$income),intercept=T,standardize=T,alpha=0,lambda=lam,family="binomial")
summary(m)

#CV Ridge
cv.rid=cv.glmnet(x=data.matrix(traind[,-15]),y=as.numeric(traind$income),nfold=15,alpha=0,standardize=T,family="binomial")
rid.lambda=cv.rid$lambda.1se
#for best lambda 
m=glmnet(data.matrix(traind[,-15]),as.matrix(traind$income),intercept=T,standardize=T,alpha=0,lambda=rid.lambda,family="binomial")
beta=coef(m)
pred=predict(m,newx=data.matrix(testd[,-15]))
nrow(testd)
length(pred)
max(pred)
min(pred)
pred[pred>.5]=1
pred[pred<=.5]=0 
ss=1-sum(pred[]!=testd$income)/nrow(testd)

library(nimble) 
#output perturbation
op=function(ep,l)
{
s=NULL
nf=sqrt(sum(beta^2))/2
lf=1/(1+exp(testd$income))
for(i in 1:length(ep))
{ 
b=rdexp(nrow(testd),0,nrow(testd)*ep[i]*l/2)
fpriv=pred+b 
fpriv[fpriv>1]=1
fpriv[fpriv<0]=0
s[i]=sum(fpriv[]!=testd$income)/nrow(testd)
} 
plot(ep,s,ylim=c(0,1),xlab="Privacy Parameter",ylab="Error in Prediction",main="Logistic Regression with Ridge Penalty under Output Perturbation",lty=2)
}
op(seq(0,.5,length=30),rid.lambda)
abline(h=ss) 
legend("topright",legend=c("Output Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")

#objective perturbation
n=ncol(testd)
objp=function(ep,l,c=0.25)
{
s=NULL
delta=NULL
t=1+2*c/(n*l)+c^2/((n^2)*(l^2))
ephat=ep-log(t)
for(i in 1:length(ep)) 
{
if(ephat[i]>0)
{ 
delta=0
}
else
{ 
delta=c/((n*(exp(ep/4)-1))-l)
}
fpriv=pred+delta*sqrt(sum(beta^2))/2 
fpriv[fpriv>.5]=1 
fpriv[fpriv<=.5]=0
s[i]=sum(fpriv[]!=testd$income)/nrow(testd)
}
plot(ep,s,ylim=c(0,1),xlab="Privacy Parameter",ylab="Error in Prediction",main="Logistic Regression with Ridge Penalty under Objective Perturbation",lty=2)
} 
objp(seq(0,0.5,length=30),rid.lambda)
abline(h=ss)
legend("topright",legend=c("Objective Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")

#---------------------------------------LASSO-------------------------------

m=glmnet(data.matrix(traind[,-15]),as.matrix(traind$income),intercept=T,standardize=T,alpha=1,lambda=lam,family="binomial")
summary(m)

#CV LASSO
cv.las=cv.glmnet(x=data.matrix(traind[,-15]),y=as.numeric(traind$income),nfold=15,alpha=1,standardize=T,family="binomial") 
las.lambda=cv.las$lambda.1se

#for best lambda
m=glmnet(data.matrix(traind[,-15]),as.matrix(traind$income),intercept=T,standardize=T,alpha=1,lambda=las.lambda,family="binomial")
beta=coef(m)
predl=predict(m,newx=data.matrix(testd[,-15]))
nrow(testd)
length(predl)
max(predl)
min(predl)
pred[predl>.5]=1 
pred[predl<=.5]=0
ssl=1-sum(predl[]!=testd$income)/nrow(testd)
library(nimble)

#output perturbation
opl=function(ep,l)
{
s=NULL
nf=sqrt(sum(beta^2))/2
lf=1/(1+exp(testd$income))
for(i in 1:length(ep))
{
b=rdexp(nrow(testd),0,nrow(testd)*ep[i]*l/2)
fpriv=predl+b
fpriv[fpriv>1]=1
fpriv[fpriv<0]=0
s[i]=sum(fpriv[]!=testd$income)/nrow(testd)
}
plot(ep,s,ylim=c(0,1),xlab="Privacy Parameter",ylab="Error in Prediction",main="Logistic Regression with LASSO Penalty under Output Perturbation",lty=2)
}
opl(seq(0,.5,length=30),rid.lambda)
abline(h=ss)
legend("topright",legend=c("Output Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")

#objective perturbation
n=ncol(testd)
objpl=function(ep,l,c=0.25) 
{
s=NULL
delta=NULL
t=1+2*c/(n*l)+c^2/((n^2)*(l^2))
ephat=ep-log(t)
for(i in 1:length(ep))
{ 
if(ephat[i]>0)
{ 
delta=0 
}
else
{ 
delta=c/(n*(exp(ep/4)-1))-l
} 
fpriv=predl+delta*sqrt(sum(beta^2))/2
fpriv[fpriv>.5]=1 
fpriv[fpriv<=.5]=0
s[i]=1-sum(fpriv[]!=testd$income)/nrow(testd)
} 
plot(ep,s,ylim=c(0,1),xlab="Privacy Parameter",ylab="Error in Prediction",main="Logistic Regression with LASSO Penalty under Objective Perturbation",lty=2)
}
objpl(seq(0,.5,length=30),rid.lambda)
abline(h=ss)
legend("topright",legend=c("Objective Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")



#Bike Sharing Rental Dataset

d=read.csv("D:\\Differential Privacy\\hour.csv",header=T)
d=na.omit(d)
index=sample(nrow(d),nrow(d)*0.75)
traind=data.matrix(d[index,])
testd=data.matrix(d[-index,])
View(traind)

#-----------------------------Ridge------------------------------ 
library("methods")
require(glmnet,quiet=T)
lam=seq(0,25,length=100)
m=glmnet(traind[,-17],traind[,17],intercept=T,standardize=T,alpha=0,lambda=lam)
summary(m)

#CV Ridge
cv.rid=cv.glmnet(x=traind[,-17],y=traind[,17],nfold=15,alpha=0,standardize=T)
rid.lambda=cv.rid$lambda.1se

#best lambda
m2=glmnet(traind[,-17],traind[,17],intercept=T,standardize=T,alpha=0,lambda=rid.lambda)
summary(m2) 
pred=predict(m2,newx=testd[,-17]) 
nrow(testd)
nrow(pred)
err=(testd[,17]-pred)^2
mean(err) 
summary(pred)
library(nimble)
#output perturbation
op=function(ep,l)
{ 
t=NULL
s=NULL
for(i in 1:length(ep))
{
b=rdexp(nrow(testd),0,nrow(testd)*ep[i]*l/2)
fpriv=pred+b
s[i]=mean((fpriv-testd[,17])^2)
} 
for(i in 0:(length(ep)-1))
{ 
t[i+1]=s[30-i]
}
plot(ep,t,xlab="Privacy Parameter",ylab="Error",main="HDR with Ridge Penalty under Output Perturbation",lty=2)
}
op(seq(0,.5,length=30),rid.lambda) 
abline(h=err) 
legend("topright",legend=c("Output Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")

#objective perturbation 
beta=coef(m2) 
n=ncol(testd)
objp=function(ep,l,c=0.25)
{
s=NULL
delta=NULL
t=1+2*c/(n*l)+c^2/((n^2)*(l^2))
ephat=ep-log(t)
for(i in 1:length(ep))
{
if(ephat[i]>0)
{
delta=0 
}
else
{
delta=c/(n*(exp(ep/4)-1))-l
ephat[i]=ep[i]/2
} 
fpriv=pred+delta*sqrt(sum(beta^2))/2
s[i]=mean((fpriv-testd[,17])^2) 
} 
plot(ep,s,xlab="Privacy Parameter",ylab="Error",main="HDR with Ridge Penalty under Objective Perturbation",lty=2)
}
objp(seq(0,.5,length=30),rid.lambda) 
abline(h=mean(err-2))
legend("topright",legend=c("Objective Perturbed Error","Non Private Error"),lty=c(2,1),bty="n")

#----------------------------------------LASSO------------------------------
library("methods") 
require(glmnet,quiet=T)
lam=seq(0,25,length=100)
ml=glmnet(traind[,-17],traind[,17],intercept=T,standardize=T,alpha=1,lambda=lam)
summary(ml)

#CV LASSO 
cv.las=cv.glmnet(x=traind[,-17],y=traind[,17],nfold=15,alpha=1,standardize=T)
las.lambda=cv.las$lambda.1se

#best lambda
ml2=glmnet(traind[,-17],traind[,17],intercept=T,standardize=T,alpha=1,lambda=las.lambda) 
summary(ml2) 
predl=predict(ml2,newx=testd[,-17])
nrow(testd) nrow(predl)
errl=(testd[,17]-pred)^2 mean(errl) 
summary(predl)
library(nimble) 
#output perturbation
opl=function(ep,l)
{
s=NULL
t=NULL
for(i in 1:30)
{
b=rdexp(nrow(testd),0,nrow(testd)*ep[i]*l/2)
fpriv=predl+b 
s[i]=mean((fpriv-testd[,17])^2)
}
for(i in 0:29)
{
t[i+1]=s[30-i]
}
plot(ep,t,xlab="Privacy Parameter",ylab="Error",main="HDR with LASSO Penalty under Output Perturbation",lty=2)
} 
opl(seq(0,.5,length=30),las.lambda)
abline(h=err)
legend("topright",legend=c("Objective Error","Non Private Error"),lty=c(2,1),bty="n")

#objective perturbation
beta=coef(ml2)
n=ncol(testd)
objpl=function(ep,l,c=0.5) 
{
s=NULL delta=NULL t=1+2*c/(n*l)+c^2/((n^2)*(l^2))
ephat=ep-log(t)
for(i in 1:30)
{
if(ephat[i]>0) 
{ 
delta=0
} 
else
{
delta=c/(n*(exp(ep/4)-1))-l
ephat[i]=ep[i]/2 
} 
fpriv=predl+delta*sqrt(sum(beta^2))/2
s[i]=mean((fpriv-testd[,17])^2)
} 
plot(ep,s,xlab="Privacy Parameter",ylab="Error",main="HDR with LASSO Penalty under Objective Perturbation",lty=2)
} 
objpl(seq(0,.5,length=30),las.lambda) 