#install.packages("plot3D")
library(plot3D)
options(max.print=1000000) # For large console output

# convert to binary for some thresholdx
toBinary = function(x, threshold){
  x[x < threshold] = 0
  x[x >= threshold] = 1
  return(x)
}

##################
## 1- LOAD DATA ##
#################
# Observations (state)
# 0       Cart Position             -4.8                    4.8
# 1       Cart Velocity             -Inf                    Inf
# 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
# 3       Pole Angular Velocity     -Inf                    Inf
obs <- read.table("~/Documents/GitHub/MastersResearch_orfanos/Setup/obsO_0.txt", quote="\"")
names(obs) = c("x", "v", "theta", "omega")

# Actions
actions <- read.table("~/Documents/GitHub/MastersResearch_orfanos/Setup/actions_0.txt", quote="\"")
names(actions) = "a"

# Neurons
P1 <- read.table("~/Documents/GitHub/MastersResearch_orfanos/Setup/predP1_0.txt", quote="\"")
P2 <- read.table("~/Documents/GitHub/MastersResearch_orfanos/Setup/predP2_0.txt", quote="\"")

names(P1) = "P1"
names(P2) = "P2"

data = cbind(obs, actions, P1, P2)
write.csv(data, file="data.csv")
getwd()



##################
## 2- Plot DATA ##
#################
# position and velocity
scatter3D(x=data$x, data$v, data$P1, phi = 10, bty = "g",  theta = 120, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "pos", ylab = "vel", zlab = "Z")

# angle and angular velocity
scatter3D(x=data$theta, data$omega, data$P1, phi = 20, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z")
scatter3D(x=data$theta, data$omega, data$P2, phi = 10, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z")

plot(data$theta, data$omega, col= ifelse(data$P2 >= 0.5, "red", "blue"), ylim = c(0, 1))


# tree
# Regression Tree Example
#install.packages("rpart")
library(rpart)

# NEURON 1
fit <- rpart(P1~ x + v + theta + omega,
             method="anova", data=data, maxdepth=3)

rpart.rules(fit)


printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# create additional plots
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results

# plot tree
plot(fit, uniform=TRUE,
     main="Regression Tree for Neuron 1")
text(fit, use.n=TRUE, all=TRUE, cex=.8)


mean(data[data$omega < 0.1643993 & data$x < 0.1943893, "a"])
mean(data[data$omega < 0.1643993 & data$x >= 0.1943893, "a"])
mean(data[data$omega >= 0.1643993 & data$x < 0.2381055, "a"])
mean(data[data$omega >= 0.1643993 & data$x >= 0.2381055, "a"])


# NEURON 2
fit <- rpart(P2~ x + v + theta + omega,
             method="anova", data=data, maxdepth=2)

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

# create additional plots
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results

# plot tree
plot(fit, uniform=TRUE,
     main="Regression Tree for Neuron 2")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# candidates for actions
mean(data[data$omega < -0.04620776 & data$theta < -0.008608433, "a"]) # 0.2 --> a=0
mean(data[data$omega < -0.04620776 & data$theta >= -0.008608433, "a"]) # 0.34 --> a=0
mean(data[data$omega >= -0.04620776 & data$theta < -0.0219456, "a"])  # 0.50 --> a= ???
mean(data[data$omega >= -0.04620776 & data$theta >= -0.0219456, "a"]) # 0.72 --> a=1


# create attractive postcript plot of tree
#post(fit, file = "c:/tree2.ps",      title = "Regression Tree for Neuron 2")





# Binary neurons
threshold = 0.5
binaryP1 = toBinary(P1, threshold)
binaryP2 = toBinary(P2, threshold)
data = cbind(obs, binaryP1, binaryP2)

# position and velocity
scatter3D(x=data$x, data$v, data$P1, phi = 10, bty = "g",  theta = 120, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "pos", ylab = "vel", zlab = "Z")


# angle and angular velocity
scatter3D(x=data$theta, data$omega, data$P1, phi = 30, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z", main="N1 Output")
scatter3D(x=data$theta, data$omega, data$P2, phi = 30, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z", main="N1 Output")

plot(data$theta, data$omega, col= ifelse(data$P2 == 1, "red", "blue"), xlab = "theta", ylab = "omega", pch=20)
legend("bottomleft", title = "neuron value \n rounded", legend = c("0","1"), col = c("blue", "red"),  pch=19, cex=0.8)
abline(v=-0.03, )
abline(h=0, v=NULL)


# def theta_omega_policy(obs):
#   theta, w = obs[2:4]
# if abs(theta) < 0.03:
#   return 0 if w < 0 else 1
# else:
#   return 0 if theta < 0 else 1

#if abs(theta) < 0.03 & w < 0
# return 0
#if abs(theta) < 0.03 & w >= 0
# return 1
#if abs(theta) > 0.03 & w < 0
# return 0
#if abs(theta) > 0.03 & theta < 0 (i.e if theta < -0.03)
# return 0


xpoints = 1:20
y = rnorm(20)
plot(NULL,ylim=c(-3,3),xlim=c(1,20))
abline(v=xpoints,col="gray90",lwd=80)
abline(v=xpoints,col="white")
abline(h = 0, lty = 2)
points(xpoints, y, pch = 16, cex = 1.2, col = "red")


mean(data[data$omega < -0.04620776 & data$theta < -0.008608433, "a"]) # 0.2 --> a=0
mean(data[data$omega < -0.04620776 & data$theta >= -0.008608433, "a"]) # 0.34 --> a=0
mean(data[data$omega >= -0.04620776 & data$theta < -0.0219456, "a"])  # 0.50 --> a= ???
mean(data[data$omega >= -0.04620776 & data$theta >= -0.0219456, "a"]) # 0.72 --> a=1

