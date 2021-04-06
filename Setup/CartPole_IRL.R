#install.packages("plot3D")
library(plot3D)
options(max.print=1000000) # For large console output

# convert to binary for some thresholdx
toBinary = function(x, threshold){
  x[x < threshold] = 0
  x[x >= threshold] = 1
  return(x)
}

# Observations (state)
# 0       Cart Position             -4.8                    4.8
# 1       Cart Velocity             -Inf                    Inf
# 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
# 3       Pole Angular Velocity     -Inf                    Inf
obs <- read.table("~/Desktop/TLOS/School/Alberta/Fall_2020/CMPUT_653/Assignment 2/part2/obsO_0.txt", quote="\"")
names(obs) = c("x", "v", "theta", "omega")

# Neurons 
P1 <- read.table("~/Desktop/TLOS/School/Alberta/Fall_2020/CMPUT_653/Assignment 2/part2/predP1_0.txt", quote="\"")
P2 <- read.table("~/Desktop/TLOS/School/Alberta/Fall_2020/CMPUT_653/Assignment 2/part2/predP2_0.txt", quote="\"")

names(P1) = "P1"
names(P2) = "P2"

data = cbind(obs, P1, P2)



# position and velocity 
scatter3D(x=data$x, data$v, data$P1, phi = 10, bty = "g",  theta = 120, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "pos", ylab = "vel", zlab = "Z")

# angle and angular velocity 
scatter3D(x=data$theta, data$omega, data$P1, phi = 20, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z")
scatter3D(x=data$theta, data$omega, data$P2, phi = 10, bty = "g",  theta = -45, type = "h",  pch = 19, cex = 0.5, ticktype = "detailed", xlab = "theta", ylab = "omega", zlab = "Z")



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


# def theta_omega_policy(obs):
#   theta, w = obs[2:4]
# if abs(theta) < 0.03:
#   return 0 if w < 0 else 1
# else:
#   return 0 if theta < 0 else 1

#if -0.03 < theta & theta < 0.03 
