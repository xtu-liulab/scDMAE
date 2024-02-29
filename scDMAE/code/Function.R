rm(list = ls())
library(data.table)
#-----------------------------------------------------------Clustering-------------------------------------------------------------------
input_file <- "D:/pyz/data/Clustering/pollen/"
namedata<-"pollen"
re_exp <- read.table(paste0(input_file,namedata,"_scDMAE_f.csv"),header = TRUE,sep=",",row.names = 1)
exp <- read.table(paste0(input_file,namedata,"_HVG.csv"),header = TRUE,sep=",",row.names = 1)
colnames(re_exp) <- colnames(exp)
rownames(re_exp) <- rownames(exp)

re_exp[re_exp < 0] <- 0
i <- 0.95
my_exp <- i*(10**(re_exp)-1)+(1-i)*exp
write.csv(my_exp,paste0(input_file,namedata,"_scDMAE.csv"),row.names = T)