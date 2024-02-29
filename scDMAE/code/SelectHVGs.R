sc_data <- read.csv("D:/pyz/data/Clustering/Pollen_RAW.csv", row.names = 1)
rownames(sc_data) <- gsub("_", "-", rownames(sc_data))
################  select 2,000 HVGs before imputation  ################
library(Seurat)
x.seurat <- CreateSeuratObject(sc_data)
x.seurat <- NormalizeData(x.seurat)
x.seurat <- ScaleData(x.seurat)
x.seurat <- FindVariableFeatures(x.seurat, verbose = FALSE)


y <-x.seurat@assays$RNA@var.features
z <- as.matrix(sc_data)[y ,]
write.csv(z,"D:/pyz/data/Clustering/Pollen_HVG.csv",row.names = T)
