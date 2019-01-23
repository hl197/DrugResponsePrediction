Sys.time()
library(data.table)
library(foreach)

removeNArows <- function (A) {
  return(A[rowSums(is.na(A))==0, ])
}

removeConstantCols <- function (A) {
  # remove columns with 0 variance
  return(A[, apply(A, 2, var) != 0])
}

###############
## TCGA expression processed by Kallisto (data from Benjamin H.K. group)
###############
# load gene and transcript annotation details: toil.genes, toil.transcripts
load("Gencode.v23.annotation.RData") 
tcga.genes <- rownames(toil.genes)

# load gene and transcript annotation details: toil.genes, toil.transcripts
tcga.samples <- read.table("tcga_sample_names", sep='\t', header=FALSE, row.names = 1)
tcga.samples <- sapply(tcga.samples, as.character)
TCGA.rnaseq <- as.data.frame(fread("tcga_Kallisto_tpm_genes", sep=',', header=FALSE))
colnames(TCGA.rnaseq) <- tcga.samples
rownames(TCGA.rnaseq) <- tcga.genes

TCGA.rnaseq <- t(TCGA.rnaseq)
TCGA.rnaseq <- removeNArows(TCGA.rnaseq)
TCGA.rnaseq <- removeConstantCols(TCGA.rnaseq)
dim(TCGA.rnaseq)

## filter TCGA samples for primary tumors
# 01 (Primary Solid Tumor)
# 03 (Primary Blood Derived Cancer - Peripheral Blood)
# 09 (Primary Blood Derived Cance - Bone Marrow)
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
tcga.sample.type <- sapply(rownames(TCGA.rnaseq), function(x) unlist(strsplit(x, '-'))[4])
select <- tcga.sample.type %in% c('01', '03', '09')
TCGA.rnaseq <- TCGA.rnaseq[select, ]
## rename and pick one of the duplicates (e.g. TCGA-BL-A13J-01 has 3 entries)
tcga.newnames <- sapply(rownames(TCGA.rnaseq), function(x) paste(unlist(strsplit(x, '-'))[1:3], collapse='-'))
# table(tcga.newnames)[table(tcga.newnames)>1]
# cor(TCGA.rnaseq[rownames(TCGA.rnaseq)=='TCGA-BL-A13J-01',])
rownames(TCGA.rnaseq) <- tcga.newnames
TCGA.rnaseq <- TCGA.rnaseq[unique(tcga.newnames),]
write.table(colnames(TCGA.rnaseq), file="gene_ontology/unprocessed_genes.txt", row.names=FALSE, col.names=FALSE, quote=FALSE)

data <- readRDS("NeoALTTO_Exp_Mappping_Meta.rds")
countMat <- data$CountMat
write.csv(row.names(countMat), "all_ENST.csv", quote=FALSE, row.names = FALSE)
