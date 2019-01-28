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
tcga.genes <- toil.genes$Symbol

# load gene and transcript annotation details: toil.genes, toil.transcripts
tcga.samples <- read.table("tcga_sample_names", sep='\t', header=FALSE, row.names = 1)
tcga.samples <- sapply(tcga.samples, as.character)
TCGA.rnaseq <- as.data.frame(fread("tcga_Kallisto_tpm_genes", sep=',', header=FALSE))
colnames(TCGA.rnaseq) <- tcga.samples
rownames(TCGA.rnaseq) <- make.names(tcga.genes, unique=TRUE)
write.table(rownames(TCGA.rnaseq), file="gene_ontology/genes.txt", row.names=FALSE, col.names=FALSE, quote=FALSE)


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

### TCGA clinical meta data + surivival data (but no drug treatment info)
## from Liu et al. (2018) (published in Cell)
tcga.metadata <- read.csv("TCGA-survival-outcomes-Liu2018.csv", sep=",", header=TRUE, row.names=1)
# all(rownames(TCGA.rnaseq) %in% rownames(tcga.metadata))

TCGA.tissueid <- as.character(tcga.metadata[rownames(TCGA.rnaseq), 'type'])
names(TCGA.tissueid) <- rownames(TCGA.rnaseq)

tcga.response <- read.csv("drug_response.txt", sep="\t", header=TRUE)

inselect <- tcga.response$patient.arr %in% rownames(TCGA.rnaseq)
tcga.response <- tcga.response[inselect, ]
print("all drug response")
dim(tcga.response)

# double check cancer type annotation of labeled TCGA patients
a <- unique(tcga.response[, c('cancers', 'patient.arr')])
stopifnot(all(TCGA.tissueid[as.character(a$patient.arr)] == a$cancers))

# subset to drugs with >50 samples
tcga.responsecounts <- table(tcga.response$drug.name)
tcga.drugselect <- names(tcga.responsecounts)[tcga.responsecounts > 50]
tcga.responsecounts[tcga.drugselect]
tcga.response <- tcga.response[tcga.response$drug.name %in% tcga.drugselect, ]
print(">50 samples")
dim(tcga.response)

## get list of all combination therapies 
allcomb <- foreach(pid = unique(as.character(tcga.response$patient.arr))) %do% {
  pdata <- tcga.response[tcga.response$patient.arr == pid,]
  # print(pid)
  comb <- foreach (st = unique(pdata$start.time)) %do% {
    paste(sort(pdata[pdata$start.time == st, "drug.name"]), collapse='+')
  }
  # print(unique(as.numeric(pdata$start.time)))
  comb
}
comb.tab <- table(unlist(allcomb))
comb.tab[comb.tab > 10]

# distribution of response to a drug
tcga.pickeddrugs = sort(c('Cisplatin', 'Paclitaxel', 'Carboplatin', 'Fluorouracil'))
# tcga.pickeddrugs = sort(unique(tcga.response$drug.name))
for(dn in tcga.pickeddrugs) {
  print(dn)
  print(table(tcga.response[tcga.response$drug.name == dn, 'response']))
}
# table(tcga.response[tcga.response$drug.name == 'Temozolomide', 'response']) ## imbalanced response, too many non-responders

# binarization of RECIST response score; NOTE: may be change to get more balanced class labels
binarization.scheme <- c(0., 0., 0., 1.)
names(binarization.scheme) <- c("Stable Disease", "Clinical Progressive Disease", "Partial Response", "Complete Response")

TCGA.labels <- matrix(-1, ncol = length(tcga.pickeddrugs), nrow = dim(TCGA.rnaseq)[1])
rownames(TCGA.labels) <- rownames(TCGA.rnaseq)
colnames(TCGA.labels) <- tcga.pickeddrugs
for(dn in colnames(TCGA.labels)) {
  L <- tcga.response[as.character(tcga.response$drug.name) == dn, c('patient.arr', 'response', 'cancers')]
  print(paste("removing patients without gene expression:", dn, sum(!L$patient.arr %in% rownames(TCGA.rnaseq))))
  # print(as.character(L$patient.arr[!L$patient.arr %in% rownames(TCGA.rnaseq)]))
  
  L <- L[L$patient.arr %in% rownames(TCGA.rnaseq), ]
  L$response <- binarization.scheme[as.character(L$response)]
  print(table(droplevels((L[,c('cancers', 'response')]))))
  
  TCGA.labels[as.character(L$patient.arr), dn] <- L$response
}
colnames(TCGA.labels) <- tolower(colnames(TCGA.labels))
# summary of binary labels
apply(TCGA.labels, MARGIN = 2, function(X) table(X))

TCGA.rnaseq <- scale(TCGA.rnaseq)
dim(TCGA.rnaseq)
# included <- read.csv("gene_ontology/included_genes.csv")
# included <- names(sapply(included, levels))
# included <- included[1:length(included)-1]
# TCGA.rnaseq <- TCGA.rnaseq[,included]

#included <- as.numeric(read.csv("gene_ontology/included_genes.csv", header = FALSE))
#TCGA.included <- TCGA.rnaseq[,included]

# filter for patients who were treated with at least 1 of the drugs
TCGA.druglabels <- TCGA.labels[rowSums(TCGA.labels) > -1 * (length(tcga.pickeddrugs) - 1),]
TCGA.outcomes = matrix(0, nrow = dim(TCGA.druglabels)[1], ncol = 1)
for (i in c(1:dim(TCGA.druglabels)[1])) {
  if (any(TCGA.druglabels[i,] > 0)) {
    TCGA.outcomes[i] = 1
  }
}

TCGA.mask <- TCGA.druglabels
TCGA.mask[TCGA.mask == 0] <- 1
TCGA.mask[TCGA.mask == -1] <- 0
dim(TCGA.rnaseq)
dim(TCGA.mask)
TCGA.all <- merge(TCGA.rnaseq, TCGA.mask, by=0)
TCGA.all <- cbind(TCGA.all, TCGA.outcomes)
dim(TCGA.all)

write.csv(TCGA.all, "rnaseq_scaled_symbols.csv", row.names = FALSE)
