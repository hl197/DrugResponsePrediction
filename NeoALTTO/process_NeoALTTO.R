data <- readRDS("NeoALTTO_Exp_Mappping_Meta.rds")
countMat <- data$CountMat
TPMMat <- data$TPMMat
clin <- data$ClinicalInfo
ann <- data$Annotation_mapping
dim(countMat)

drugs <- clin$randarm
names(drugs) <- clin$PatientID
pCR <- clin$pCR
names(pCR) <- clin$PatientID

new_header <- read.csv('gene_ontology/ENSG.csv')$x
row.names(countMat) <- new_header
unique_genes = unique(new_header)

# combine all rows with the same name (same gene)
aggr <- by(countMat, INDICES=row.names(countMat), FUN=colSums) # collapse the rows with the same name
newCountMat <- as.matrix(do.call(cbind,aggr)) # convert by() result to a data.frame
# print
dim(newCountMat)
length(unique_genes)

colnames(newCountMat) <- unique_genes
rownames(newCountMat) <- colnames(countMat)
newCountMat <- scale(newCountMat)

patients <- colnames(countMat)
patient_num <- ann$`Patient screening no.`
names(patient_num) <- ann$`FASTQ files`
num <- as.character(patient_num[patients])
responses <- as.numeric(pCR[num]) - 1
drugs <- as.numeric(drugs[num]) - 1
newCountMat <- cbind(newCountMat, drugs, responses)

# write.csv(newCountMat, "count_all_rnaseq.csv")

included <- read.csv("gene_ontology/included_genes.csv")
included <- names(sapply(included, levels))
included <- c(included[1:length(included)-1], 'drugs', 'responses')
length(included)
processed <- newCountMat[,included]
dim(processed)
write.csv(processed, "count_processed_rnaseq.csv", quote=F)