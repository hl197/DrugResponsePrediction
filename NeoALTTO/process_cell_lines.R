load("Lapatinib_CTRPv2.RData")
write.table(colnames(expression_logTPM), file="cell_lines_ENSG.txt", row.names=FALSE, col.names=FALSE, quote=FALSE)

colnames(expression_logTPM) <- unlist(read.table("cell_lines_genes.txt"))
ordered_symbols <- data$GeneAnnot$Symbol
cl_symbols <- colnames(expression_logTPM)
matched_expr <- matrix(0, ncol=length(ordered_symbols), nrow=dim(expression_logTPM)[1])

for (i in c(1:length(ordered_symbols))) {
  if (ordered_symbols[i] %in% cl_symbols) {
    matched_expr[,i] = expression_logTPM[,ordered_symbols[i]]
  }
}

matched_expr <- scale(matched_expr)
colnames(matched_expr) <- ordered_symbols
dim(matched_expr)

drugMat <- matrix(0, ncol=2, nrow=dim(matched_expr)[1])
for (i in c(dim(matched_expr)[1])) {
  drugMat[i, 1] = 1
}
colnames(drugMat) <- c('Lapatinib', 'Trastuzumab')

outMat <- cbind(matched_expr, drugMat, lapatinib$response)
write.csv(outMat, "cell_lines_scaled_symbols.csv", quote=F)
