Neural networks that use gene ontology terms to decide the architecture. Uses TCGA & NeoALTTO data.

TCGA pipeline (in TCGA-Kallisto):
1. get_gene_names.R
2. gene_ontology/process_GO_genes.py
3. load_TCGA.R
4. gene_ontology/filter_genes.py
5. gene_ontology/pathways.py

NeoALTTO pipeline (in NeoALTTO):
1. get_gene_names.R
2. gene_ontology/process_GO_genes.py
3. process_NeoALTTO.R
4. gene_ontology/pathways.py

