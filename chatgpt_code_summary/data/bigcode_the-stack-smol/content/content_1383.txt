from enum import Enum


class GeneClass(str, Enum):
    PROTEIN_CODING = ("protein coding,nonsense mediated decay",)
    PSEUDOGENE = "pseudogene,unprocessed pseudogene,polymorphic pseudogene,unitary pseudogene,transcribed unprocessed pseudogene,transcribed processed pseudogene, IG pseudogene"
    MICRO_RNA = "micro RNA"
    SHORT_NCRNA = (
        "piRNA,rRNA,siRNA,snRNA,snoRNA,tRNA,scaRNA,vaultRNA,sRNA,misc RNA"
    )
    LONG_NCRNA = "lincRNA,macro IncRNA,prime3 overlapping ncrna,antisense,retained intron,sense intronic,sense overlapping,macro IncRNA,bidirectional IncRNA"
    IMMUNOGLOBULIN = "IG C gene,IG D gene,IG J gene,IG V gene"
    T_CELL_RECEPTOR = "TR C gene,TR J gene, TR V gene"
