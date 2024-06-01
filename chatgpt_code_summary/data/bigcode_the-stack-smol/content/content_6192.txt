class Hamming:
    def distance(self, gene1, gene2):
        if type(gene1) != str or type(gene2) != str:
            return "Genes have to be strings"
        if len(gene1) != len(gene2):
            return "Genes have to have same lenghts"
        diff = 0
        for i in range(0, len(gene1)):
            if gene1[i] != gene2[i]:
                diff += 1
        return diff