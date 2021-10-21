from Bio import SeqIO
import csv
import pandas

""" Experimenting with fasta and tsv files """

"""fasta files"""
fasta_sequence = SeqIO.parse("sars_cov_spike_protein_data\spikeprot.fasta", "fasta") 
# for record in fasta_sequence:
#     print(record.id)


# only for files with only one record
# fasta_read = SeqIO.read("sars_cov_spike_protein_data\spikeprot.fasta", "fasta") only for files with only one record


# only the first record 
first_record = next(fasta_sequence)
print (first_record.id)

"""tsv files"""
tsv_file = open("sars_cov_spike_protein_data\hcov_global.tsv")
tsv_table = csv.reader(tsv_file, delimiter = "\t")
first_tsv_line = next(tsv_file)
print (first_tsv_line)
i = 1
for line in tsv_file:
    if i>0 :
        print (line)
        i-=1
    else:
        break


tsv_data = pandas.read_csv("sars_cov_spike_protein_data\hcov_global.tsv", sep = "\t")
print(tsv_data)