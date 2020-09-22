import csv
import reader_writer_csv as rw
from progress.bar import IncrementalBar

bynode_pos = []
bynode_neg = []
bytfidf_pos = []
bytfidf_neg = []

countnode_pos = 1
countnode_neg = -1
counttfidf_pos = 1
counttfidf_neg = -1


def filter():
    file = rw.get_data(f'./deepcut_pythai/max_range_3/tfidf_p95_t7.csv')
    max = len(file)
    bar = IncrementalBar(f'Progress', max=max,
                         suffix='%(percent)d%% %(elapsed_td)s')

    for row in file:
        word = row["word"]
        tfidf_neg = row["tf-idf-neg"]
        tfidf_pos = row["tf-idf-pos"]
        node_label = row["node-label"]

        try:
            if node_label == "pos":
                global countnode_pos
                countnode_pos += 1
                bynode_pos.append([word, node_label])
            elif node_label == "neg":
                global countnode_neg
                countnode_neg -= 1
                bynode_neg.append([word, node_label])

            if tfidf_pos != "0":
                global counttfidf_pos
                counttfidf_pos += 1
                bytfidf_pos.append([word, node_label])
            elif tfidf_neg != "0":
                global counttfidf_neg
                counttfidf_neg -= 1
                bytfidf_neg.append([word, node_label])
        except:
            pass
        bar.next()
    bar.finish()

    fieldnames = ['word', 'node-label']

    with open(f'./weight_tfidf/filter_file_tfidf/filter_bynode_pos.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bynode_pos:
            writer.writerow({'word': row[0], 'node-label': row[1]})

    with open(f'./weight_tfidf/filter_file_tfidf/filter_bynode_neg.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bynode_neg:
            writer.writerow({'word': row[0], 'node-label': row[1]})

    with open(f'./weight_tfidf/filter_file_tfidf/filter_bytfidf_pos.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bytfidf_pos:
            writer.writerow({'word': row[0], 'node-label': row[1]})

    with open(f'./weight_tfidf/filter_file_tfidf/filter_bytfidf_neg.csv', mode='w', newline='', encoding='utf-8') as writefile:
        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bytfidf_neg:
            writer.writerow({'word': row[0], 'node-label': row[1]})


def weight():
    bynode_pos = rw.get_data(
        f'./weight_tfidf/filter_file_tfidf/filter_bynode_pos.csv')
    bynode_neg = rw.get_data(
        f'./weight_tfidf/filter_file_tfidf/filter_bynode_neg.csv')
    bytfidf_pos = rw.get_data(
        f'./weight_tfidf/filter_file_tfidf/filter_bytfidf_pos.csv')
    bytfidf_neg = rw.get_data(
        f'./weight_tfidf/filter_file_tfidf/filter_bytfidf_neg.csv')

    nodepos = countnode_pos
    nodeneg = countnode_neg
    tfidfpos = counttfidf_pos
    tfidfneg = counttfidf_neg

    for row in bynode_pos:
            newval = nodepos/countnode_pos
            row.update({"new-val":newval})
            nodepos -=1
    
    for row in bynode_neg:
            newval = -(nodeneg/countnode_neg)
            row.update({"new-val":newval})
            nodeneg +=1

    for row in bytfidf_pos:
            newval = tfidfpos/counttfidf_pos
            row.update({"new-val":newval})
            tfidfpos -=1

    for row in bytfidf_neg:
            newval = -(tfidfneg/counttfidf_neg)
            row.update({"new-val":newval})
            tfidfneg +=1

    fieldnames = ['word', 'node-label', 'new-val']
    rw.write_data_by_columns(
        f"./weight_tfidf/comp_weight_tfidf/weight_bynode_pos.csv", fieldnames, bynode_pos)
    rw.write_data_by_columns(
        f"./weight_tfidf/comp_weight_tfidf/weight_bynode_neg.csv", fieldnames, bynode_neg)
    rw.write_data_by_columns(
        f"./weight_tfidf/comp_weight_tfidf/weight_bytfidf_pos.csv", fieldnames, bytfidf_pos)
    rw.write_data_by_columns(
        f"./weight_tfidf/comp_weight_tfidf/weight_bytfidf_neg.csv", fieldnames, bytfidf_neg)

def main():
    filter()
    weight()

main()