import math

def is_number(s):
    if s != s.strip():
        return False
    try:
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return False
        return True
    except ValueError:
        return False

def data_check(file, file_type="doc"):
    """ check if a file is UTF8 without BOM,
        doc_embedding index starts with 1,
        query_embedding index starts with 200001,
        the dimension of the embedding is 128.
    """
    count = 1
    id_set = set()
    with open(file) as f:
        for line in f:
            sp_line = line.strip('\n').split("\t")
            if len(sp_line) != 2:
                print("[Error] Please check your line. The line should be two parts, i.e. index \t embedding")
                print("line number: ", count)
                return
            index, embedding = sp_line

            if not is_number(index):
                print("[Error] Please check your id. The id should be int without other char")
                print("line number: ", count)
                return
            id_set.add(int(index))

            embedding_list = embedding.split(',')
            if len(embedding_list) != 128:
                print("[Error] Please check the dimension of embedding. The dimension is not 128")
                print("line number: ", count)
                return
            for emb in embedding_list:
                if not is_number(emb):
                    print("[Error] Please check your embedding. The embedding should be float without other char")
                    print("line number: ", count)
                    return

            count += 1

        if file_type == "doc":
            for i in range(1, 1001501):
                if i not in id_set:
                    print("[Error] The index[{}] of doc_embedding is not found. Please check it.".format(i))
                    return
        elif file_type == "query":
            for i in range(200001, 201001):
                if i not in id_set:
                    print("[Error] The index[{}] of query_embedding is not found. Please check it.".format(i))
                    return

    print("Check done!\n")


if __name__ == "__main__":
    print("*"*10, "Checking query_embedding ...")
    data_check("query_embedding", file_type="query")
    print("*"*10, "Checking doc_embedding ...")
    data_check("doc_embedding", file_type="doc")

