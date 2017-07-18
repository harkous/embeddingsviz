import argparse
import numpy as np
import faiss  # make faiss available
import time
import random
from pdb import set_trace as bp

random.seed(42)


def read_lines_from_file(filename, unicode=True):
    """Read the file to a list of lines

    Args:
        filename:

    Returns:

    """
    lines = []

    if unicode:
        file = open(filename, 'r')
    else:
        file = open(filename, 'r')

    for line in file:
        lines.append(str(line.rstrip()))

    file.close()

    return lines


def knn(embedding_file, vocab_filename, output_file, k=100, no_header=False):
    """Searches for the  k nearest neighbors of the words in the vocab_filename among those in the embedding_file
    If the file format is fasttext, which has a header then the embedding_dimensions and vocab_size are
    obtained from the header line.
    If not, i.e. like in glove, the embedding_dimensions and vocab_size need to be given as a parameter

    Args:
        embedding_file:
        vocab_filename:
        output_file:
        k:
        embedding_dimensions:
        emb_vocab_size:

    Returns:

    """
    with open(embedding_file, 'r') as inputfile:
        vocab = read_lines_from_file(vocab_filename)
        vocab_dict = {item: '' for item in vocab}

        j = 0
        index_to_word = {}
        print('reading the embeddings')
        lines = inputfile.readlines()

        # as in fasttext
        if no_header:
            emb_vocab_size = len(lines)
            embedding_dimensions = len(lines[0].rstrip().split()) - 1
        else:
            line = lines[0]
            line = line.rstrip()
            values = line.split()
            emb_vocab_size = int(values[0])
            embedding_dimensions = int(values[1])
            lines.pop(0)

        print(
        'detected the embedding_dimensions as ' + str(embedding_dimensions) + ' and the full vocab size as ' + str(
            emb_vocab_size))

        # for storing all the embeddings
        embeddings = np.empty((emb_vocab_size, embedding_dimensions), dtype=np.float32)
        # for storing the embeddings of the vocab
        vocab_embeddings = np.empty((len(vocab_dict), embedding_dimensions), dtype=np.float32)

        for i, line in enumerate(lines):
            line = line.rstrip()
            values = line.split()
            try:
                # account for the cases where the tokens (words) contain spaces
                word = ' '.join(values[0:len(values) - embedding_dimensions]).strip()
                coefs = np.asarray(values[-embedding_dimensions:], dtype='float32')
                embeddings[i] = coefs
                index_to_word[i] = word
                if word in vocab_dict:
                    vocab_embeddings[j] = coefs
                    j += 1
            except:
                # if a specific line cannot be processed
                print(i, line)

        # Based on: https://github.com/facebookresearch/faiss/wiki/Getting-started-tutorial
        print('building index')
        start = time.time()
        index = faiss.IndexFlatL2(embedding_dimensions)  # build the index
        print (index.is_trained)
        index.add(embeddings)  # add vectors to the index
        print (index.ntotal)
        print('time for building index: ', time.time() - start)

        print('searching in the index for the neighbors')
        start = time.time()
        D, I = index.search(vocab_embeddings, k)  # actual search

        # get list of indices for the neighbors in the original embeddings
        indices = []
        for coords, x in np.ndenumerate(I):
            indices.append(x)

        print('time for searching index: ', time.time() - start)

        # finally we write the embedding vectors file, formed of the vocab words and their neighbors
        with open(output_file, 'w') as outputfile:
            # write the header
            outputfile.write(str(len(indices)) + ' ' + str(embedding_dimensions) + '\n')
            for m, ind in enumerate(indices):
                # convert array of floats to string of space-separated numbers
                coefs = ' '.join(map(str, embeddings[ind]))
                outputfile.write(index_to_word[ind] + ' ' + coefs + '\n')
            print('finished writing output')
        print('Success!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create embeddings with the k-nearest neighbors of input vocabulary')
    parser.add_argument('-e', '--embedding', help='Embedding filename', dest='embedding_file')
    parser.add_argument('-v', '--vocab', help='Vocab filename', dest='vocab_filename')
    parser.add_argument('-o', '--output', help='Output embeddings filename', dest='output_file', required=True)
    parser.add_argument('-k', '--k_nearest', help='k', dest='k', type=int)
    parser.add_argument('-n', '--no_header', help='Embeddings file has no header', dest='no_header',
                        action='store_true')

    options = parser.parse_args()

    knn(options.embedding_file,
        options.vocab_filename,
        options.output_file,
        k=options.k,
        no_header=options.no_header)
