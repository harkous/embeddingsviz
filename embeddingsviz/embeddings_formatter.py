import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import argparse


def create_embeddings(sess, log_dir, embedding_file='', tensor_name='embedding'):
    """ Add the embeddings to input TensorFlow session and writes a metadata_file containing the words in the vocabulary

    :param sess: TF session
    :param log_dir: destination directory for the model and metadata (the one to which TensorBoard points)
    :param embedding_file: embedding file
    :param tensor_name: tensor name
    :return:
    """
    embedding = None
    embedding_dimensions = 0
    vocab_size = 0
    # write labels
    with open(os.path.join(log_dir, tensor_name + '_' + 'metadata.tsv'), 'w') as metadata_file:

        with open(embedding_file, 'r') as inputfile:

            for i, line in enumerate(inputfile):

                line = line.rstrip()
                values = line.split()

                # the first line is always the header based on what we produce in the embeddings_knn.py
                if i == 0:
                    vocab_size = int(values[0])
                    embedding_dimensions = int(values[1])
                    embedding = np.empty((vocab_size, embedding_dimensions), dtype=np.float32)
                else:
                    # accounts for the case of words with spaces
                    word = ' '.join(values[0:len(values) - embedding_dimensions]).strip()
                    coefs = np.asarray(values[-embedding_dimensions:], dtype='float32')
                    embedding[i - 1] = coefs
                    metadata_file.write(word + '\n')

    X = tf.Variable([0.0], name=tensor_name)
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)

    sess.run(set_x, feed_dict={place: embedding})


def add_multiple_embeddings(log_dir, file_list, name_list):
    """ Creates the files necessary for the multiple embeddings

    :param log_dir: destination directory for the model and metadata (the one to which TensorBoard points)
    :param file_list: list of embeddings files
    :param name_list: names of the embeddings files
    :return:
    """
    # setup a TensorFlow session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    config = projector.ProjectorConfig()

    for i, file in enumerate(file_list):
        tensor_name = name_list[i]

        print('creating the embedding with the name ' + tensor_name)
        create_embeddings(sess, log_dir, embedding_file=file,
                          tensor_name=tensor_name)
        # create a TensorFlow summary writer
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = tensor_name + ':0'
        embedding_conf.metadata_path = os.path.join(tensor_name + '_' + 'metadata.tsv')
        projector.visualize_embeddings(summary_writer, config)

        # save the model
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_dir, tensor_name + '_' + "model.ckpt"))

    print('finished successfully!')


if __name__ == "__main__":
    # create_embeddings(embedding_file='/mnt/storage01/lebret/data/embeddings/glove_noswearing_embeddings.vec',
    #                   tensor_name='glove_no_swearing')
    parser = argparse.ArgumentParser(description='Create files for tensorboard visualization')
    parser.add_argument('-l', '--logfile',
                        help='Path for log file read by TensorBoard; defaults to "log"; ' +
                             'file is created if it doesn\'t exist',
                        dest='logfile')
    parser.add_argument('-f', '--files',
                        help='List of embedding files to be visualized', nargs='+', type=str,
                        dest='file_list', required=True)
    parser.add_argument('-n', '--names',
                        help='List of names you want to assign to the embeddings', nargs='+', type=str,
                        dest='name_list', required=True)

    options = parser.parse_args()
    if options.logfile:
        log_dir = options.logfile
    else:
        log_dir = 'log'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_list = options.file_list
    name_list = options.name_list

    add_multiple_embeddings(log_dir, file_list, name_list)
