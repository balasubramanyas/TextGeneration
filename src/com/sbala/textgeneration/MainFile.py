'''
Created on Nov 15, 2018

@author: balasubramanyas
'''


from com.sbala.textgeneration import TextDataConverter as TDC
from com.sbala.textgeneration import TextGenerationModel as TGM
import tensorflow as tf

filepath = "/home/balasubramanyas/ProjectWorkspace/Dataset/shakespeare.txt"
sequenceLength = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000 
embedding_dim = 256
rnn_units = 1024
number_of_unique_chars = len(TDC.ReadDataSet(filepath).getUniqueChars());
dataset = TDC.TrainAndTargetData(sequenceLength, filepath, BUFFER_SIZE, BATCH_SIZE).getTrainAndTargetDataSet()

tgmModel = TGM.TextGenModel(number_of_unique_chars, embedding_dim,
                            rnn_units, 2).buildModel();

idx2char = TDC.ReadDataSet(filepath).getIndexToChar()
char2idx = TDC.ReadDataSet(filepath).getUniqueCharToIndex()
checkpoint_dir = '/home/balasubramanyas/ProjectWorkspace/Dataset/training_checkpoints'
tgmModel.compile(optimizer=tf.train.AdamOptimizer(),
                           loss=tf.losses.sparse_softmax_cross_entropy)
tgmModel.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

tgmModel.build(tf.TensorShape([BATCH_SIZE, None]))
tgmModel.summary()

print(TGM.TrainModelWithDataSet(tgmModel,dataset).genrateText("REMO: ", char2idx, idx2char))

'''                        
for input_example_batch, target_example_batch in dataset.take(1): 
    example_batch_predictions = tgmModel(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    sampled_indices = tf.multinomial(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

'''
'''
text_data_len = len(TDC.ReadDataSet(filepath).readData()) // sequenceLength;
train_data_len = text_data_len // BATCH_SIZE
epochs = 3
print(train_data_len)
history = TGM.TrainModelWithDataSet(tgmModel,dataset).trainModelWithEpoch(epochs, 
                                                                train_data_len)
'''
