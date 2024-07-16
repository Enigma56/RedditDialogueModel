import xml.etree.ElementTree as ET
import math
import pandas
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
# from keras.callbacks import *
from keras.metrics import *

# Retrieves and creates xml tree
tree = ET.parse('RedditAnnotation.txt.xml')
root = tree.getroot()

class ParseXML:

    def __init__(self, data, txt):
        self.tree = ET.parse(data)
        self.root = tree.getroot()
        self.txtLines = self.read_txt(txt)
        self.category_map = {'Inquiry': 0, 'Solution': 1, 'Statement': 2, 'Reference': 3, 'Removed': 4}
        self.posts = dict()
        self.comments = dict()
        self.comments_train = dict()
        self.comments_test = dict()
        self.relate = dict()
        self.retrieve_comments()
        self.retrieve_posts()

    # PREP FUNCTIONS

    @staticmethod
    def read_txt(txt) -> dict:
        line_dict = dict()
        with open(txt, encoding='utf-8') as f:
            line_num = 1
            for line in f:
                line_dict[line_num] = line.rstrip("\n")
                line_num += 1
        return line_dict

    def retrieve_posts(self):
        for post in root.iter('POST'):
            self.posts[post.attrib["line"]] = self.txtLines[int(post.attrib["line"])][3:] + "," + post.attrib["type"]

        # partitions data into train and test
        post_train_size = math.floor(len(self.posts) * .8)
        for i, key in enumerate(self.posts):
            if i < post_train_size or i == post_train_size:
                self.posts[key] = "train," + self.posts[key]
            elif i > post_train_size:
                self.posts[key] = "test," + self.posts[key]

    def retrieve_comments(self):
        for comment in root.iter('COMMENT'):
            self.comments[comment.attrib["line"]] = self.txtLines[int(comment.attrib["line"])][3:] + ",;/" + str(self.category_map[comment.attrib["type"]])

        # Partitions data into train and test
        comment_train_size = math.floor(len(self.comments) * .8)
        for i, key in enumerate(self.comments):
            if i < comment_train_size or i == comment_train_size:
                self.comments_train[key] = "train,;/" + self.comments[key]
            elif i > comment_train_size:
                self.comments_test[key] = "test,;/" + self.comments[key]

    def to_csv(self):
        csv_columns = "set,;/comment,;/type\n"
        with open('comments_train.csv', 'w', encoding="utf-8") as f:
            f.write(csv_columns)
            for key in self.comments_train.keys():
                f.write(f"{self.comments_train[key]}\n")
        with open('comments_test.csv', 'w', encoding="utf-8") as f:
            f.write(csv_columns)
            for key in self.comments_test.keys():
                f.write(f"{self.comments_test[key]}\n")


'''
Creates a deep neural net to classify comments from reddit subs based on guidelines. Vocab is used as features with word
embeddings made from the dataset as well.
'''


class Model:  # A WHOLE REWRITE MAY BE NEEDED

    def __init__(self):
        self.train = pandas.read_csv("comments_train.csv", sep=",;/")
        self.train = self.train.astype(str)

        self.test = pandas.read_csv("comments_test.csv", sep=",;/")
        self.test = self.test.astype(str)

        # comments and labels for traning data
        self.x_train = self.train['comment'].values
        self.y_train = self.train['type'].values.astype(float)

        # comments and labels for traning data
        self.x_test = self.test['comment'].values
        self.y_test = self.test['type'].values.astype(float)

        self.x_train_seq = None
        self.x_test_seq = None

        self.vocab_size = None
        
        self.tokenizer = Tokenizer()

    # ML MODEL FUNCTIONS

    def process_vocab(self):
        self.tokenizer.fit_on_texts(list(self.x_train)) #fits tokenizer on text

        # converts text to a squence of numbers
        x_train_seq = self.tokenizer.texts_to_sequences(self.x_train)
        x_test_seq = self.tokenizer.texts_to_sequences(self.x_test)

        # pads sequences for data well-roundedness
        self.x_train_seq = pad_sequences(x_train_seq, maxlen=128)
        self.x_test_seq = pad_sequences(x_test_seq, maxlen=128)

        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(self.vocab_size)

    def scratch_model_creation_execution(self):
        model = Sequential()  # type of model

        # embedding layer
        # model.add(Embedding(self.vocab_size, 300, input_length=128, trainable=True))

        # lstm layer
        # model.add(LSTM(128, return_sequences=True, dropout=0.2))

        # Global Maxpooling
        # model.add(GlobalMaxPooling1D())

        # Dense Layer
        model.add(Dense(64, input_dim=128,  activation='relu'))
        model.add(Dense(1, activation='softmax'))

        # model.add(Dropout(.2))

        # Add loss function, metrics, optimizer
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(), Precision(), "acc"])

        # Adding callbacks
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        # mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)

        # Print summary of model
        print(model.summary())

        history = model.fit(np.array(self.x_train_seq), np.array(self.y_train), batch_size=128, epochs=10,
                            validation_data=(np.array(self.x_test_seq), np.array(self.y_test)), verbose=1)

        ## MODEL DOE SNOT FIT CORRECTLY BC OF VERSION UPDATE TO TENSORFLOW
        predictions = model.predict((np.array(self.x_test_seq), np.array(self.y_test)), batch_size=128)


        print(predictions)

       # model = load_model('best_model.h5')

        _, recall, precision, val_acc = model.evaluate(self.x_test_seq, self.y_test, batch_size=128)
        f1 = 2*(precision * recall)/(precision + recall)
        print(f"Recall: {recall}, Precision: {precision}, F1: {f1}, val_acc")


if __name__ == "__main__":
    # Run to make changes to csv file
    # parser = ParseXML("RedditAnnotation.txt.xml", "RedditAnnotation.txt")
    # parser.to_csv()

    # Runs Model
    model = Model()
    model.process_vocab()
    model.scratch_model_creation_execution()


