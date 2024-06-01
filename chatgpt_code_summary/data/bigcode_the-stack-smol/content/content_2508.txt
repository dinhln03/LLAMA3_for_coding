
import config
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



"""
Versuch #1
"""

# Gibt den classification-report aus
def evaluate(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    Y_pred = Y_pred.argmax(axis=-1)
    Y_test = Y_test.argmax(axis=-1)

    print(classification_report([Y_test], [Y_pred]))

# Nimmt ein history-Objekt und zeichnet den loss für
# sowohl testing als auch training Daten.
def plot_model(history, fold):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='test_loss')
    plt.legend()
    plt.savefig(f"../plots/covid_model_without_vaccine_loss_{config.EPOCHS}epochs_{fold}v{config.K_FOLD_SPLITS}fold.png")
    clear_plot()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train_acc', c="r")
    plt.plot(history.history['val_accuracy'], label='test_acc', c="b")
    plt.legend()
    plt.savefig(f"../plots/covid_model_without_vaccine_accuracy_{config.EPOCHS}epochs_{fold}v{config.K_FOLD_SPLITS}fold.png")
    clear_plot()

def clear_plot():
    plt.close()
    plt.cla()
    plt.clf()

def plot_confusion_matrix(model, X_test, y_test, fold):
    y_pred = model.predict(X_test)

    y_pred = y_pred.argmax(axis=-1)
    y_test = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_test, y_pred)

    ax=plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title(f'Confusion Matrix – {config.EPOCHS}|{fold}') 
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])

    plt.savefig(f"../plots/covid_confusion_{config.EPOCHS}epochs_{fold}v{config.K_FOLD_SPLITS}fold.png")
    clear_plot()

# Erstellen eines Tokenizers für das LSTM Modell
def create_tokenizer(df, save_path):
    tokenizer = Tokenizer(num_words=config.MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    words = df.link.values.tolist()
    words.extend(df.meta_data.values.tolist())
    words.extend(df.title.values.tolist())
    words.extend(df.body.values.tolist())
    tokenizer.fit_on_texts(words)
    save_tokenizer(tokenizer, save_path)
    return tokenizer

# Laden und speichern des Tokenizers
def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer 

"""
Die in Tokens verwandelte Texte sehen so aus:
    [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
gepaddet sehen sie so aus:
    [[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  2  3  4]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  6  7]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8  9 10 11 12]]

werden danach die Covid Count Zahlen angefügt, sieht die Repräsentation beispielsweise so aus
    [[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  2  3  4 10 20 30]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  5  6  7 40 50 60]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8  9 10 11 12 70 80 90]]

Das np.expand ist notwendig, um das array in beispielsweise folgende Form zu bringen: [ 2 1 20] => [ [2] [1] [20]]     
"""
def transform_text(tokenizer, df):
    if (isinstance(tokenizer, str)):
        tokenizer = load_tokenizer(tokenizer)

    # Tokenizing der Link Informationen
    X_input = tokenizer.texts_to_sequences(df['link'].values)
    X_input = pad_sequences(X_input, maxlen=config.MAX_LINK_SEQUENCE_LENGTH)
    # Tokenizing der Meta Informationen
    X_meta = tokenizer.texts_to_sequences(df['meta_data'].values)
    X_meta = pad_sequences(X_meta, maxlen=config.MAX_META_SEQUENCE_LENGTH)
    # Tokenizing der Titel Informationen
    X_title = tokenizer.texts_to_sequences(df['title'].values)
    X_title = pad_sequences(X_title, maxlen=config.MAX_TITLE_SEQUENCE_LENGTH)
    # Tokenizing des Seiteninhalts
    X_body = tokenizer.texts_to_sequences(df['body'].values)
    X_body = pad_sequences(X_body, maxlen=config.MAX_BODY_SEQUENCE_LENGTH)
    covid_word_count = df['covid_word_count'].values
    covid_word_count_url = df['covid_word_count_url'].values
    restriction_word_count = df['restriction_word_count'].values
    restriction_word_count_url = df['restriction_word_count_url'].values

    X_input = np.concatenate([X_input, X_meta], axis=-1)
    X_input = np.concatenate([X_input, X_title], axis=-1)
    X_input = np.concatenate([X_input, X_body], axis=-1)

    covid_word_count = np.expand_dims(covid_word_count, axis=(-1))
    X_input = np.concatenate([X_input, covid_word_count], axis=-1)

    covid_word_count_url = np.expand_dims(covid_word_count_url, axis=(-1))
    X_input = np.concatenate([X_input, covid_word_count_url], axis=-1)

    restriction_word_count = np.expand_dims(restriction_word_count, axis=(-1))
    X_input = np.concatenate([X_input, restriction_word_count], axis=-1)

    restriction_word_count_url = np.expand_dims(restriction_word_count_url, axis=(-1))
    X_input = np.concatenate([X_input, restriction_word_count_url], axis=-1) # Schlussendlich alles zusammefügen

    return X_input

def remove_stopwords(df):
    ger = pd.read_csv(config.STOPWORDS_PATH)['stopwords'].values

    df['link'] = df['link'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (ger)]))
    df['meta_data'] = df['meta_data'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (ger)]))
    df['title'] = df['title'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (ger)]))
    df['body'] = df['body'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (ger)]))

    return df


# Nimmt den input DataFrame und einen LabelEncoder Objekt, 
# trainiert ein LSTM Modell, speichert es, evaluiert es 
# und gibt den Loss aus.
def train_model(train_df, valid_df, tokenizer, fold):
    
    X_train = transform_text(tokenizer, train_df)
    X_valid = transform_text(tokenizer, valid_df)
    Y_train = pd.get_dummies(train_df['label'])
    Y_valid = pd.get_dummies(valid_df['label']).to_numpy()
    
    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(1e-3) # 0.001
    model.add(Embedding(config.MAX_NUM_WORDS, config.EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, bias_regularizer=regularizers.l2(1e-4),)) # TODO: damit rumspielen
    model.add(Dense(2, activation='softmax'))
    loss='categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE # 64

    #es = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2) # , callbacks=[es]
    accr = model.evaluate(X_valid,Y_valid)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save(f"{config.MODEL_PATH}_without_vaccine_{fold}.h5")
    evaluate(model, X_valid, Y_valid)
    plot_model(history, fold)

    plot_confusion_matrix(model, X_valid, Y_valid, fold)

# Laden und evaluieren eines existierenden Modells
def load_and_evaluate_existing_model(model_path, tokenizer_path, df, le):
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    X = transform_text(tokenizer, df['text'].values)
    Y = pd.get_dummies(df['label']).values
    evaluate(model, X, Y, le)

# Testen eines neuen Beispiels. Hauptsächlich zu Testzwecken während der Entwicklung
# Die Funktion nimmt einen String, den Classifier,
# den Vectorizer und einen LabelEncoder und 
# gibt eine Vorhersage zurück.
def test_new_example(model, tokenizer, le, text_input):
    X_example = transform_text(tokenizer, [text_input])
    label_array = model.predict(X_example)
    new_label = np.argmax(label_array, axis=-1)
    print(new_label)
    print(le.inverse_transform(new_label))

def run(df, fold, use_vaccine):

    # der Trainingdataframe
    train_df = df[df.kfold != fold].reset_index(drop=True)
    print(f"Länge Traing_DF  {len(train_df)}")
    # Validation Dataframe
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    print(f"Länge Valid_DF  {len(valid_df)}")

    # Das Validationset enthält weiterhin die Impf-Beispiele
    # Bei 10 Folds sind die Sets folgendermaßen aufgeteil:
    # 0 – 126
    # 1 – 78
    # 2 – 10
    if not use_vaccine:
        train_df = train_df[train_df['label'] != 2]

    # Jetzt müssen alle 2 er noch in einsergewandelt werden
    train_df['label'] = train_df['label'].apply(lambda x : 1 if x > 0 else 0)
    valid_df['label'] = valid_df['label'].apply(lambda x : 1 if x > 0 else 0)


    print("Fitting tokenizer")
    # tf.keras Tokenizer
    tokenizer = create_tokenizer(train_df, f"{config.TOKENIZER_SAVE_PATH}_{fold}.pickle")
    
    train_model(train_df, valid_df, tokenizer, fold)
    
    # load_and_evaluate_existing_model(f"{config.MODEL_PATH}_{fold}", config.TOKENIZER_PATH, df, le)
    #model = load_model(config.MODEL_PATH)
    #tokenizer = config.TOKENIZER_PATH

if (__name__ == "__main__"):

    tf.get_logger().setLevel('ERROR')

    # load data
    df = pd.read_csv(config.DATASET_PATH).sample(frac=1)
    df = remove_stopwords(df)

    """
    # TODO: ein Test, Gleichverteilung
    """
    df2 = df[df['label'] != 0]
    
    # Wir nehmen einfach den hinteren Teil des Körpers und den Metadaten
    df2['body'] = df2['body'].apply(lambda x : str(x)[config.MAX_BODY_SEQUENCE_LENGTH:])
    df2['meta_data'] = df2['meta_data'].apply(lambda x : str(x)[config.MAX_META_SEQUENCE_LENGTH:])

    df = df.append(df2, ignore_index=True).reset_index()
    

    # initiate the kfold class from the model_selection module
    kf = StratifiedKFold(n_splits=config.K_FOLD_SPLITS)

    # füllen den kfold Spalte
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.label.values)):
        df.loc[v_, 'kfold'] = f


    # training für alle Faltungen
    for i in range(config.K_FOLD_SPLITS):
        print(f"\n–––––––––––– FOLD {i} ––––––––––––\n")
        run(df, fold=i, use_vaccine=config.USE_VACCINE)