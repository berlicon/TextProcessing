using System;
using System.Linq;
using System.IO;
using Numpy;
using Keras.Callbacks;
using Keras.Datasets;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using Keras.PreProcessing.sequence;
using Keras.PreProcessing.Text;
using System.Collections.Generic;

namespace TextProcessing
{
    public class SentimentClassification
    {
        public static Dictionary<string, int> indexesByFrequency = new Dictionary<string, int>();
        // Load the dataset but only keep the top n words, zero the rest
        public static int top_words = 10000;
        public static int max_words = 500;
        public static int train_count = 9000;   //14.412 записей, но на ~9.5 тыс дохнет, не хватает памяти
        public static int test_count = 100;

        public static void Run()
        {
            //var ((x_train, y_train), (x_test, y_test)) = IMDB.LoadData(num_words: top_words);
            var ((x_train, y_train), (x_test, y_test)) = LoadDataRussianLanguageToxicComments(
                trainCount: train_count, testCount: test_count, numWords: top_words, maxWords: max_words);

            //Не нужно массивы дополнять до 500 элементов, т.к. они уже размером в 500 элементов
            //x_train = SequenceUtil.PadSequences(x_train, maxlen: max_words);
            //x_test = SequenceUtil.PadSequences(x_test, maxlen: max_words);

            //Create model
            Sequential model = new Sequential();
            model.Add(new Embedding(top_words, 32, input_length: max_words));
            model.Add(new Conv1D(filters: 32, kernel_size: 3, padding: "same", activation: "relu"));
            model.Add(new MaxPooling1D(pool_size: 2));
            model.Add(new Flatten());
            model.Add(new Dense(250, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(loss: "binary_crossentropy", optimizer: "adam", metrics: new string[] { "accuracy" });
            model.Summary();

            // Fit the model
            model.Fit(x_train, y_train, validation_data: new NDarray[] { x_test, y_test },
                epochs: 2/*10*/, batch_size: 128, verbose: 2);
            // Final evaluation of the model
            var scores = model.Evaluate(x_test, y_test, verbose: 0);
            Console.WriteLine("Accuracy: " + (scores[1] * 100));

            model.Save("model.h5");
            File.WriteAllText("model.json", model.ToJson());    //save model
            //model.SaveTensorflowJSFormat("./");   //error - Cannot perform runtime binding on a null reference
        }

        public static void Predict(string text)
        {
            var model = Sequential.LoadModel("model.h5");
            string result = "";

            float[,] tokens = new float[1, max_words];
            string[] words = TextUtil.TextToWordSequence(text).Take(max_words).ToArray();
            for (int i = 0; i < words.Length; i++)
            {
                tokens[0, i] = indexesByFrequency.ContainsKey(words[i]) 
                    ? (float)indexesByFrequency[words[i]] : 0f;
            }

            NDarray x = np.array(tokens);
            var y = model.Predict(x);
            var binary = Math.Round(y[0].asscalar<float>());
            result = binary == 0 ? "Норм, не токсично" : "ТОКСИЧНО";
            Console.WriteLine($"Результат для \"{text}\": {result}, оценка: {y[0].asscalar<float>()}");
        }

        /// <summary>
        /// Получить массивы индексов слов из набора русских комментов и флаг токсичности
        /// https://www.kaggle.com/blackmoon/russian-language-toxic-comments/data
        /// 14.412 строк, 4Mb, 2 колонки : comment(text), toxic(0 или 1)
        /// Плохое качество датасета: оценка то на одной строке, то на разных, то кавычки, то нет.
        /// </summary>
        /// <param name="trainCount">Число комментов для тренировки</param>
        /// <param name="testCount">Число комментов для теста</param>
        /// <param name="numWords">Обработать ТОП Х слов (по популярности)</param>
        /// <param name="maxWords">Макс число слов в комменте для обработки</param>
        /// <returns>Массивы: ((x_train, y_train), (x_test, y_test))</returns>
        private static ((NDarray, NDarray), (NDarray, NDarray)) LoadDataRussianLanguageToxicComments(
            int trainCount = 1000, int testCount = 100, int numWords = 1000, int maxWords = 500)
        {
            const string FILE_PATH = @"G:\My\_Projects\TextProcessing\labeled.csv";

            Console.WriteLine("Читаем данные из файла");
            var text = File.ReadAllText(FILE_PATH).Substring("comment,toxic".Length);
            //var indexes = IMDB.GetWordIndex();
            //var tokenizer = new Tokenizer(num_words: numWords);
            //tokenizer.FitOnTexts(texts: rows.ToArray());  //fit_on_texts() missing 1 required positional argument: 'texts'
            //var seq = tokenizer.TextsToSequences(new string[] { rows[0] });

            string[] textWords = TextUtil.TextToWordSequence(text);
            var indexes = new Dictionary<string, int>();
            foreach (var word in textWords)
            {
                //if (word.Length < 3) continue;    //можно не учитывать короткие слова и цифры
                if (word == "1" || word == "0") continue;

                if (indexes.ContainsKey(word))
                {
                    indexes[word]++;
                }
                else
                {
                    indexes[word] = 1;
                }
            }

            var orderedIndexes = indexes.OrderByDescending(x => x.Value);
            var index = 1;
            foreach (var item in orderedIndexes.Take(numWords - 1))
            {
                indexesByFrequency[item.Key] = index++;
            }

            //var x = TextUtil.HashingTrick(text, 0);//ZeroDivisionError : integer division or modulo by zero
            //var y = TextUtil.OneHot(text, 0);//ZeroDivisionError : integer division or modulo by zero

            Console.WriteLine("Заполняем датасеты данными");
            float[,] x_train = new float[trainCount, maxWords];
            float[] y_train = new float[trainCount];
            float[,] x_test = new float[testCount, maxWords];
            float[] y_test = new float[testCount];

            var startIndex = 0;
            for (int i = 0; i < trainCount + testCount; i++)
            {
                Console.WriteLine($"Итерация {i + 1} из {trainCount + testCount}");
                var endIndex = text.IndexOf(".0\n", startIndex);
                var commentWithFlag = text.Substring(startIndex, endIndex - startIndex);

                var comment = commentWithFlag.Substring(0, commentWithFlag.Length - 1);
                float toxic = (commentWithFlag[commentWithFlag.Length - 1]) == '1' ? 1f : 0f;

                string[] words = TextUtil.TextToWordSequence(comment).Take(maxWords).ToArray();
                float[] tokens = words.Select(word =>
                indexesByFrequency.ContainsKey(word) ? (float)indexesByFrequency[word] : 0f).ToArray();

                if (i < trainCount)
                {
                    for (int j = 0; j < tokens.Length; j++) { x_train[i, j] = tokens[j]; }
                    y_train[i] = toxic;
                }
                else
                {
                    for (int j = 0; j < tokens.Length; j++) { x_test[i - trainCount, j] = tokens[j]; }
                    y_test[i - trainCount] = toxic;
                }
                startIndex = endIndex + ".0\n".Length;
            }

            var result = ((new NDarray(x_train), new NDarray(y_train)),
                          (new NDarray(x_test), new NDarray(y_test)));
            return result;
        }

        ////https://habr.com/ru/post/470035/ - Бредогенератор: создаем тексты на любом языке с помощью нейронной сети
        ////https://keras.io/examples/lstm_text_generation/ - Example script to generate text from Nietzsche's writings.
        ///Начал переводить с питона на C# но бросил. Не вижу практической пользы в бредогенераторе :]
        //public static void Do()
        //{
        //    string text = System.IO.File.ReadAllText(@"nietzsche.txt");
        //    Console.WriteLine("corpus length:", text.Length);

        //    var chars = sorted(list(set(text)));    //коллекция уникальных символов в тексте
        //    Console.WriteLine("total chars:", len(chars));  //число уник символов
        //    var char_indices = dict((c, i) for i, c in enumerate(chars));   //хэш key=char, value=index
        //    var indices_char = dict((i, c) for i, c in enumerate(chars)) ;  //хэш key=index, value=char

        //    // cut the text in semi-redundant sequences of maxlen characters
        //    var maxlen = 40;
        //    var step = 3;
        //    var sentences = new string[];
        //    var next_chars = new string[];
        //    for (i in range(0, len(text) - maxlen, step))
        //    {
        //        sentences.append(text[i: i + maxlen]);  //массив предложений по 40 символов
        //        next_chars.append(text[i + maxlen]);
        //    }
        //    Console.WriteLine("nb sequences:", sentences.Length); //число предложений

        //    Console.WriteLine("Vectorization...");
        //    var x = np.zeros((sentences.Length, maxlen, len(chars)), dtype : np.bool_);  //массив с 0/false
        //    var y = np.zeros((sentences.Length, len(chars)), dtype : np.bool_);          //массив с 0/false
        //    for (i, sentence in enumerate(sentences))
        //    {
        //        for (t, vchar in enumerate(sentence))
        //        {
        //            x[i, t, char_indices[vchar]] = 1;//ставим 1 в масссив там где символ есть в хеше
        //        }
        //        y[i, char_indices[next_chars[i]]] = 1;
        //    }

        //    // build the model: a single LSTM
        //    Console.WriteLine("Build model...");
        //    var model = new Sequential();
        //    model.Add(new LSTM(128, input_shape : (maxlen, len(chars))));
        //    model.Add(new Dense(len(chars), activation: "softmax"));

        //    var optimizer = new RMSprop(lr: 0.01f);
        //    model.Compile(loss : "categorical_crossentropy", optimizer : optimizer);

        //    var print_callback = new LambdaCallback(on_epoch_end = OnEpochEnd());
        //    model.Fit(x, y,
        //              batch_size : 128,
        //              epochs : 60,
        //              callbacks : [print_callback]);
        //}

        //public static NDarray Sample(NDarray preds, float temperature = 1.0f)
        //{
        //    // helper function to sample an index from a probability array
        //    preds = np.asarray(preds).astype(np.float32);
        //    preds = np.log(preds) / temperature;
        //    var exp_preds = np.exp(preds);
        //    preds = exp_preds / np.sum(exp_preds);
        //    var probas = np.random.multinomial(1, preds, 1);
        //    return np.argmax(probas);
        //}

        //public static void OnEpochEnd(int epoch, _)
        //{
        //    // Function invoked at end of each epoch. Prints generated text.
        //    Console.WriteLine();
        //    Console.WriteLine("----- Generating text after Epoch: " + epoch);

        //    var start_index = random.randint(0, len(text) - maxlen - 1);
        //    for (diversity in [0.2, 0.5, 1.0, 1.2])
        //    {
        //        Console.WriteLine("----- diversity:", diversity);

        //        var generated = "";
        //        var sentence = text[start_index: start_index + maxlen];
        //        generated += sentence;
        //        Console.WriteLine("----- Generating with seed: " + sentence);
        //        Console.Write(generated);

        //        for (i in range(400))
        //        {
        //            var x_pred = np.zeros((1, maxlen, len(chars)));
        //            for (t, char in enumerate(sentence))
        //            {
        //                x_pred[0, t, char_indices[char]] = 1.;
        //            }

        //            var preds = model.predict(x_pred, verbose = 0)[0];
        //            var next_index = Sample(preds, diversity);
        //            var next_char = indices_char[next_index];

        //            sentence = sentence[1:] + next_char;

        //            Console.Write(next_char);
        //        }
        //        Console.WriteLine();
        //    }
        //}
    }
}