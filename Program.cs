using System;

namespace TextProcessing
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                SentimentClassification.Run();

                Console.WriteLine("=== Проверяем токсичные коменты ===");
                SentimentClassification.Predict("В Гестапо иди работай");
                SentimentClassification.Predict("Этот слабоумный покупает прокси же.");
                SentimentClassification.Predict("Старый жырик это клоун у пидорасов.");
                SentimentClassification.Predict("А Лизонька то как там наша ? Живая хоть? Животик не болит?");
                SentimentClassification.Predict("А ты чурка. Иди монобровь сбрей.");
                SentimentClassification.Predict("О сектантов завезли. Раскол МУДЭ произошёл?");
                SentimentClassification.Predict("к нам в иред У тебя нет треда своего, сучка");
                SentimentClassification.Predict("если только скота. обычные русские норм живут");
                SentimentClassification.Predict("Подустал браток... Ностальгирующий критик тупой мудак.");
                SentimentClassification.Predict("Мерзкие леваки повсюду - фе");

                Console.WriteLine("=== Проверяем НЕ-токсичные коменты ===");
                SentimentClassification.Predict("Не нравится, что мод не выполняет свою работу.");
                SentimentClassification.Predict("модель чего Квадрокоптера?");
                SentimentClassification.Predict("США станут модом на обливион?");
                SentimentClassification.Predict("Понятно, что пиздато, интересно, лучше, чем 99 ютуба, но как-то тяжко дается смотреть");
                SentimentClassification.Predict("видео не выложили, чтобы не палить причины произошедшего Ясно.");
                SentimentClassification.Predict("Вообще то все в Польше собирают клубнику по безвизу.");
                SentimentClassification.Predict("Во все тяжки Последний сезон говнина полная, даже досматривать не стал.");
                SentimentClassification.Predict("чем больше компания тем больше бардак");
                SentimentClassification.Predict("а это даже не 100 км, а 20-50 км быстрофикс радиус со стартовой площадки");
                SentimentClassification.Predict("Все нормальные сайты уже перенесли сервера в Россию, а кто не успел - его проблемы.");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
        }
    }
}
