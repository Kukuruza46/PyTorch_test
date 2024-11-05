import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from pydantic import BaseModel

# Данные для обучения (простые примеры)
data = [
    ("продукт отличный", 1),
    ("очень плохое качество", 0),
    ("мне все понравилось", 1),
    ("ужасный сервис", 0),
    ("супер, рекомендую", 1),
    ("не советую покупать", 0),
    ("отличное качество", 1),
    ("очень разочарован", 0),
    ("ужасный опыт", 0),
    ("все супер", 1),
    ("плохой продукт", 0),
    ("советую", 1),
    ("не понравилось", 0),
    ("отвратительное обслуживание", 0),
    ("очень рекомендую", 1),
    ("ужасный запах", 0),
    ("прекрасный товар", 1)
]

# Словарь для преобразования слов в числа
word_to_idx = {
    "продукт": 0,
    "отличный": 1,
    "очень": 2,
    "плохое": 3,
    "качество": 4,
    "мне": 5,
    "все": 6,
    "понравилось": 7,
    "ужасный": 8,
    "сервис": 9,
    "супер": 10,
    "рекомендую": 11,
    "не": 12,
    "советую": 13,
    "покупать": 14,
}

# Преобразуем данные в числовую форму и добавляем padding
def prepare_data(data, max_len):
    inputs = []
    labels = []
    for sentence, label in data:
        word_indices = [word_to_idx[word] for word in sentence.split() if word in word_to_idx]
        # Добавляем padding (дополняем нулями до max_len)
        while len(word_indices) < max_len:
            word_indices.append(0)
        inputs.append(word_indices[:max_len])  # Ограничиваем длину max_len
        labels.append(label)
    return inputs, labels

# Определяем максимальную длину последовательности
max_len = 5  # Длина, до которой будут дополнены все предложения
x_train, y_train = prepare_data(data, max_len)

# Преобразуем данные в тензоры
x_train = torch.tensor(x_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Модель для классификации текста
class ImprovedTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(ImprovedTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 50, batch_first=True)  # Добавляем LSTM слой
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]  # Берем последнее состояние LSTM
        output = self.fc(lstm_out)
        return torch.sigmoid(output)  # Используем сигмоиду для получения вероятности

# Параметры модели
vocab_size = len(word_to_idx)  # Количество уникальных слов
embed_dim = 5  # Размерность встраивания

# Инициализация модели, функции потерь и оптимизатора
model = ImprovedTextClassifier(vocab_size, embed_dim)
criterion = nn.BCELoss()  # Функция потерь для бинарной классификации
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Включаем режим обучения
    optimizer.zero_grad()  # Обнуление градиентов

    # Прямой проход (предсказание)
    outputs = model(x_train).squeeze()

    # Вычисление функции потерь
    loss = criterion(outputs, y_train)
    
    # Обратное распространение ошибки
    loss.backward()
    
    # Шаг оптимизации
    optimizer.step()

# Создание FastAPI приложения
app = FastAPI()

# Определение модели данных, которую мы будем принимать через API
class TextInput(BaseModel):
    text: str

# Маршрут для предсказания тональности текста
@app.post("/predict/")
def predict_sentiment(input_text: TextInput):
    model.eval()  # Включаем режим оценки (без обучения)
    words = input_text.text.split()
    word_indices = [word_to_idx[word] for word in words if word in word_to_idx]

    # Добавляем padding, если длина меньше max_len
    while len(word_indices) < max_len:
        word_indices.append(0)
    word_indices = word_indices[:max_len]

    if not word_indices:
        return {"sentiment": "Неизвестный текст", "confidence": 0.0}

    input_tensor = torch.tensor([word_indices], dtype=torch.long)
    output = model(input_tensor).item()
    sentiment = "Положительный" if output > 0.5 else "Отрицательный"
    confidence = round(output, 4)
    return {"sentiment": sentiment, "confidence": confidence}
