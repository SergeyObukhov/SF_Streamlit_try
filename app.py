#Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings("ignore")


import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px

@st.cache_data
def read_files(folder_name='data'):
    """
    Функция для чтения файлов.
    Возвращает два DataFrame с рейтингами и характеристиками книг.
    """
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    return ratings, books


def make_mappers(books):
    """
    Функция для создания отображения id в title и authors.
    Возвращает два словаря:
    * Ключи первого словаря — идентификаторы книг, значения — их названия.
    * Ключи второго словаря — идентификаторы книг, значения — их авторы.
    """
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))

    return name_mapper, author_mapper

def load_embeddings(file_name='item_embeddings.pkl'):
    """
    Функция для загрузки векторных представлений.
    Возвращает прочитанные эмбеддинги книг и индекс (граф) для поиска похожих книг.
    """
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx

def nearest_books_nms(book_id, index, n=10):
    """
    Функция для поиска ближайших соседей, возвращает построенный индекс.
    Возвращает n наиболее похожих книг и расстояние до них.
    """
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn

def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    """
    Функция для составления таблицы из рекомендованных книг.
    Возвращает DataFrame со столбцами:
    * book_name — название книги;
    * book_author — автор книги;
    * distance — значение метрики расстояния до книги.
    """
    names = []
    authors = []
    #Для каждого индекса книги находим её название и автора
    #Результаты добавляем в списки
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
    #Составляем DataFrame
    recomendation_df = pd.DataFrame({'book_name': names, 'book_author': authors, 'distance': distances})
    return recomendation_df

#Загружаем данные
ratings, books = read_files(folder_name='data')
#Создаём словари для сопоставления id книг и их названий/авторов
name_mapper, author_mapper = make_mappers(books)
#Загружаем эмбеддинги и создаём индекс для поиска
item_embeddings, nms_idx = load_embeddings()


# добавим заголовок нашего проекта
st.title("Рекомендательная система книг")

# добавим описание проекта
st.markdown("""Добро пожаловать на веб-страницу рекомендаций книг!
Это приложение является прототипом рекомендательной системы, основанной на машинном обучении.

Чтобы использовать приложение, вам нужно:
1. Введите примерное название понравившейся книги на английском языке
2. Выберите его точное название во всплывающем списке книг
3. Укажите количество книг, которые вам нужно порекомендовать

После этого приложение выдаст вам список книг, наиболее похожих на указанную вами книгу.""")

# Вводим строку для поиска книг
title = st.text_input('Пожалуйста, введите название книги на английском языке', '')
title = title.lower()

#Выполняем поиск по книгам — ищем неполные совпадения
output = books[books['title'].apply(lambda x: x.lower().find(title)) >= 0]

#Выбор книги из списка
option = st.selectbox("Выберите нужную книгу", output['title'].values)

#Проверяем, что поле не пустое
if option:
    #Выводим выбранную книгу
    st.markdown('Вы выбрали: "{}"'.format(option))

    #Находим book_id для указанной книги
    val_index = output[output['title'].values == option]['book_id'].values

    #Указываем количество рекомендаций
    count_recomendation = st.number_input(
        label="Укажите необходимое количество рекомендаций",
        value=10
    )

    #Находим count_recomendation+1 наиболее похожих книг
    ids, distances = nearest_books_nms(val_index, nms_idx, count_recomendation+1)
    #Убираем из результатов книгу, по которой производился поиск
    ids, distances = ids[1:], distances[1:]

    #Выводим рекомендации к книге
    st.markdown('Наиболее похожие книги: ')
    #Составляем DataFrame из рекомендаций
    df = get_recomendation_df(ids, distances, name_mapper, author_mapper)
    #Выводим DataFrame в интерфейсе
    st.dataframe(df[['book_name', 'book_author']])

    # Строим столбчатую диаграмму
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        hover_data=['book_author'],
        title='Уровень схожести'
    )
    # Отображаем график в интерфейсе
    st.write(fig)

# запуск вручную
# streamlit run app.py


# Создайте пустой репозиторий проекта на GitHub. Он может быть общедоступным или частным — это не имеет значения
# Создайте файл с зависимостями
# pip freeze > requirements.txt
# Инициализируйте локальный репозиторий, добавьте все файлы в список отслеживаемых и сделайте коммит
# git init
# git add .
# git commit -m 'Initial commit'
# Свяжите локальный репозиторий с удалённым и сделайте push проекта на GitHub
# git branch -M master
# git remote add origin https://github.com/SergeyObukhov/SF_Streamlit_try
# git push -u origin master