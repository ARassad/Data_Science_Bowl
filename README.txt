В папке с исходными кодами должна располагаться папка data с входыми данными, она должна содержать папки stage1_test и stage1_train с данными

data_preparation.py : формирует входные данные для model_preparation и prediction
model_preparation.py : создание и тренировка модели, сохраняет лучший вариант модели в файл model-dsbowl2018-1.h5
prediction.py : формирует маски на тестовых данных, результат в файле sub-dsbowl2018-1.csv
function.py : функции метрики и кодирования
