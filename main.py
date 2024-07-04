import time
from ml_models import DataAnalyzer
from PIL import Image


analyzer = DataAnalyzer('/data/stock_move.csv')

data = analyzer.pre_processing()

start_time = time.time()
filter_products = analyzer.model_hyperparam_tuning(run=False)
end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")

prediction = analyzer.prediction(61578, 4)

plot_test = analyzer.plot_test(61578)

plot_prediction = analyzer.plot_months_to_predict(61578)
