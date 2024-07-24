# functions.py

import setuptools
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, precision_recall_curve

# Intentamos importar matplotlib.pyplot con manejo de errores
try:
    import matplotlib.pyplot as plt
except TimeoutError as e:
    print(f"Error al importar matplotlib.pyplot: {e}")
    plt = None

def load_data(model_path, dict_path):
    """
    Carga el modelo y los resultados del pipeline desde los archivos especificados.
    """
    model = joblib.load(model_path)
    results_pipeline = joblib.load(dict_path)
    return model, results_pipeline

def mlflow_log_score(model_name, metric_name, metric_value):
    """
    Registra una métrica en MLflow bajo un run específico.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric(metric_name, metric_value)

def best_model(results_pipeline, remote_server_uri, exp_name):
    """
    Encuentra el mejor modelo y registra los resultados en MLFlow.
    """
    best_mean_result = 0
    best_std_result = 0
    best_model = None
    best_model_name = ""

    # Establecemos la URI del servidor remoto de MLflow
    mlflow.set_tracking_uri(remote_server_uri)
    print(f"La URL es: {remote_server_uri}")
    print(f"El experimento es: {exp_name}")

    # Establecemos el experimento activo por nombre
    mlflow.set_experiment(exp_name)

    names = results_pipeline['names']
    results = results_pipeline['results']
    models = results_pipeline['models']
    X_test = results_pipeline['X_test']
    y_test = results_pipeline['y_test']

    for i in range(len(results)):
        with mlflow.start_run(run_name=names[i]) as run:
            cv_results = results[i]
            name = names[i]
            model = models[i][1]

            mlflow.log_metric('mean_accuracy', np.mean(cv_results))
            mlflow.log_param('model', name)
            mlflow.log_param('accuracy', cv_results.tolist())
            # Registramos el modelo manualmente especificando los requisitos
            mlflow.sklearn.log_model(model, 'clf_model', conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.10.12',
                    'scikit-learn=1.2.2',
                    'pip',
                    {
                        'pip': [
                            'cloudpickle==3.0.0',
                            'mlflow'
                        ]
                    }
                ]
            })

            print(name + ": mean(accuracy)=" + str(round(np.mean(cv_results), 3)) + ", std(accuracy)=" + str(round(np.std(cv_results), 3)))

            # Predecir usando el modelo actual
            y_pred = model.predict(X_test)

            # Matriz de Confusión
            if plt:
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix for {name}')
                    plt.savefig('confusion_matrix.png')
                    plt.close()
                    mlflow.log_artifact('confusion_matrix.png')
                except Exception as e:
                    print(f"Error al guardar la matriz de confusión: {e}")

            # Verificar si el modelo tiene el método predict_proba
            if hasattr(model, "predict_proba") and plt:
                try:
                    # Curva de Precisión-Recall
                    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
                    plt.figure()
                    plt.plot(recall, precision, marker='.')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve for {name}')
                    plt.savefig('precision_recall_curve.png')
                    plt.close()
                    mlflow.log_artifact('precision_recall_curve.png')
                except Exception as e:
                    print(f"Error al guardar la curva de precisión-recall: {e}")
            else:
                print(f"El modelo {name} no tiene el método predict_proba o plt no está disponible.")

            # Calcular y registrar RMSE y MAE
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('mae', mae)

            if (best_mean_result < np.mean(cv_results)) or \
               ((best_mean_result == np.mean(cv_results)) and (best_std_result > np.std(cv_results))):
                best_mean_result = np.mean(cv_results)
                best_std_result = np.std(cv_results)
                best_model_name = name
                best_model = model

    return best_model, best_model_name
