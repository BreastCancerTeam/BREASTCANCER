import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.ndimage import sobel
from sklearn.metrics import jaccard_score, precision_score, recall_score
import skfuzzy as fuzz
from skimage import filters

# Función para cargar la imagen DICOM y convertirla a un formato usable
def load_dicom_image(dicom_path):
    try:
        dicom_image = pydicom.dcmread(dicom_path)
        pixel_array = dicom_image.pixel_array
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"Error cargando la imagen DICOM en {dicom_path}: {e}")
        return None

# Función para aplicar la transformada wavelet direccional (DWT) y extraer textura
def directional_texture_extraction(image, wavelet='db2', level=2):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    cA = coeffs[0]
    sobel_horizontal = sobel(cA, axis=0)
    sobel_vertical = sobel(cA, axis=1)
    directional_texture = np.hypot(sobel_horizontal, sobel_vertical)
    return directional_texture

# Visualización de comparación de bordes con cierre automático
def visualize_comparison(original_image, edges_image):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Imagen Original')
    plt.subplot(1, 2, 2)
    plt.imshow(edges_image, cmap='gray')
    plt.title('Bordes Detectados')
    plt.show(block=False)
    plt.pause(1)  # Pausa de 1 segundo
    plt.close()

# Visualización de subregiones segmentadas con cierre automático
def visualize_subregions(original_image, segmented_image):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Imagen Original')
    plt.subplot(2, 4, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Imagen Segmentada')
    for i in range(1, 6):
        plt.subplot(2, 4, i + 2)
        subregion = np.where(segmented_image == (i - 1), 255, 0).astype(np.uint8)
        plt.imshow(subregion, cmap='gray')
        plt.title(f'Subregión {i}')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

# Aplicar KMeans clustering para segmentar en subregiones
def apply_kmeans(image, n_clusters=5):
    pixels = image.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    clustered = kmeans.labels_.reshape(image.shape)
    return clustered

# Aplicar Fuzzy C-Means clustering para segmentar en subregiones
def apply_fuzzy_c_means(image, n_clusters=5):
    pixels = image.flatten().reshape(1, -1)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        pixels, n_clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    clustered = cluster_membership.reshape(image.shape)
    return clustered

# Aplicar detección de bordes con Canny
def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# Clasificar BIRADS basado en la textura
def classify_birads_by_texture(texture_image):
    mean_intensity = np.mean(texture_image)
    if mean_intensity < 50:
        return 1  # BIRADS 1
    elif mean_intensity < 100:
        return 2  # BIRADS 2
    elif mean_intensity < 150:
        return 3  # BIRADS 3
    elif mean_intensity < 200:
        return 4  # BIRADS 4
    else:
        return 5  # BIRADS 5

# Procesar imágenes y aplicar análisis completo
def process_images(csv_file_path, image_directory, ground_truth_directory=None, clustering_method='kmeans'):
    print("Iniciando el procesamiento de imágenes...")
    if not os.path.exists(csv_file_path):
        print(f"El archivo CSV en {csv_file_path} no existe.")
        return
    if not os.path.exists(image_directory):
        print(f"El directorio de imágenes en {image_directory} no existe.")
        return

    clinical_data = pd.read_csv(csv_file_path)
    if clinical_data.empty:
        print("El archivo CSV está vacío.")
        return

    birads_classifications = []
    
    # Crear directorios de salida si no existen
    if not os.path.exists('output/benign'):
        os.makedirs('output/benign/left')
        os.makedirs('output/benign/right')
    if not os.path.exists('output/malignant'):
        os.makedirs('output/malignant/left')
        os.makedirs('output/malignant/right')
    
    for index, row in clinical_data.iterrows():
        image_filename = row['ID1']
        diagnosis = row['classification'].lower()
        side = 'left' if row['LeftRight'] == 'L' else 'right'
        
        dicom_path = os.path.join(image_directory, f"{image_filename}.dcm")
        if not os.path.exists(dicom_path):
            print(f"No se encontró la imagen DICOM en {dicom_path}.")
            birads_classifications.append(None)
            continue
        
        original_image = load_dicom_image(dicom_path)
        if original_image is None:
            birads_classifications.append(None)
            continue

        # Segmentación y detección de bordes
        segmented_image = apply_kmeans(original_image, n_clusters=5) if clustering_method == 'kmeans' else apply_fuzzy_c_means(original_image, n_clusters=5)
        edges_image = apply_canny_edge_detection(original_image, low_threshold=100, high_threshold=200)

        # Visualización
        visualize_comparison(original_image, edges_image)
        visualize_subregions(original_image, segmented_image)

        # Clasificación BIRADS
        texture_image = directional_texture_extraction(original_image)
        birads_classification = classify_birads_by_texture(texture_image)
        birads_classifications.append(birads_classification)

        # Guardar las imágenes procesadas
        output_folder = f'output/{diagnosis}/{side}'
        output_path_segmented = os.path.join(output_folder, f"{image_filename}_segmented.png")
        output_path_edges = os.path.join(output_folder, f"{image_filename}_edges.png")
        
        plt.imsave(output_path_segmented, segmented_image, cmap='gray')
        plt.imsave(output_path_edges, edges_image, cmap='gray')
        
        print(f"Imagen {image_filename} procesada y guardada en {output_folder}")

    # Añadir la columna BIRADS al archivo CSV y guardarlo
    clinical_data['BIRADS_Classification'] = birads_classifications
    output_csv_path = os.path.join(image_directory, 'clinical_data_with_birads.csv')
    clinical_data.to_csv(output_csv_path, index=False)
    print(f"Clasificaciones BIRADS guardadas en {output_csv_path}")

# Ruta del archivo CSV y el directorio de las imágenes DICOM
csv_file_path = 'D:/MastoAI/Clinical_data.csv'
image_directory = 'D:/MastoAI/'
ground_truth_directory = 'D:/MastoAI/ground_truth/'

# Procesar las imágenes y generar la clasificación BIRADS
process_images(csv_file_path, image_directory, ground_truth_directory, clustering_method='kmeans')

#######################################################33
#Generacion de graficos

import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from scipy.ndimage import sobel
from skimage import filters

# Ruta del archivo CSV generado con las clasificaciones BIRADS
csv_file_path = 'D:/MastoAI/clinical_data_with_birads.csv'

# Cargar el archivo CSV con las clasificaciones BIRADS
clinical_data = pd.read_csv(csv_file_path)

# Contar el número de casos por cada clasificación BIRADS
birads_counts = clinical_data['BIRADS_Classification'].value_counts().sort_index()

# Crear el gráfico de barras con colores diferentes para cada clasificación
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colores para BIRADS 1, 2, 3, 4

# Generar gráfico de barras
plt.figure(figsize=(8, 6))
birads_counts.plot(kind='bar', color=colors)

# Etiquetas y título
plt.xlabel('BIRADS Classification')
plt.ylabel('Number of Cases')
plt.title('Distribution of Cases by BIRADS Classification')

# Mostrar el gráfico
plt.show()


############################################################################################

import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.ndimage import sobel
from skimage.filters import gabor
from scipy.stats import entropy, skew, kurtosis

# Función para cargar la imagen DICOM
def load_dicom_image(dicom_path):
    try:
        dicom_image = pydicom.dcmread(dicom_path)
        pixel_array = dicom_image.pixel_array
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"Error cargando la imagen DICOM en {dicom_path}: {e}")
        return None

# Extracción de características de textura direccional
def extract_directional_features(image):
    texture_image = directional_texture_extraction(image)
    num_directional_tissues = np.sum(texture_image > filters.threshold_otsu(texture_image))
    avg_length = np.mean(texture_image[texture_image > filters.threshold_otsu(texture_image)])
    orientations = np.arctan2(sobel(image, axis=1), sobel(image, axis=0))
    angular_entropy = entropy(np.histogram(orientations, bins=16)[0])
    return {
        "num_directional_tissues": num_directional_tissues,
        "avg_length": avg_length,
        "angular_entropy": angular_entropy
    }

# Aplicar filtros Gabor y extraer características
def extract_gabor_features(image):
    frequencies = [0.1, 0.2, 0.3]
    features = {}
    for frequency in frequencies:
        filt_real, filt_imag = gabor(image, frequency=frequency)
        features[f"gabor_contrast_{frequency}"] = np.std(filt_real)
        features[f"gabor_entropy_{frequency}"] = entropy(np.histogram(filt_real, bins=16)[0])
    return features

# Cálculo manual de GLCM con normalización
def compute_glcm(image, distance=1, angle=0):
    # Normalizar imagen a rango 0-255 y redondear a enteros
    image = np.clip(image, 0, 255).astype(int)
    
    # Calcular el nivel máximo de gris para dimensionar la GLCM
    max_gray_level = image.max() + 1
    
    # Asegurar que max_gray_level no sea cero y crear matriz GLCM
    if max_gray_level == 0:
        raise ValueError("La imagen está vacía o tiene valores uniformemente cero.")

    glcm = np.zeros((max_gray_level, max_gray_level), dtype=int)
    offsets = {
        0: (0, distance),  # 0 grados
        np.pi / 4: (-distance, distance),  # 45 grados
        np.pi / 2: (-distance, 0),  # 90 grados
        3 * np.pi / 4: (-distance, -distance)  # 135 grados
    }
    
    dx, dy = offsets[angle]
    for y in range(image.shape[0] - abs(dy)):
        for x in range(image.shape[1] - abs(dx)):
            i = image[y, x]
            j = image[y + dy, x + dx]
            if i < max_gray_level and j < max_gray_level:
                glcm[i, j] += 1

    # Normalizar la GLCM para obtener probabilidades
    glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
    return glcm

def compute_glcm_features(glcm):
    """Función para calcular características a partir de una matriz GLCM."""
    contrast = np.sum([(i - j) ** 2 * glcm[i, j] for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])
    homogeneity = np.sum([glcm[i, j] / (1 + abs(i - j)) for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])
    energy = np.sum(glcm ** 2)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Evitar log(0) usando 1e-10
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "entropy": entropy
    }

def extract_glcm_features(image):
    # Calcular GLCM en varias direcciones y distancias
    glcm_0 = compute_glcm(image, distance=1, angle=0)
    glcm_45 = compute_glcm(image, distance=1, angle=np.pi / 4)
    glcm_90 = compute_glcm(image, distance=1, angle=np.pi / 2)
    glcm_135 = compute_glcm(image, distance=1, angle=3 * np.pi / 4)

    # Calcular características para cada GLCM
    features_0 = compute_glcm_features(glcm_0)
    features_45 = compute_glcm_features(glcm_45)
    features_90 = compute_glcm_features(glcm_90)
    features_135 = compute_glcm_features(glcm_135)

    # Promediar características entre todas las direcciones
    features = {
        "glcm_contrast": np.mean([features_0["contrast"], features_45["contrast"], features_90["contrast"], features_135["contrast"]]),
        "glcm_homogeneity": np.mean([features_0["homogeneity"], features_45["homogeneity"], features_90["homogeneity"], features_135["homogeneity"]]),
        "glcm_energy": np.mean([features_0["energy"], features_45["energy"], features_90["energy"], features_135["energy"]]),
        "glcm_entropy": np.mean([features_0["entropy"], features_45["entropy"], features_90["entropy"], features_135["entropy"]])
    }
    return features

# Características de intensidad y estadísticas
def extract_intensity_features(image):
    hist, bins = np.histogram(image.flatten(), bins=256)
    return {
        "mean_intensity": np.mean(image),
        "std_dev_intensity": np.std(image),
        "skewness": skew(image.flatten()),
        "kurtosis": kurtosis(image.flatten()),
        "hist_entropy": entropy(hist),
        "smoothness": 1 - (1 / (1 + np.var(image))),
        "uniformity": np.sum(hist**2)
    }

# Extracción completa de características
def extract_features(image, side, diagnosis):
    directional_features = extract_directional_features(image)
    gabor_features = extract_gabor_features(image)
    glcm_features = extract_glcm_features(image)
    intensity_features = extract_intensity_features(image)

    # Combinar todas las características en un solo diccionario
    features = {**directional_features, **gabor_features, **glcm_features, **intensity_features}
    features["side"] = side
    features["diagnosis"] = diagnosis
    return features

# Procesar imágenes y almacenar características en CSV
def process_images_to_csv(csv_file_path, image_directory):
    print("Iniciando extracción de características de imágenes...")

    clinical_data = pd.read_csv(csv_file_path)
    all_features = []
    
    for index, row in clinical_data.iterrows():
        image_filename = row['ID1']
        diagnosis = row['classification'].lower()
        side = 'left' if row['LeftRight'] == 'L' else 'right'
        
        dicom_path = os.path.join(image_directory, f"{image_filename}.dcm")
        original_image = load_dicom_image(dicom_path)
        if original_image is None:
            continue

        # Extraer todas las características
        features = extract_features(original_image, side, diagnosis)
        features["ID"] = image_filename
        all_features.append(features)
        print(f"Características extraídas para {image_filename}")
    
    # Guardar en un archivo CSV
    features_df = pd.DataFrame(all_features)
    features_csv_path = os.path.join(image_directory, 'extracted_features.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"Archivo de características guardado en {features_csv_path}")

# Gráfico de barras de características importantes
def plot_features_comparison(selected_features_path, all_features_path):
    all_features_df = pd.read_csv(all_features_path)
    selected_features_df = pd.read_csv(selected_features_path)

    all_counts = all_features_df.count()
    selected_counts = selected_features_df.count()

    # Crear gráfico comparativo
    plt.figure(figsize=(14, 7))
    all_counts.plot(kind='bar', alpha=0.6, label="Características Extraídas")
    selected_counts.plot(kind='bar', alpha=0.6, color='orange', label="Características Seleccionadas")
    plt.xlabel("Características")
    plt.ylabel("Frecuencia")
    plt.title("Comparación de Características Extraídas vs. Seleccionadas")
    plt.legend()
    plt.show()

# Etapa de selección de características importantes
def select_important_features(features_csv_path):
    features_df = pd.read_csv(features_csv_path)
    important_features = features_df.iloc[:, :20]  # Selección simulada
    selected_features_path = features_csv_path.replace("extracted_features", "selected_features")
    important_features.to_csv(selected_features_path, index=False)
    print(f"Características seleccionadas guardadas en {selected_features_path}")
    return selected_features_path

# Rutas de archivo y directorio de imágenes
csv_file_path = 'D:/MastoAI/Clinical_data.csv'
image_directory = 'D:/MastoAI/'

# Ejecutar las etapas del proceso
process_images_to_csv(csv_file_path, image_directory)
selected_features_path = select_important_features('D:/MastoAI/extracted_features.csv')
plot_features_comparison(selected_features_path, 'D:/MastoAI/extracted_features.csv')



#########################################################################################
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo CSV que contiene las clasificaciones BIRADS y las etiquetas de verdad
csv_with_predictions_path = 'D:/MastoAI/clinical_data_with_birads.csv'

# Cargar el archivo con predicciones y etiquetas de verdad
data = pd.read_csv(csv_with_predictions_path)

# Mapear las etiquetas 'classification' a valores numéricos para comparar
label_mapping = {'Benign': 1, 'Malignant': 2}
data['classification_mapped'] = data['classification'].map(label_mapping)

# Definir predicciones y etiquetas de verdad (mapeado a numérico)
y_pred = data['BIRADS_Classification']
y_true = data['classification_mapped']

# Cálculo de métricas de rendimiento
jaccard = jaccard_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

# Mostrar resultados
print(f"Jaccard Score: {jaccard:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Accuracy: {accuracy:.3f}")

# Visualización de métricas de desempeño
metrics_data = {
    'Metric': ['Jaccard', 'Precision', 'Recall', 'Accuracy'],
    'Score': [jaccard, precision, recall, accuracy]
}
metrics_df = pd.DataFrame(metrics_data)

# Gráfico de barras de las métricas
plt.figure(figsize=(10, 6))  # Figura 1
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.ylim(0, 1)
plt.title("Modelo de Desempeño en Clasificación BIRADS")
plt.ylabel("Score")
plt.xlabel("Métricas")

# Visualización de la comparación entre predicciones y ground truth
pred_counts = y_pred.value_counts().sort_index()
true_counts = y_true.value_counts().sort_index()

# Crear un DataFrame para la comparación
comparison_df = pd.DataFrame({
    'Predicciones': pred_counts,
    'Ground Truth': true_counts
}).fillna(0)

# Gráfico de comparación sin generar figura extra
plt.figure(figsize=(10, 6))  # Figura 2
plt.bar(comparison_df.index - 0.2, comparison_df['Predicciones'], width=0.4, label="Predicciones", color='skyblue')
plt.bar(comparison_df.index + 0.2, comparison_df['Ground Truth'], width=0.4, label="Ground Truth", color='salmon')
plt.title("Comparación de Clasificación BIRADS: Predicciones vs. Ground Truth")
plt.xlabel("Clasificación BIRADS")
plt.ylabel("Número de Casos")
plt.xticks(comparison_df.index)
plt.legend()

# Mostrar ambas figuras sin crear una figura adicional vacía
plt.show()

#######################################################
##################################################################################################

import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.ndimage import sobel
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import skfuzzy as fuzz
from skimage import filters
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.stats import entropy, skew, kurtosis
from skimage.filters import gabor

# Función para cargar la imagen DICOM y convertirla a un formato usable
def load_dicom_image(dicom_path):
    try:
        dicom_image = pydicom.dcmread(dicom_path)
        pixel_array = dicom_image.pixel_array
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
        return pixel_array.astype(np.uint8)
    except Exception as e:
        print(f"Error cargando la imagen DICOM en {dicom_path}: {e}")
        return None

# Función para dividir la imagen en cinco subregiones
def divide_into_subregions(image):
    h, w = image.shape
    subregion_height = h // 2
    subregion_width = w // 2
    subregions = [
        image[:subregion_height, :subregion_width],  # Arriba a la izquierda
        image[:subregion_height, subregion_width:],  # Arriba a la derecha
        image[subregion_height:, :subregion_width],  # Abajo a la izquierda
        image[subregion_height:, subregion_width:],  # Abajo a la derecha
        image[subregion_height//2:3*subregion_height//2, subregion_width//2:3*subregion_width//2]  # Centro
    ]
    return subregions

# Modificar la función de extracción de características para calcularlas en cada subregión
def extract_features_for_subregions(image, side, diagnosis):
    subregions = divide_into_subregions(image)
    all_features = []
    for i, subregion in enumerate(subregions):
        directional_features = extract_directional_features(subregion)
        gabor_features = extract_gabor_features(subregion)
        glcm_features = extract_glcm_features(subregion)
        intensity_features = extract_intensity_features(subregion)
        subregion_features = {f"{key}_subregion_{i+1}": val for key, val in {
            **directional_features, **gabor_features, **glcm_features, **intensity_features
        }.items()}
        all_features.append(subregion_features)
    combined_features = {key: val for d in all_features for key, val in d.items()}
    combined_features["side"] = side
    combined_features["diagnosis"] = diagnosis
    return combined_features

# Función para aplicar la transformada wavelet direccional (DWT) y extraer textura
def directional_texture_extraction(image, wavelet='db2', level=2):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    cA = coeffs[0]
    sobel_horizontal = sobel(cA, axis=0)
    sobel_vertical = sobel(cA, axis=1)
    directional_texture = np.hypot(sobel_horizontal, sobel_vertical)
    return directional_texture

# Funciones para extraer características
def extract_directional_features(image):
    texture_image = directional_texture_extraction(image)
    num_directional_tissues = np.sum(texture_image > filters.threshold_otsu(texture_image))
    avg_length = np.mean(texture_image[texture_image > filters.threshold_otsu(texture_image)])
    orientations = np.arctan2(sobel(image, axis=1), sobel(image, axis=0))
    angular_entropy = entropy(np.histogram(orientations, bins=16)[0])
    return {"num_directional_tissues": num_directional_tissues, "avg_length": avg_length, "angular_entropy": angular_entropy}

def extract_gabor_features(image):
    frequencies = [0.1, 0.2, 0.3]
    features = {}
    for frequency in frequencies:
        filt_real, filt_imag = gabor(image, frequency=frequency)
        features[f"gabor_contrast_{frequency}"] = np.std(filt_real)
        features[f"gabor_entropy_{frequency}"] = entropy(np.histogram(filt_real, bins=16)[0])
    return features

def compute_glcm(image, distance=1, angle=0):
    image = np.clip(image, 0, 255).astype(int)
    max_gray_level = image.max() + 1
    if max_gray_level == 0:
        raise ValueError("La imagen está vacía o tiene valores uniformemente cero.")
    glcm = np.zeros((max_gray_level, max_gray_level), dtype=int)
    offsets = {0: (0, distance), np.pi / 4: (-distance, distance), np.pi / 2: (-distance, 0), 3 * np.pi / 4: (-distance, -distance)}
    dx, dy = offsets[angle]
    for y in range(image.shape[0] - abs(dy)):
        for x in range(image.shape[1] - abs(dx)):
            i = image[y, x]
            j = image[y + dy, x + dx]
            if i < max_gray_level and j < max_gray_level:
                glcm[i, j] += 1
    glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
    return glcm

def compute_glcm_features(glcm):
    contrast = np.sum([(i - j) ** 2 * glcm[i, j] for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])
    homogeneity = np.sum([glcm[i, j] / (1 + abs(i - j)) for i in range(glcm.shape[0]) for j in range(glcm.shape[1])])
    energy = np.sum(glcm ** 2)
    entropy_value = -np.sum(glcm * np.log2(glcm + 1e-10))
    return {"contrast": contrast, "homogeneity": homogeneity, "energy": energy, "entropy": entropy_value}

def extract_glcm_features(image):
    glcm_0 = compute_glcm(image, distance=1, angle=0)
    glcm_45 = compute_glcm(image, distance=1, angle=np.pi / 4)
    glcm_90 = compute_glcm(image, distance=1, angle=np.pi / 2)
    glcm_135 = compute_glcm(image, distance=1, angle=3 * np.pi / 4)
    features_0 = compute_glcm_features(glcm_0)
    features_45 = compute_glcm_features(glcm_45)
    features_90 = compute_glcm_features(glcm_90)
    features_135 = compute_glcm_features(glcm_135)
    features = {
        "glcm_contrast": np.mean([features_0["contrast"], features_45["contrast"], features_90["contrast"], features_135["contrast"]]),
        "glcm_homogeneity": np.mean([features_0["homogeneity"], features_45["homogeneity"], features_90["homogeneity"], features_135["homogeneity"]]),
        "glcm_energy": np.mean([features_0["energy"], features_45["energy"], features_90["energy"], features_135["energy"]]),
        "glcm_entropy": np.mean([features_0["entropy"], features_45["entropy"], features_90["entropy"], features_135["entropy"]])
    }
    return features

def extract_intensity_features(image):
    hist, bins = np.histogram(image.flatten(), bins=256)
    return {
        "mean_intensity": np.mean(image),
        "std_dev_intensity": np.std(image),
        "skewness": skew(image.flatten()),
        "kurtosis": kurtosis(image.flatten()),
        "hist_entropy": entropy(hist),
        "smoothness": 1 - (1 / (1 + np.var(image))),
        "uniformity": np.sum(hist**2)
    }

# Procesar imágenes y almacenar características en CSV
def process_images_to_csv(csv_file_path, image_directory):
    print("Iniciando extracción de características de imágenes...")
    clinical_data = pd.read_csv(csv_file_path)
    all_features = []
    for index, row in clinical_data.iterrows():
        image_filename = row['ID1']
        diagnosis = row['classification'].lower()
        side = 'left' if row['LeftRight'] == 'L' else 'right'
        dicom_path = os.path.join(image_directory, f"{image_filename}.dcm")
        original_image = load_dicom_image(dicom_path)
        if original_image is None:
            continue
        features = extract_features_for_subregions(original_image, side, diagnosis)
        features["ID"] = image_filename
        all_features.append(features)
        print(f"Características extraídas para {image_filename}")
    features_df = pd.DataFrame(all_features)
    features_csv_path = os.path.join(image_directory, 'extracted_features.csv')
    features_df.to_csv(features_csv_path, index=False)
    print(f"Archivo de características guardado en {features_csv_path}")

#########################################################################################################
#Visualizador

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pydicom
import cv2
import numpy as np

# Función para cargar la imagen DICOM y convertirla a un formato usable
def load_dicom_image(dicom_path):
    dicom_image = pydicom.dcmread(dicom_path)
    pixel_array = dicom_image.pixel_array
    pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    return pixel_array.astype(np.uint8)

# Función para actualizar la visualización con una nueva imagen y parámetros
def update_viewer(dicom_path, parameters):
    # Cargar la imagen DICOM
    image = load_dicom_image(dicom_path)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    # Actualizar el panel de parámetros
    param_text.set("\n".join([f"{k}: {v}" for k, v in parameters.items()]))

    # Refrescar el canvas
    canvas.draw()

# Función para seleccionar una imagen DICOM y cargarla con los parámetros
def select_image():
    # Seleccionar archivo DICOM
    dicom_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
    if dicom_path:
        # Simular parámetros para la demostración (esto debería reemplazarse con valores reales)
        parameters = {
            "BIRADS": "3",
            "Mean Intensity": "120.5",
            "Contrast": "0.75",
            "Entropy": "2.1",
            "Diagnosis": "Benign"
        }
        # Actualizar visualización
        update_viewer(dicom_path, parameters)

# Crear la ventana principal de la interfaz gráfica
root = tk.Tk()
root.title("Visualizador de Mamografía")

# Configuración de colores
bg_color = "#FFC0CB"  # Rosado para la interfaz

# Configurar el frame izquierdo para la imagen
frame_left = tk.Frame(root, bg=bg_color)
frame_left.pack(side="left", fill="both", expand=True)

# Crear un contenedor de Matplotlib para la imagen
fig = plt.Figure(figsize=(5, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, frame_left)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Configurar el frame derecho para los parámetros
frame_right = tk.Frame(root, bg=bg_color, width=200)
frame_right.pack(side="right", fill="both")

# Título de parámetros
title_label = tk.Label(frame_right, text="Parámetros Evaluados", bg=bg_color, font=("Helvetica", 14, "bold"))
title_label.pack(pady=10)

# Texto para mostrar parámetros
param_text = tk.StringVar()
param_label = tk.Label(frame_right, textvariable=param_text, bg=bg_color, font=("Helvetica", 12), justify="left")
param_label.pack(pady=10, padx=10)

# Botón para cargar una nueva imagen
load_button = tk.Button(frame_right, text="Cargar Imagen DICOM", command=select_image, bg="white", font=("Helvetica", 12))
load_button.pack(pady=20)

# Iniciar la aplicación
root.mainloop()
