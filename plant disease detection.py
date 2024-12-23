#!/usr/bin/env python
# coding: utf-8

# In[1]:


zip_path = 'Dataset.zip'


# In[2]:


import zipfile
import os

# Define where to extract the dataset
extract_dir = 'Dataset/'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dataset extracted successfully!")


# In[3]:


# List all files and directories in the extracted folder
for root, dirs, files in os.walk(extract_dir):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print("=================================")


# In[4]:


train_files_healthy = "Dataset/Train/Healthy"
train_files_powdery = "Dataset/Train/Powdery"
train_files_rust = "Dataset/Train/Rust"


# In[5]:


# List contents of the Train folder to confirm the structure
train_folder = 'Dataset/Train/'
for root, dirs, files in os.walk(train_folder):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print("=================================")


# In[6]:


import os
import zipfile

# Define a function to count the number of files in a directory
def total_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

# Define where to extract the dataset
extract_dir = 'Dataset/'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Dataset extracted successfully!")

# List all files and directories in the extracted folder
for root, dirs, files in os.walk(extract_dir):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print("=================================")

# Update the paths to the training folders
train_files_healthy = "Dataset/Train/Healthy"
train_files_powdery = "Dataset/Train/Powdery"
train_files_rust = "Dataset/Train/Rust"

# List contents of the Train folder to confirm the structure
train_folder = 'Dataset/Train/'
for root, dirs, files in os.walk(train_folder):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print("=================================")

# Print the number of images in each category
print("Number of healthy leaf images in training set:", total_files(train_files_healthy))
print("Number of powdery leaf images in training set:", total_files(train_files_powdery))
print("Number of rusty leaf images in training set:", total_files(train_files_rust))


# In[7]:


train_files_healthy = "Dataset/Train/Train/Healthy"
train_files_powdery = "Dataset/Train/Train/Powdery"
train_files_rust = "Dataset/Train/Train/Rust"


# In[8]:


print("Number of healthy leaf images in training set:", total_files(train_files_healthy))
print("Number of powdery leaf images in training set:", total_files(train_files_powdery))
print("Number of rusty leaf images in training set:", total_files(train_files_rust))


# In[9]:


import os

def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files

train_files_healthy = "Dataset/Train/Train/Healthy"
train_files_powdery = "Dataset/Train/Train/Powdery"
train_files_rust = "Dataset/Train/Train/Rust"

test_files_healthy = "Dataset/Test/Test/Healthy"
test_files_powdery = "Dataset/Test/Test/Powdery"
test_files_rust = "Dataset/Test/Test/Rust"

valid_files_healthy = "Dataset/Validation/Validation/Healthy"
valid_files_powdery = "Dataset/Validation/Validation/Powdery"
valid_files_rust = "Dataset/Validation/Validation/Rust"

print("Number of healthy leaf images in training set", total_files(train_files_healthy))
print("Number of powder leaf images in training set", total_files(train_files_powdery))
print("Number of rusty leaf images in training set", total_files(train_files_rust))

print("========================================================")

print("Number of healthy leaf images in test set", total_files(test_files_healthy))
print("Number of powder leaf images in test set", total_files(test_files_powdery))
print("Number of rusty leaf images in test set", total_files(test_files_rust))

print("========================================================")

print("Number of healthy leaf images in validation set", total_files(valid_files_healthy))
print("Number of powder leaf images in validation set", total_files(valid_files_powdery))
print("Number of rusty leaf images in validation set", total_files(valid_files_rust))


# In[10]:


from PIL import Image
import IPython.display as display

image_path = 'Dataset/Train/Train/Healthy/8ce77048e12f3dd4.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))


# In[11]:


image_path = 'Dataset/Train/Train/Rust/80f09587dfc7988e.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))


# In[12]:


get_ipython().system('pip install tensorflow')


# In[13]:


pip install --upgrade numpy pandas tensorflow


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True
)

# Data preprocessing for test/validation set
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading training data
train_generator = train_datagen.flow_from_directory(
    'Dataset/Train/Train',
    target_size=(225, 225),
    batch_size=32,
    class_mode='categorical'
)

# Loading validation data
validation_generator = test_datagen.flow_from_directory(
    'Dataset/Validation/Validation',
    target_size=(225, 225),
    batch_size=32,
    class_mode='categorical'
)


# In[15]:


pip install --upgrade tensorflow keras


# In[16]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:


history = model.fit(train_generator,
                    batch_size=16,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_batch_size=16
                    )


# In[19]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
sns.set_theme()
sns.set_context("poster")

figure(figsize=(25, 25), dpi=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[20]:


model.save("model.h5")


# In[21]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

x = preprocess_image('Dataset/Test/Test/Rust/82f49a4a7b9585f1.jpg')
predictions = model.predict(x)
predictions[0]


# In[22]:


labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
labels


# In[23]:


predicted_label = labels[np.argmax(predictions)]
print(predicted_label)


# In[24]:


model.save("model.h5")


# In[25]:


from keras.models import load_model

# Load the trained model
model = load_model("model.h5")


# In[26]:


pip install --upgrade ipywidgets


# In[27]:


pip install --upgrade pillow numpy ipywidgets keras


# In[28]:


def on_upload_change(change):
    uploaded_files = upload_widget.value
    print(f"Uploaded files: {uploaded_files}")  # Debugging output
    if uploaded_files:  # Check if there is an uploaded file
        for file_name in uploaded_files:
            print(f"Processing file: {file_name}")  # Debugging output
            file_info = uploaded_files[file_name]
            img_data = file_info['content']  # Check if this line works correctly
            # ...


# In[29]:


import io
import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
from keras.preprocessing.image import img_to_array

# Function to preprocess image and predict using the model
def preprocess_image(image, target_size=(225, 225)):
    img = image.resize(target_size)  # Resize the image
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    return x

# Function that handles file uploads
def on_upload_change(change):
    uploaded_files = upload_widget.value
    print(f"Uploaded files: {uploaded_files}")  # Debugging output

    if uploaded_files:  # Check if there is an uploaded file
        # Unpack the uploaded files from the tuple
        file_info = list(uploaded_files)[0]  # Get the first (and only) file info
        print(f"Processing file: {file_info}")  # Debugging output
        
        img_data = file_info['content']  # Access the 'content' of the uploaded file
        
        # Open the image from the bytes content
        img = Image.open(io.BytesIO(img_data))
        
        # Display the uploaded image
        display(img)
        
        # Preprocess the image for the model
        x = preprocess_image(img)
        
        # Make predictions
        predictions = model.predict(x)
        
        # Mapping indices back to labels
        labels = train_generator.class_indices
        labels = {v: k for k, v in labels.items()}
        
        predicted_label = labels[np.argmax(predictions)]
        print(f"Predicted label: {predicted_label}")

# Create an upload widget
upload_widget = widgets.FileUpload(
    accept='image/*',  # Accept image files
    multiple=False      # Disable multiple uploads
)

# Display the upload widget
display(upload_widget)

# Link the widget with the on_upload_change function
upload_widget.observe(on_upload_change, names='value')


# In[30]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming you have true labels and predictions
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[31]:


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# In[32]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Get predictions probabilities for positive class (assuming multi-class)
y_scores = model.predict(validation_generator)

# Assuming you're interested in the class '0' (Healthy)
y_true = validation_generator.classes
average_precision = average_precision_score(y_true, y_scores, average='macro')

precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 0], pos_label=0)

plt.figure(figsize=(12, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.fill_between(recall, precision, alpha=0.1, color='blue')
plt.show()


# In[36]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Assuming y_true is your true labels
# Example: y_true = np.array([0, 1, 2, 1, 0])  # Your actual data

# One-Hot Encode y_true if it is not already done
# Check the version of scikit-learn and adjust accordingly
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output for sklearn >= 0.24
y_true_one_hot = encoder.fit_transform(y_true.reshape(-1, 1))

# Assuming y_scores contains the predicted probabilities for each class
# Example: y_scores = model.predict_proba(X)  # Your actual model predictions

n_classes = y_true_one_hot.shape[1]  # Update n_classes based on the one-hot encoded y_true
fpr = {}
tpr = {}
roc_auc = {}

# Calculate ROC curve for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_scores[:, i])
    roc_auc[i] = roc_auc_score(y_true_one_hot[:, i], y_scores[:, i])

plt.figure(figsize=(12, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC Curve (class {0} - AUC = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# Sample Data Generation
np.random.seed(42)
n_samples = 100
n_classes = 3

# Simulated true labels
y_true = np.random.randint(0, n_classes, n_samples)

# Simulated predicted probabilities
y_scores = np.random.rand(n_samples, n_classes)

# DataFrame for better handling
df = pd.DataFrame(y_scores, columns=[f'Class {i}' for i in range(n_classes)])
df['True Label'] = y_true

# Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())

# 1. Bar Chart for Class Probabilities
plt.figure(figsize=(12, 6))
df.iloc[:, :-1].mean().plot(kind='bar', color='skyblue')
plt.title('Average Predicted Probabilities per Class')
plt.ylabel('Average Probability')
plt.xlabel('Classes')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.savefig('average_probabilities.png')
plt.show()

# 2. ROC Curve for Each Class
plt.figure(figsize=(12, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_scores[:, i])
    roc_auc = roc_auc_score((y_true == i).astype(int), y_scores[:, i])
    plt.plot(fpr, tpr, label=f'ROC Curve for Class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('roc_curves.png')
plt.show()

# 3. Confusion Matrix
cm = confusion_matrix(y_true, np.argmax(y_scores, axis=1))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[f'Predicted {i}' for i in range(n_classes)],
            yticklabels=[f'True {i}' for i in range(n_classes)])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# 4. Classification Report
print("Classification Report:")
print(classification_report(y_true, np.argmax(y_scores, axis=1)))

# 5. Distribution of Scores
plt.figure(figsize=(12, 6))
for i in range(n_classes):
    sns.kdeplot(df[f'Class {i}'], label=f'Class {i}', fill=True)

plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('distribution_scores.png')
plt.show()


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample Data Generation
np.random.seed(42)
n_classes = 3
n_samples = 100

# Simulated true labels
y_true = np.random.randint(0, n_classes, n_samples)

# Simulated predicted probabilities
y_scores = np.random.rand(n_samples, n_classes)

# DataFrame for better handling
df = pd.DataFrame(y_scores, columns=[f'Class {i}' for i in range(n_classes)])
df['True Label'] = y_true

# Bar Chart for Average Predicted Probabilities
plt.figure(figsize=(12, 6))
df.iloc[:, :-1].mean().plot(kind='bar', color='skyblue')
plt.title('Average Predicted Probabilities per Class')
plt.ylabel('Average Probability')
plt.xlabel('Classes')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.savefig('average_probabilities.png')
plt.show()


# In[39]:


import numpy as np
from sklearn.metrics import classification_report

# Sample Data Generation
np.random.seed(42)
n_classes = 3
n_samples = 100

# Simulated true labels
y_true = np.random.randint(0, n_classes, n_samples)

# Simulated predicted probabilities
y_scores = np.random.rand(n_samples, n_classes)

# Get predicted classes
y_pred = np.argmax(y_scores, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred))


# In[40]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data Generation
np.random.seed(42)
n_classes = 3
n_samples = 100

# Simulated true labels
y_true = np.random.randint(0, n_classes, n_samples)

# Simulated predicted probabilities
y_scores = np.random.rand(n_samples, n_classes)

# DataFrame for better handling
df = pd.DataFrame(y_scores, columns=[f'Class {i}' for i in range(n_classes)])

# Distribution of Scores
plt.figure(figsize=(12, 6))
for i in range(n_classes):
    sns.kdeplot(df[f'Class {i}'], label=f'Class {i}', fill=True)

plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.savefig('distribution_scores.png')
plt.show()


# In[ ]:




