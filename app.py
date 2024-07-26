# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.filechooser import FileChooserIconView
# from kivy.uix.label import Label
# from kivy.uix.button import Button
# from kivy.uix.image import Image
# from kivy.uix.textinput import TextInput
# from kivy.uix.scrollview import ScrollView
# from kivy.uix.gridlayout import GridLayout
# import cv2
# import numpy as np
# import pandas as pd
# from PIL import Image as PILImage
# import easyocr
# import os
# from inference_sdk import InferenceHTTPClient

# class HealthcareApp(App):

#     def build(self):
#         self.layout = BoxLayout(orientation='vertical')
#         self.filechooser = FileChooserIconView()
#         self.filechooser.bind(on_selection=self.load_image)
#         self.layout.add_widget(self.filechooser)
        
#         self.image_display = Image()
#         self.layout.add_widget(self.image_display)
        
#         self.result_label = Label(text="Upload an image to extract values.")
#         self.layout.add_widget(self.result_label)
        
#         self.classify_button = Button(text="Classify Image")
#         self.classify_button.bind(on_press=self.classify_image)
#         self.layout.add_widget(self.classify_button)
        
#         self.result_text = TextInput(size_hint_y=None, height=44, readonly=True)
#         self.layout.add_widget(self.result_text)
        
#         self.save_button = Button(text="Save Data")
#         self.save_button.bind(on_press=self.save_data)
#         self.layout.add_widget(self.save_button)
        
#         self.clear_button = Button(text="Clear Data")
#         self.clear_button.bind(on_press=self.clear_data)
#         self.layout.add_widget(self.clear_button)

#         return self.layout

#     def load_image(self, instance, value):
#         if value:
#             self.image_path = value[0]
#             pil_image = PILImage.open(self.image_path)
#             kivy_image = Image()
#             kivy_image.texture = self.pil_image_to_texture(pil_image)
#             self.image_display.texture = kivy_image.texture
#             self.result_label.text = "Image loaded. Press 'Classify Image' to analyze."

#     def pil_image_to_texture(self, pil_image):
#         pil_image = pil_image.convert('RGB')
#         data = np.array(pil_image)
#         texture = kivy.graphics.texture.Texture.create(size=(data.shape[1], data.shape[0]), colorfmt='rgb')
#         texture.blit_buffer(data.flatten(), colorfmt='rgb', bufferfmt='ubyte')
#         return texture

#     def classify_image(self, instance):
#         glucose_values, device_type = self.preprocess_and_extract(self.image_path)
#         if glucose_values and device_type:
#             result_text = [f"Device: {device_type}, Value: {value}" for value in glucose_values]
#             self.result_text.text = "\n".join(result_text)
#         else:
#             self.result_text.text = "Unable to detect values. Please try again with a clearer image."

#     def preprocess_and_extract(self, image_path):
#         reader = easyocr.Reader(['en'])
#         image = cv2.imread(image_path)
#         if image is None:
#             return [], None

#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#         kernel_sharpening = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
#         sharpened = cv2.filter2D(gray, -1, kernel_sharpening)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(sharpened)
#         _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         glucose_values = []
#         device_type = None

#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = w / float(h)
#             area = cv2.contourArea(contour)

#             if 0.5 < aspect_ratio < 2.0 and area > 1000:
#                 roi = image_rgb[y:y+h, x:x+w]
#                 results = reader.readtext(roi)

#                 for (box, text, prob) in results:
#                     numeric_text = ''.join(c for c in text if c.isdigit() or c == '.')
#                     try:
#                         value = float(numeric_text)
#                         if 20 <= value <= 600:
#                             if device_type is None:
#                                 device_type = self.classify_device(image_path)
#                             glucose_values.append(value)
#                             break
#                     except ValueError:
#                         continue

#         return glucose_values, device_type

#     def classify_device(self, image_path):
#         CLIENT = InferenceHTTPClient(
#             api_url="https://detect.roboflow.com",
#             api_key="EJSq61e3dlQXnJ0sOCAA"
#         )
#         result = CLIENT.infer(image_path, model_id="medical_device_classification/3")
#         return result['predicted_classes']

#     def save_data(self, instance):
#         # Implement saving functionality (e.g., save to CSV)
#         pass

#     def clear_data(self, instance):
#         # Implement clearing functionality
#         pass

# if __name__ == '__main__':
#     HealthcareApp().run()
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout as PopupBoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import easyocr
import os
from inference_sdk import InferenceHTTPClient

class HealthcareApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        self.filechooser.bind(on_selection=self.load_image)
        self.layout.add_widget(self.filechooser)
        
        self.image_display = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.image_display)
        
        self.result_label = Label(text="Upload an image to extract values.")
        self.layout.add_widget(self.result_label)
        
        self.classify_button = Button(text="Classify Image")
        self.classify_button.bind(on_press=self.classify_image)
        self.layout.add_widget(self.classify_button)
        
        self.result_text = TextInput(size_hint_y=None, height=200, readonly=True, text="Results will be shown here.")
        self.layout.add_widget(self.result_text)
        
        self.save_button = Button(text="Save Data")
        self.save_button.bind(on_press=self.save_data)
        self.layout.add_widget(self.save_button)
        
        self.clear_button = Button(text="Clear Data")
        self.clear_button.bind(on_press=self.clear_data)
        self.layout.add_widget(self.clear_button)
        
        self.data_file = "all_device_values.csv"
        self.all_device_values = pd.DataFrame(columns=['Image'] + self.get_classes())

        return self.layout

    def get_classes(self):
        return ['blood pressure set', 'breast pump', 'commode', 'crutch',
                'glucometer', 'oximeter', 'rippled mattress',
                'therapeutic ultrasound machine', 'thermometer']

    def load_image(self, instance, value):
        if value:
            self.image_path = value[0]
            pil_image = PILImage.open(self.image_path)
            self.image_display.texture = self.pil_image_to_texture(pil_image)
            self.result_label.text = "Image loaded. Press 'Classify Image' to analyze."

    def pil_image_to_texture(self, pil_image):
        pil_image = pil_image.convert('RGB')
        data = np.array(pil_image)
        texture = Texture.create(size=(data.shape[1], data.shape[0]), colorfmt='rgb')
        texture.blit_buffer(data.flatten(), colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def classify_image(self, instance):
        if not hasattr(self, 'image_path'):
            self.show_popup("Error", "No image selected. Please upload an image first.")
            return

        glucose_values, device_type = self.preprocess_and_extract(self.image_path)
        if glucose_values and device_type:
            result_text = [f"Device: {device_type}, Value: {value}" for value in glucose_values]
            self.result_text.text = "\n".join(result_text)
        else:
            self.result_text.text = "Unable to detect values. Please try again with a clearer image."

    def preprocess_and_extract(self, image_path):
        reader = easyocr.Reader(['en'])
        image = cv2.imread(image_path)
        if image is None:
            self.show_popup("Error", "Failed to read the image.")
            return [], None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kernel_sharpening = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpening)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        glucose_values = []
        device_type = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)

            if 0.5 < aspect_ratio < 2.0 and area > 1000:
                roi = image_rgb[y:y+h, x:x+w]
                results = reader.readtext(roi)

                for (box, text, prob) in results:
                    numeric_text = ''.join(c for c in text if c.isdigit() or c == '.')
                    try:
                        value = float(numeric_text)
                        if 20 <= value <= 600:
                            if device_type is None:
                                device_type = self.classify_device(image_path)
                            glucose_values.append(value)
                            break
                    except ValueError:
                        continue

        return glucose_values, device_type

    def classify_device(self, image_path):
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="EJSq61e3dlQXnJ0sOCAA"
        )
        result = CLIENT.infer(image_path, model_id="medical_device_classification/3")
        return result['predicted_classes']

    def save_data(self, instance):
        try:
            self.all_device_values.to_csv(self.data_file, index=False)
            self.show_popup("Success", f"Data saved to {self.data_file}")
        except Exception as e:
            self.show_popup("Error", f"Failed to save data: {e}")

    def clear_data(self, instance):
        self.all_device_values = pd.DataFrame(columns=['Image'] + self.get_classes())
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        self.result_text.text = "Data cleared."
        self.show_popup("Success", "All data cleared successfully.")

    def show_popup(self, title, message):
        layout = PopupBoxLayout(orientation='vertical')
        popup_label = Label(text=message, size_hint_y=0.8)
        close_button = Button(text="Close", size_hint_y=0.2)
        close_button.bind(on_press=self.dismiss_popup)
        layout.add_widget(popup_label)
        layout.add_widget(close_button)
        self.popup = Popup(title=title, content=layout, size_hint=(0.8, 0.5))
        self.popup.open()

    def dismiss_popup(self, instance):
        self.popup.dismiss()

if __name__ == '__main__':
    HealthcareApp().run()
