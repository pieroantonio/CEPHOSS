from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from joblib import load
import pandas as pd
import numpy as np

def carica_modello(nome_file):
    return load(nome_file)

class PrevisioniBoxLayout(BoxLayout):
    def __init__(self, initial_input_features, prediction_sequence, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.initial_input_features = initial_input_features
        self.prediction_sequence = prediction_sequence
        self.inputs = {}
        self.models = {feature: carica_modello(f'modello_{feature}.joblib') for feature in self.prediction_sequence}

        for feature in self.initial_input_features:
            row = BoxLayout(orientation="horizontal", size_hint_y=None, height=30)
            row.add_widget(Label(text=f"{feature}: ", size_hint_x=None, width=100))
            input_box = TextInput(multiline=False)
            row.add_widget(input_box)
            self.inputs[feature] = input_box
            self.add_widget(row)

        predict_button = Button(text="Predici", size_hint_y=None, height=50)
        predict_button.bind(on_press=self.on_predici_button_clicked)
        self.add_widget(predict_button)

        reset_button = Button(text="Reset", size_hint_y=None, height=50)
        reset_button.bind(on_press=self.on_reset_button_clicked)
        self.add_widget(reset_button)

    def on_predici_button_clicked(self, instance):
        try:
            input_data = {feature: float(self.inputs[feature].text) if self.inputs[feature].text else np.nan for feature in self.initial_input_features}
            predictions = self.effettua_predizione_in_catena(input_data)
            predictions_str = "\n".join([f"{k}: {np.round(v, 2)}" for k, v in predictions.items() if pd.notnull(v)])
            scroll_content = Label(text=predictions_str, size_hint_y=None)
            scroll_content.bind(texture_size=scroll_content.setter('size'))
            content = ScrollView(size_hint=(None, None), size=(400, 400))
            content.add_widget(scroll_content)
            popup = Popup(title="Previsioni", content=content, size_hint=(None, None), size=(500, 500))
            popup.open()
        except ValueError as e:
            popup = Popup(title="Errore", content=Label(text="Inserire valori numerici validi."), size_hint=(None, None), size=(400, 200))
            popup.open()

    def effettua_predizione_in_catena(self, input_data):
        input_df = pd.DataFrame([input_data])
        for feature in self.prediction_sequence:
            if pd.isnull(input_data.get(feature)):
                model = self.models[feature]
                prediction = model.predict(input_df)[0]
                input_data[feature] = prediction
                input_df.at[0, feature] = prediction
        return input_data

    def on_reset_button_clicked(self, instance):
        for input_box in self.inputs.values():
            input_box.text = ''  # Clear the text in each input box

class PrevisioniApp(App):
    def build(self):
        initial_input_features = ['Sex', 'Age', 'WITSA', 'OB_FH', 'OJ_OC', 'U6L6D', 'LFHNP', 'IIANG', 'IMPA']
        prediction_sequence = ['PLHNP', 'SNB', 'SNA', 'CONVX', 'COPOD', 'SADLA', 'SNFHA', 'SNDST', 'COPAD', 'TFHNP', 'FMPA', 'NP2PA', 'ANB', 'U1SNA', 'NP2PO', 'AB2FH']
        return PrevisioniBoxLayout(initial_input_features, prediction_sequence)

if __name__ == "__main__":
    PrevisioniApp().run()
