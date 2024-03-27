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

    def __init__(self, input_features, output_features, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.modello = {feature: carica_modello(f'modello_migliore_{feature}.joblib') for feature in output_features}
        self.input_features = input_features
        self.output_features = output_features
        self.inputs = {}
        
        for feature in self.input_features:
            row = BoxLayout(orientation="horizontal", size_hint_y=None, height=30)
            row.add_widget(Label(text=f"{feature}: ", size_hint_x=None, width=100))
            input_box = TextInput(multiline=False)
            row.add_widget(input_box)
            self.inputs[feature] = input_box
            self.add_widget(row)

        predict_button = Button(text="Predici", size_hint_y=None, height=50)
        predict_button.bind(on_press=self.on_predici_button_clicked)
        self.add_widget(predict_button)

    def on_predici_button_clicked(self, instance):
        try:
            input_values = [float(self.inputs[feature].text) for feature in self.input_features]
            previsioni = self.effettua_predizione(input_values)
            # Arrotonda le previsioni al primo decimale e crea una stringa per il Popup
            previsioni_str = "\n".join([f"{k}: {np.round(v, 1)}" for k, v in previsioni.items()])
            scroll_content = Label(text=previsioni_str, size_hint_y=None)
            scroll_content.bind(texture_size=scroll_content.setter('size'))
            content = ScrollView(size_hint=(None, None), size=(400, 400))
            content.add_widget(scroll_content)
            popup = Popup(title="Previsioni", content=content, size_hint=(None, None), size=(500, 500))
            popup.open()
        except ValueError as e:
            popup = Popup(title="Errore", content=Label(text="Inserire valori numerici validi."), size_hint=(None, None), size=(400, 200))
            popup.open()

    def effettua_predizione(self, input_values):
        dati_nuovi_df = pd.DataFrame([input_values], columns=self.input_features)
        previsioni = {}
        for feature, model in self.modello.items():
            previsione = model.predict(dati_nuovi_df)[0]
            previsioni[feature] = previsione
        return previsioni

class PrevisioniApp(App):
    def build(self):
        input_features = ['Sex', 'Age', 'WITSA', 'OB_FH', 'OJ_OC', 'U6L6D', 'LFHNP', 'IIANG', 'IMPA']
        output_features = ['SNA', 'SNB', 'ANB', 'NP2PA', 'NP2PO', 'SNDST', 'SNFHA', 'CONVX', 'AB2FH', 'TFHNP', 'PLHNP', 'U1SNA', 'FMPA', 'SADLA', 'COPAD', 'COPOD']
        return PrevisioniBoxLayout(input_features, output_features)

if __name__ == "__main__":
    PrevisioniApp().run()
