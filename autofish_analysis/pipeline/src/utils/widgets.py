import ipywidgets as widgets
from IPython.display import display
import numpy as np
        
class StringListWidget:
    def __init__(self):
        # Create a text input widget
        self.text_input = widgets.Text(
            value='',
            placeholder='Type something',
            description='Input:',
            disabled=False
        )
        # Create a button widget to add to the list
        self.add_button = widgets.Button(
            description='Add to list',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to add the input to the list'
        )

        # Create a button widget to delete the list
        self.delete_button = widgets.Button(
            description='Delete list',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to delete the list'
        )
        # Create an output widget to display the list
        self.output = widgets.Output()

        # Initialize an empty list to store the strings
        self.string_list = []

        # Link the button click events to the functions
        self.add_button.on_click(self.handle_add_button_click)
        self.delete_button.on_click(self.handle_delete_button_click)

    def handle_add_button_click(self, b):
        self.string_list.append(self.text_input.value)
        self.text_input.value = ''  # Clear the input field
        self.update_output()

    def handle_delete_button_click(self, b):
        self.string_list = []  # Clear the list
        self.update_output()

    def update_output(self):
        with self.output:
            self.output.clear_output()
            display(self.string_list)

    def display(self):
        display(self.text_input, self.add_button, self.delete_button, self.output)


class IntegerInputWidget:
    def __init__(self, title= 'Input:'):
        # Create an integer input widget
        self.input = widgets.IntText(
            value=0,
            description= title,
            disabled=False)
        
        # Create a button widget to submit the input
        self.submit_button = widgets.Button(
            description='Submit',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to submit the input')
        
        # Create an output widget to display the input
        self.output = widgets.Output()
        
        # Link the button click event to the function
        self.submit_button.on_click(self.handle_submit_button_click)
    
    def handle_submit_button_click(self, b):
        with self.output:
            self.output.clear_output()
            display(self.input.value)
    
    def display(self):
        display(self.input, self.submit_button, self.output)
       