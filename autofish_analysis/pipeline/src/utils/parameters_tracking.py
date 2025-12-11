import inspect
import json
import datetime
import os
import subprocess
from pathlib import Path

class Parameter_tracking:
    def collect_constants(self):
        # global constants
        constants = {}
        frame = inspect.currentframe()
        while frame:
            for k, v in frame.f_locals.items():
                v = self.convert_to_string_if_path(v)
                if (
                    not k.startswith("_")
                    and isinstance(v, (int, float, str, bool, Path, list))
                    and k.isupper()
                ):
                    constants[k] = v
            frame = frame.f_back            
        return constants

    def save_constants_and_commit_hash(self, constants: dict, file_name: str, folder_path =None):
        """Saves a dictionary of constants to a JSON file.

        Args:
            constants: A dictionary containing the constants to be saved.
            FILE_NAME: the file name is the name of the experiment without the .tif extension.
            The files will always be stored in '../Analysis/', with the date in format
            yymmdd and the file name.
        """
        constants["git_hash"] = self.get_last_commit_hash()
        if folder_path is None:
            file_path = Path("../Analysis/"  + file_name + ".json") 
        else:
            file_path = Path(folder_path) / Path(file_name + ".json")
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        #with open(file_path, "w") as f:
        #    json.dump(constants, f, indent=4)

        self.update_json_file(file_path, constants)

    def update_json_file(self, file_path, new_data):
        """Updates an existing JSON file with new data.

        Args:
            file_path: The path to the JSON file.
            new_data: A dictionary containing the data to add or update.  
                    If keys already exist in the JSON file, their values will be updated.
                    If keys are new, they will be added.
        """
        try:
            if os.path.exists(file_path):   
                with open(file_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: File {file_path} is not valid JSON. Creating a new JSON object.")
                        existing_data = {}   
            else:
                existing_data = {} 
                print(f"Warning: File {file_path} does not exist. Creating a new file.")
            
            new_data['DATE'] = self.get_yyyymmdd_date()
            existing_data.update(new_data)  # .update merges dictionaries
            copy_data = existing_data.copy()
            for key, val in copy_data.items():
                if isinstance(copy_data[key], list):
                    if len(copy_data[key]):
                        for ind, u in enumerate(copy_data[key]):
                            if isinstance(u, Path):
                                copy_data[key][ind] = str(u)
            
            with open(file_path, "w") as f:
                json.dump(copy_data, f, indent=4)
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_yymmdd_date(self):
        """Returns the current date in the format 'yymmdd'.

        Returns:
            A string representing the current date in the format 'yymmdd'.
        """
        today = datetime.date.today()
        return today.strftime("%y%m%d")

    def get_yyyymmdd_date(self):
        """Returns the current date in the format 'yyyymmdd'.

        Returns:
            A string representing the current date in the format 'yyyymmdd'.
        """
        today = datetime.date.today()
        return today.strftime("%Y%m%d")



    def save_and_commit_conda_env(self):
        """Saves the current conda environment to a YAML file and commits it to Git,
        in order to have every time the good hash associated with the analysis.
        """
        export_command = ["rm", "../../*.yml"]
        subprocess.run(export_command, check=True)

        export_command = [
            "conda",
            "env",
            "export",
            "-f",
            "../../environment" + self.get_yyyymmdd_date() + ".yml",
        ]
        subprocess.run(export_command, check=True)

        export_command = ["cd", ".."]
        subprocess.run(export_command, check=True)

        export_command = ["cd", ".."]
        subprocess.run(export_command, check=True)

        git_add_command = ["git", "add", "*.yml"]
        subprocess.run(git_add_command, check=True)

        # git_commit_message = "Update conda environment"
        # git_commit_command = ["git", "commit", "-m", git_commit_message]
        # subprocess.run(git_commit_command, check=True)

        git_commit_command = f"git commit -m 'UpdateCondaEnvironment'"
        subprocess.run(git_commit_command.split(), check=True)

        git_commit_command = ["git", "push"]
        subprocess.run(git_commit_command, check=True)

    def get_last_commit_hash(self):
        """Gets the message of the last Git commit hash.

        Returns:
            str: The last commit hash.
        """
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H"], capture_output=True, text=True
        )
        if isinstance(result.stdout, str):
            return result.stdout.strip()
        else:
            return ""
        
        
    def convert_to_string_if_path(self, variable):
        """Converts a variable to a string if it's a Path object.

        Args:
            variable: The variable to check and potentially convert.

        Returns:
            The variable as a string if it was a Path, otherwise the 
            original variable.
        """
        if isinstance(variable, Path):
            return str(variable)
        return variable    
     
    
    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data