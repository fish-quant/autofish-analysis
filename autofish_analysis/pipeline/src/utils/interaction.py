import os


class UserInteraction:
    def ask_well_not_xpo1(self, constants: dict) -> str:
        """Prompts the user to give the well to observe (not a xpo1 well)."""
        wells = constants["MODALITIES"]
        xpo1_wells = constants[f"WELLS_XPO1"]
        wells_not_xpo1 = [el for el in wells if el not in xpo1_wells]

        print(f"Available wells: {', '.join(wells_not_xpo1)}")
        while True:
            try:
                user_input = input(f"Enter the well you want to visualize. ")
                if user_input in wells_not_xpo1:
                    return user_input
            except ValueError:
                print("Invalid input. Please enter valid well.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_wells_not_xpo1(self, constants: dict) -> list:
        """Prompts the user to give the well to observe (not a xpo1 well)."""
        wells = constants["MODALITIES"]
        xpo1_wells = constants[f"WELLS_XPO1"]
        wells_not_xpo1 = [el for el in wells if el not in xpo1_wells]

        print(f"Available wells: {', '.join(wells_not_xpo1)}")

        output_wells_list = []
        while True:
            try:
                user_input = input(f"""Enter the next well you want to include.
                Write stop to spot the list.
                Write all to include them all    """)

                if user_input in wells_not_xpo1 and user_input not in output_wells_list:
                    output_wells_list.append(user_input)
                    print(f"current list: {' , '.join(output_wells_list)}")
                if user_input == "stop":
                    return output_wells_list
                if user_input == "all":
                    return wells_not_xpo1.copy()
            except ValueError:
                print("Invalid input. Please enter valid well.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_list(self, constants: dict, keyword: str, text=None) -> str:
        """Prompts the user to choose among a list of strings, this list
        belongs to constants, and it is called keyword.
        """
        list_op = constants[keyword]
        while True:
            try:
                user_input = input(f"Choose {text} among: {list_op}. ")
                if user_input in list_op:
                    return user_input
            except ValueError:
                print(f"Invalid input. Please enter valid {text}.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_channel(self, constants: dict, text: None) -> str:
        """Prompts the user to give the right channel to which apply the thresholding."""
        while True:
            try:
                user_input = input(
                    f"Enter the right channel among: {' '.join(constants['CHANNELS'])}, {text} :   "
                ).strip()
                if user_input in constants["CHANNELS"]:
                    return user_input
            except ValueError:
                print("Invalid input. Please enter valid channel.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_xpo1_wells(self, wells: list):
        entered_wells = []
        while True:
            well_input = (
                input(
                    f"Enter XPO1 well name (e.g., {', '.join(wells)}) or 'stop' to finish: "
                )
                .strip()
                .upper()
            )
            if well_input == "STOP":
                print("Stopping well input.")
                break
            elif well_input in wells:
                entered_wells.append(well_input)
                print(
                    f"Well '{well_input}' added. Current wells: {', '.join(entered_wells)}"
                )
            else:
                print(
                    f"Invalid well name '{well_input}'. Please enter one of {wells} or 'stop'."
                )
        print(
            f"Finished entering XPO1 wells. Collected wells: {', '.join(entered_wells)}"
        )
        return entered_wells

    def choose_option(self, list_options) -> str:
        while True:
            try:
                user_input = input(f"Choose among: {list_options}. ")
                if user_input in list_options:
                    return user_input
            except ValueError:
                print(f"Invalid input. Please enter valid option.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def choose_options(self, list_options) -> list:
        output_list = []
        while True:
            try:
                user_input = input(
                    f"Choose among: {list_options}. To stop, input: stop. To choose all: all   "
                )
                if user_input in list_options and user_input not in output_list:
                    output_list.append(user_input)
                elif user_input == "stop":
                    return output_list
                elif user_input == "all":
                    return list_options.copy()
            except ValueError:
                print(f"Invalid input. Please enter valid option.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_value(self, default_val=1000, text=" "):
        while True:
            try:
                user_input = input(text + f"default value  {default_val}     ")
                user_input = float(user_input)
                return user_input
            except ValueError:
                print(f"Invalid input. Please enter valid option.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def ask_yes_no(self, text=" "):
        while True:
            try:
                user_input = input(text + f": enter yes or no.    ")
                if user_input == "yes":
                    return True
                elif user_input == "no":
                    return False
            except ValueError:
                print(f"Invalid input. Please enter valid option.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def get_num_workers(self):
        max_cores = os.cpu_count()
        if max_cores is None:
            print("Could not determine the number of CPU cores. Defaulting to 1.")
            return 1
        while True:
            prompt = (
                f"\nYour system has {max_cores} available CPU cores.\n"
                f"How many workers would you like to use for pandarallel? (Recommended: {max(1, max_cores - 2)} or less)\n"
                f"Please enter a number between 1 and {max_cores}: "
            )
            try:
                selected_cores_str = input(prompt)
                selected_cores = int(selected_cores_str)
                if 1 <= selected_cores <= max_cores:
                    print(f"Using {selected_cores} workers for parallel processing.")
                    return selected_cores
                else:
                    print(f"Enter a number between 1 and {max_cores}.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")
